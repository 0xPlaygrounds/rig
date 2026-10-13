use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::CommandPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
    tui_view,
};
use rig_tui::TuiInputState;

use super::*;

/// What the pickers were answered with.
#[derive(Resource, Default)]
struct Picked(Vec<Option<Chosen>>);

/// A `wait` that never ends, keeping its turn running.
fn wait(_: On<ToolCalled<serde_json::Value>>) {}

/// Presses each of `keys` (as `ctrl+j` or `esc`), a frame each, as a person
/// types.
fn press(app: &mut App, keys: &str) {
    for key in keys.split_whitespace() {
        rig_harness_test_support::press(app, key);
    }
}

/// Types `text`, a key and a frame for each character.
fn type_text(app: &mut App, text: &str) {
    for character in text.chars() {
        let key = match character {
            ' ' => "space".to_owned(),
            other => other.to_string(),
        };
        press(app, &key);
    }
}

fn input(app: &App) -> String {
    app.world().resource::<TuiInput>().text().to_owned()
}

fn status(app: &App) -> String {
    let status = &app.world().resource::<AppStatus>().0;
    status.iter().map(|item| item.text.clone()).collect()
}

#[test]
fn edits_the_input_as_vim_does_and_leaves_other_keys_alone() {
    let model = MockCompletionModel::from_stream_turns([
        reply("Noted."),
        calls("c1", "wait", serde_json::json!({})),
    ]);
    let home = scratch_home("sn-vim-input");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, JournalPlugin, CommandPlugin, VimPlugin))
        .add_open_tool("wait", "Waits.", ToolOptions::default(), wait)
        .init_resource::<Ended>()
        .init_resource::<Picked>()
        .add_observer(log_ended)
        .add_observer(|pick: On<Pick>, mut picked: ResMut<Picked>| {
            picked.0.push(pick.chosen.clone());
        });
    tui_view(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    app.insert_resource(Shown(Some(agent)));
    app.update();

    // Insert mode types; Esc, `0`, `dw`, `A!`, Esc, Enter sends `world!`,
    // every key in one frame.
    assert_eq!(status(&app), "-- INSERT --");
    rig_harness_test_support::press(
        &mut app,
        "h e l l o space w o r l d esc 0 d w A ! esc enter",
    );
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let requests = model.requests();
    let asked = requests
        .first()
        .map(|request| format!("{:?}", request.chat_history));
    assert!(
        asked
            .as_ref()
            .is_some_and(|asked| asked.contains("world!") && !asked.contains("hello")),
        "{asked:?}"
    );
    assert_eq!(
        status(&app),
        "-- INSERT --",
        "back to the start mode once sent"
    );

    // `gg2dd` deletes two lines; `u` brings them back, Ctrl-r takes them
    // again.
    type_text(&mut app, "a");
    press(&mut app, "ctrl+j");
    type_text(&mut app, "b");
    press(&mut app, "ctrl+j");
    type_text(&mut app, "c");
    press(&mut app, "esc");
    type_text(&mut app, "gg2dd");
    assert_eq!(input(&app), "c");
    type_text(&mut app, "u");
    assert_eq!(input(&app), "a\nb\nc");
    press(&mut app, "ctrl+r");
    assert_eq!(input(&app), "c");

    // `v$y$p` on `abc` doubles it.
    type_text(&mut app, "cc");
    type_text(&mut app, "abc");
    press(&mut app, "esc");
    type_text(&mut app, "0v");
    assert_eq!(status(&app), "-- VISUAL --");
    type_text(&mut app, "$y$p");
    assert_eq!(input(&app), "abcabc");
    assert_eq!(status(&app), "-- NORMAL --");

    // An open picker takes `j` and Enter: `j` filters it to `jam`, Enter
    // picks that; the input and its mode stay.
    let picker = Picker::new("Pick one").item("alpha", "a").item("jam", "j");
    let picker = app.world_mut().spawn(picker).id();
    app.update();
    app.update();
    let picking = app.world().resource::<TuiInputState>().picking();
    assert_eq!(picking, Some(picker));
    press(&mut app, "j enter");
    let picked = app.world().resource::<Picked>().0.clone();
    assert_eq!(picked, vec![Some(Chosen::Item(1))]);
    assert_eq!(input(&app), "abcabc");
    assert_eq!(status(&app), "-- NORMAL --");
    app.world_mut().despawn(picker);
    app.update();

    // With the agent busy, Esc in insert mode only leaves insert mode.
    type_text(&mut app, "ccwait");
    press(&mut app, "enter");
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && world.get::<ActiveTurn>(agent).is_some()
    });
    press(&mut app, "esc");
    assert_eq!(status(&app), "-- NORMAL --");
    app.update();
    assert!(
        app.world().get::<ActiveTurn>(agent).is_some(),
        "Esc stopped the turn"
    );
    // Twice in normal mode with nothing typed, it stops the agent as Esc
    // does without the plugin.
    press(&mut app, "esc esc");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    assert!(
        app.world().get::<ActiveTurn>(agent).is_none(),
        "the turn still runs"
    );

    // `/vim off` is kept for the next session, and frees the keys.
    app.world_mut().trigger(RunCommand::new(agent, "vim off"));
    app.update();
    app.update();
    assert_eq!(status(&app), "");
    let saved = app.world().resource::<PluginData>().user.read_file(SAVED);
    assert_eq!(saved.ok(), Some(br#"{"enabled":false}"#.to_vec()));
    type_text(&mut app, "x");
    assert_eq!(input(&app), "x", "typed as without the plugin");
}

#[test]
fn the_demo_fixes_a_word_three_back() {
    let mut editor = editor::Editor::new(Mode::Insert);
    let mut buffer = Buffer {
        text: "fix the parser bug in lexer".to_owned(),
        cursor: 27,
    };
    for key in "\u{1b}3bcwtokenizer\u{1b}".chars() {
        let key = if key == '\u{1b}' {
            Key::Esc
        } else {
            Key::Char(key)
        };
        editor.key(key, &mut buffer);
    }
    assert_eq!(buffer.text, "fix the parser tokenizer in lexer");
}
