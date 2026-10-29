use bevy_app::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use rig_core::completion::Message;
use rig_core::conversation::{self, Author, Entry};
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::{Picked, Picker};
use rig_harness::rig_memory::MemoryStore;

use super::*;
use crate::tests::{edit, input, press, type_text, typed, view_app};
use crate::view::{ShowAgent, TuiView};
use crate::{InputSystems, TuiKeys};

/// Sends `prompt`, which comes back to the input with no model to answer
/// it, and empties the input.
fn send(app: &mut App, prompt: &str) {
    type_text(app, prompt);
    press(app, "enter");
    edit(app, |input| drop(input.take()));
}

/// A conversation of the typed `prompts`, each answered, among messages
/// the user did not type: a plugin's note, a compaction summary, an
/// attached file and a helper agent's request.
fn conversation(prompts: &[&str]) -> Conversation {
    let mut conversation = conversation::Conversation::new();
    conversation.push(Entry::new(Message::user("summary")).by(Author::plugin("compaction")));
    for prompt in prompts {
        conversation.push(Message::user(*prompt));
        conversation.push(Message::assistant("ok"));
    }
    conversation.push(Message::user("<file path=\"a.txt\">\nfile text\n</file>"));
    conversation.push(Entry::new(Message::user("helper asks")).by(Author::agent("helper", None)));
    conversation.push(Entry::new(Message::user("build failed")).by(Author::plugin("build")));
    Conversation(conversation)
}

/// The history's entries.
fn entries(app: &App) -> Vec<String> {
    app.world().resource::<PromptHistory>().entries.clone()
}

/// Whether a completion list is open.
fn listing(app: &App) -> bool {
    app.world().resource::<TuiView>().completion.is_some()
}

/// A first Up on typed text goes to its start, and only the next leaves it
/// for the history, which Down past the newest brings it back from.
#[test]
fn up_and_down_browse_the_history_a_plugin_supplies() {
    let (mut app, _) = view_app();
    app.world_mut().resource_mut::<PromptHistory>().entries = vec!["one".into(), "two".into()];
    type_text(&mut app, "first line");
    press(&mut app, "shift+enter");
    type_text(&mut app, "draft");
    press(&mut app, "up up");
    assert_eq!(
        (typed(&app).as_str(), input(&app).cursor()),
        ("first line\ndraft", 0)
    );
    press(&mut app, "up");
    assert_eq!(typed(&app), "two");
    press(&mut app, "up up");
    assert_eq!(typed(&app), "one");
    press(&mut app, "down down");
    assert_eq!(typed(&app), "first line\ndraft");
    assert_eq!(entries(&app), ["one", "two"]);
    // Down on the last line goes to the end of the text.
    press(&mut app, "ctrl+a down !");
    assert_eq!(typed(&app), "first line\ndraft!");
    // Ctrl+P goes to the history from anywhere on the first line.
    press(&mut app, "up ctrl+p");
    assert_eq!(typed(&app), "two");
}

/// A recalled slash command opens no list, so Up and Down walk past it both
/// ways; Tab makes it the person's to complete, and typing opens the list
/// as ever, which Up and Down then move in.
#[test]
fn up_and_down_walk_past_a_recalled_command_and_typing_still_completes() {
    let (mut app, _) = view_app();
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    app.add_command("more", "Show more", |_: In<CommandArgs>| {});
    app.world_mut().resource_mut::<PromptHistory>().entries =
        vec!["older".into(), "/mo".into(), "newer".into()];
    type_text(&mut app, "draft");
    press(&mut app, "up up up");
    assert_eq!((typed(&app).as_str(), listing(&app)), ("/mo", false));
    press(&mut app, "up");
    assert_eq!(typed(&app), "older");
    press(&mut app, "down");
    assert_eq!((typed(&app).as_str(), listing(&app)), ("/mo", false));
    press(&mut app, "down down");
    assert_eq!(typed(&app), "draft");
    press(&mut app, "up up up tab");
    assert_eq!((typed(&app).as_str(), listing(&app)), ("/mo", true));
    press(&mut app, "ctrl+c");
    type_text(&mut app, "/mo");
    assert!(listing(&app));
    press(&mut app, "down tab");
    assert_eq!(typed(&app), "/more ");
}

/// What the person typed and did not send goes to the history whenever
/// the input drops it, once, with secrets redacted, and blank text never.
#[test]
fn text_the_input_drops_unsent_goes_to_the_history() {
    let (mut app, _) = view_app();
    app.world_mut().resource_mut::<PromptHistory>().entries = vec!["old".into()];
    // A recalled prompt sent in place of the draft.
    type_text(&mut app, "draft one");
    press(&mut app, "ctrl+a up enter");
    edit(&mut app, |input| drop(input.take()));
    assert_eq!(entries(&app), ["draft one", "old"]);
    // Cleared.
    type_text(&mut app, "key sk-proj-abcdefghijklmnop1234");
    press(&mut app, "ctrl+c");
    assert_eq!(
        entries(&app).last().map(String::as_str),
        Some("key [REDACTED]")
    );
    // Cleared while a prompt is recalled: the draft behind it.
    type_text(&mut app, "draft cleared");
    press(&mut app, "ctrl+a up ctrl+c");
    assert_eq!(
        entries(&app).last().map(String::as_str),
        Some("draft cleared")
    );
    // Replaced by a plugin while a prompt is recalled.
    type_text(&mut app, "draft two");
    press(&mut app, "ctrl+a up");
    edit(&mut app, |input| input.set("a template"));
    app.update();
    assert_eq!(entries(&app).last().map(String::as_str), Some("draft two"));
    // A recalled prompt edited is the person's: the draft before it is
    // kept, and Up keeps the edit as the draft.
    edit(&mut app, |input| drop(input.take()));
    type_text(&mut app, "draft three");
    press(&mut app, "ctrl+a up");
    type_text(&mut app, "!");
    press(&mut app, "ctrl+a up");
    assert_eq!(typed(&app), "draft three");
    press(&mut app, "down");
    assert_eq!(typed(&app), "draft two!");
    // Blank text is not kept, and nothing twice: the last cleared is the
    // newest, also within a frame.
    press(&mut app, "ctrl+c space ctrl+c a ctrl+c b ctrl+c a ctrl+c");
    // A recalled prompt edited and cleared in one frame: the edit is the
    // newest, after the draft behind it.
    type_text(&mut app, "draft four");
    press(&mut app, "ctrl+a up");
    press(&mut app, "! ctrl+c");
    // A prompt sent, then another cleared, in one frame: in that order.
    press(&mut app, "x enter y ctrl+c");
    edit(&mut app, |input| drop(input.take()));
    let kept = [
        "draft one",
        "old",
        "key [REDACTED]",
        "draft cleared",
        "draft two",
        "draft three",
        "draft two!",
        "b",
        "a",
        "draft four",
        "a!",
        "x",
        "y",
    ];
    assert_eq!(entries(&app), kept);
}

#[test]
fn a_prompt_filtered_out_before_the_save_never_reaches_the_file() {
    let (mut app, _) = view_app();
    let files: Arc<dyn Files> = Arc::new(MemoryStore::default());
    app.world_mut().resource_mut::<PromptHistory>().file = Some(files.clone());
    app.add_systems(
        Update,
        (|mut history: ResMut<PromptHistory>| {
            history.entries.retain(|prompt| !prompt.contains("sk-"));
        })
        .after(HistorySystems::Record)
        .before(HistorySystems::Save),
    );
    for prompt in ["hello", "my key is sk-123", "bye"] {
        type_text(&mut app, prompt);
        press(&mut app, "enter");
        // No model answers it, so it comes back to the input.
        edit(&mut app, |input| drop(input.take()));
    }
    let saved = files.read_file(HISTORY_FILE).unwrap_or_default();
    assert_eq!(saved, b"\"hello\"\n\"bye\"\n");
    assert_eq!(
        app.world().resource::<PromptHistory>().entries,
        ["hello", "bye"]
    );
    // Read back by the next session.
    assert_eq!(PromptHistory::load(files).entries, ["hello", "bye"]);
}

#[test]
fn a_repeated_prompt_moves_to_the_newest_and_is_kept_once() {
    let (mut app, _) = view_app();
    let files: Arc<dyn Files> = Arc::new(MemoryStore::default());
    app.world_mut().resource_mut::<PromptHistory>().file = Some(files.clone());
    for prompt in ["fix it", "test it", "fix it"] {
        send(&mut app, prompt);
    }
    let history = app.world().resource::<PromptHistory>();
    assert_eq!(history.entries, ["test it", "fix it"]);
    assert_eq!(PromptHistory::load(files).entries, ["test it", "fix it"]);
}

#[test]
fn a_long_file_loads_the_newest_distinct_prompts_and_is_rewritten() {
    let files: Arc<dyn Files> = Arc::new(MemoryStore::default());
    // The last 500 lines repeat one prompt, so the distinct ones before
    // them are kept too.
    let lines: String = (0..700)
        .map(|n| format!("prompt {n}"))
        .chain(std::iter::repeat_n("prompt 699".to_owned(), 500))
        .map(|prompt| format!("{prompt:?}\n"))
        .chain(["not json\n".to_owned()])
        .collect();
    files.write_file(HISTORY_FILE, lines.as_bytes()).ok();
    let newest: Vec<String> = (200..700).map(|n| format!("prompt {n}")).collect();
    assert_eq!(PromptHistory::load(files.clone()).entries, newest);
    let rewritten = files.read_file(HISTORY_FILE).unwrap_or_default();
    assert_eq!(String::from_utf8_lossy(&rewritten).lines().count(), 500);
    assert_eq!(PromptHistory::load(files).entries, newest);
}

#[test]
fn the_history_is_the_projects() {
    let project: Arc<dyn Files> = Arc::new(MemoryStore::default());
    let user: Arc<dyn Files> = Arc::new(MemoryStore::default());
    project
        .write_file(HISTORY_FILE, b"\"in this project\"\n")
        .ok();
    user.write_file(HISTORY_FILE, b"\"in another project\"\n")
        .ok();
    let mut world = World::new();
    world.insert_resource(PluginData {
        session: Arc::new(MemoryStore::default()),
        project,
        user,
    });
    world.init_resource::<PromptHistory>();
    world.run_system_once(load).ok();
    assert_eq!(
        world.resource::<PromptHistory>().entries,
        ["in this project"]
    );
}

#[test]
fn a_resumed_conversations_prompts_come_first_filtered_and_are_not_saved_again() {
    let (mut app, agent) = view_app();
    let files: Arc<dyn Files> = Arc::new(MemoryStore::default());
    *app.world_mut().resource_mut::<PromptHistory>() = PromptHistory {
        entries: vec!["older".into(), "fix the build".into()],
        file: Some(files.clone()),
        unsaved: Vec::new(),
    };
    app.add_systems(
        TuiKeys,
        (|mut history: ResMut<PromptHistory>| {
            history.entries.retain(|prompt| !prompt.contains("sk-"));
        })
        .after(HistorySystems::Seed)
        .before(InputSystems::Dispatch),
    );
    let resumed = conversation(&["fix the build", "run the tests", "my key is sk-123"]);
    app.world_mut().entity_mut(agent).insert(resumed);
    for expected in ["run the tests", "fix the build", "older", "older"] {
        press(&mut app, "up");
        assert_eq!(typed(&app), expected);
    }
    assert!(files.read_file(HISTORY_FILE).is_err());
}

#[test]
fn showing_another_agent_seeds_its_prompts_and_keeps_the_draft() {
    let (mut app, _) = view_app();
    app.world_mut().resource_mut::<PromptHistory>().entries = vec!["older".into()];
    let other = app
        .world_mut()
        .spawn((Agent, conversation(&["review the diff"])))
        .id();
    // Picking the agent shows it, and the Up after it in the same frame
    // browses its prompts.
    app.add_observer(move |_: On<Picked>, mut commands: Commands| {
        commands.trigger(ShowAgent { entity: other });
    });
    type_text(&mut app, "draft");
    app.world_mut()
        .spawn(Picker::new("Agents").item("other", ""));
    app.update();
    press(&mut app, "enter ctrl+a up");
    assert_eq!(typed(&app), "review the diff");
    press(&mut app, "up");
    assert_eq!(typed(&app), "older");
    press(&mut app, "down down");
    assert_eq!(typed(&app), "draft");
}

#[test]
fn browsing_follows_the_prompt_shown_when_the_history_moves() {
    let (mut app, _) = view_app();
    app.world_mut().resource_mut::<PromptHistory>().entries =
        vec!["one".into(), "two".into(), "three".into()];
    type_text(&mut app, "draft");
    press(&mut app, "ctrl+a up up");
    assert_eq!(typed(&app), "two");
    app.world_mut()
        .resource_mut::<PromptHistory>()
        .seed(["one".to_owned()]);
    for expected in ["three", "one", "draft"] {
        press(&mut app, "down");
        assert_eq!(typed(&app), expected);
    }
}
