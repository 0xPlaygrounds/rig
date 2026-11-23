use rig_core::completion::Message;
use rig_ecs::agent::{StopTurn, TurnOf, TurnOutcome};
use rig_ecs::prelude::Entity;
use rig_ecs::spawn::{AgentSpec, Origin, spawn};

use bevy_app::App;

use super::{POP_TITLE, PUSH_TITLE, TerminalTitlePlugin};
use crate::TuiTestScreen;
use crate::tests::{Delivered, press, type_text, view_app, view_app_with};

/// The marks (without a nonce) and titles written since the last call, in
/// order, as `133;C` or `2;rig · idle`.
fn told(app: &mut bevy_app::App, seen: &mut usize) -> Vec<String> {
    let written = app.world().resource::<TuiTestScreen>().written();
    let text = String::from_utf8_lossy(written.get(*seen..).unwrap_or_default()).into_owned();
    *seen = written.len();
    text.split("\x1b]")
        .skip(1)
        .filter_map(|osc| osc.split_once("\x1b\\").map(|(osc, _)| osc))
        .map(|osc| osc.split(";nonce=").next().unwrap_or_default().to_owned())
        .collect()
}

/// The view of [`view_app`] with the [`TerminalTitlePlugin`].
fn titled_app() -> (App, Entity) {
    view_app_with(|app| {
        app.add_plugins(TerminalTitlePlugin);
    })
}

const IDLE: [&str; 3] = ["133;A", "133;B", "133;S;activity=reading"];
const BUSY: [&str; 2] = ["133;C", "133;S;activity=busy"];

#[test]
fn the_terminal_is_told_of_each_turn_of_the_main_agent_only() {
    let (mut app, agent) = titled_app();
    // The first frame's writes, kept for two frames, reach the screen now.
    app.insert_resource(TuiTestScreen::new(40, 10));
    let mut seen = 0;
    app.update();
    let written = app.world().resource::<TuiTestScreen>().written();
    let pushed = [PUSH_TITLE, b"\x1b]2;"].concat();
    let first = written
        .windows(pushed.len())
        .position(|bytes| bytes == pushed);
    assert!(
        first.is_some(),
        "the title is pushed before the first: {written:?}"
    );
    assert_eq!(
        told(&mut app, &mut seen),
        [&IDLE[..], &["2;rig · idle"]].concat()
    );

    // A turn runs, then fails: C, then D with status 1 and the prompt.
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    app.update();
    assert_eq!(
        told(&mut app, &mut seen),
        [&BUSY[..], &["2;rig · working"]].concat()
    );
    app.update();
    assert_eq!(told(&mut app, &mut seen), Vec::<String>::new());
    let failed = TurnOutcome::Failed("no model".to_owned());
    app.world_mut().entity_mut(turn).insert(failed);
    app.update();
    assert_eq!(
        told(&mut app, &mut seen),
        [&["133;D;1"], &IDLE[..], &["2;rig · error"]].concat()
    );

    // A subagent's turn tells nothing.
    let helper = spawn(
        app.world_mut(),
        Origin::ChildOf(agent),
        AgentSpec::default(),
    );
    let helper = helper.unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().spawn(TurnOf(helper));
    app.update();
    assert_eq!(told(&mut app, &mut seen), Vec::<String>::new());

    // A turn that started and answered within a frame still gets both
    // marks.
    let answered = TurnOutcome::Answered(Message::assistant("Done."));
    app.world_mut().spawn((TurnOf(agent), answered));
    app.update();
    let marks = [&BUSY[..], &["133;D;0"], &IDLE[..], &["2;rig · idle"]].concat();
    assert_eq!(told(&mut app, &mut seen), marks);

    // Only a failed turn is an error: a stop, the user's or a
    // compaction's, and a turn gone without an outcome are not.
    let failed = TurnOutcome::Failed("refused".to_owned());
    app.world_mut().spawn((TurnOf(agent), failed));
    app.update();
    told(&mut app, &mut seen);
    let compacted = StopTurn::plugin_alone(agent, "compaction", "compacted");
    for stop in [StopTurn::new(agent), compacted] {
        app.world_mut().spawn(TurnOf(agent));
        app.world_mut().trigger(stop);
        app.update();
        let marks = [&BUSY[..], &["133;D;0"], &IDLE[..], &["2;rig · idle"]].concat();
        assert_eq!(told(&mut app, &mut seen), marks);
        app.world_mut()
            .spawn((TurnOf(agent), TurnOutcome::Failed("refused".to_owned())));
        app.update();
        told(&mut app, &mut seen);
    }
    let gone = app.world_mut().spawn(TurnOf(agent)).id();
    app.update();
    told(&mut app, &mut seen);
    app.world_mut().despawn(gone);
    app.update();
    let marks = [&["133;D;0"], &IDLE[..], &["2;rig · idle"]].concat();
    assert_eq!(told(&mut app, &mut seen), marks);

    // The title goes back on exit.
    app.world_mut().write_message(bevy_app::AppExit::Success);
    app.update();
    let written = app.world().resource::<TuiTestScreen>().written();
    assert!(written.ends_with(POP_TITLE), "{written:?}");
}

/// Whether `bytes` touch the title (OSC 0 or 2, or a push or pop of it),
/// and whether they hold a prompt mark (OSC 133).
fn tells(bytes: &[u8]) -> (bool, bool) {
    let holds = |osc: &[u8]| bytes.windows(osc.len()).any(|seen| seen == osc);
    let title = [&b"\x1b]0;"[..], b"\x1b]2;", PUSH_TITLE, POP_TITLE];
    (title.into_iter().any(holds), holds(b"\x1b]133;"))
}

#[test]
fn the_title_and_the_marks_are_the_title_plugins() {
    let apps: [(bool, fn() -> (App, Entity)); 2] = [(false, view_app), (true, titled_app)];
    for (titled, app) in apps {
        let (mut app, agent) = app();
        app.insert_resource(TuiTestScreen::new(40, 10));
        app.update();
        type_text(&mut app, "hello");
        press(&mut app, "enter");
        assert_eq!(app.world().resource::<Delivered>().0, ["hello"]);
        let turn = app.world_mut().spawn(TurnOf(agent)).id();
        app.update();
        let answered = TurnOutcome::Answered(Message::assistant("Done."));
        app.world_mut().entity_mut(turn).insert(answered);
        app.update();
        app.world_mut().write_message(bevy_app::AppExit::Success);
        app.update();
        let written = app.world().resource::<TuiTestScreen>().written();
        let text = String::from_utf8_lossy(written);
        assert_eq!(tells(written), (titled, titled), "{text:?}");
    }
}

#[test]
fn the_marks_carry_the_nonce_title_mark_nonce_env_names() {
    let settings = "[title]\nmark_nonce_env = \"CARGO_PKG_NAME\"\n";
    let (mut app, _) = view_app_with(|app| {
        app.insert_resource(rig_harness::settings::Settings::from_toml(settings))
            .add_plugins(TerminalTitlePlugin);
    });
    app.insert_resource(TuiTestScreen::new(40, 10));
    app.update();
    let written = app.world().resource::<TuiTestScreen>().written();
    let mark = format!("\x1b]133;A;nonce={}\x1b\\", env!("CARGO_PKG_NAME"));
    let text = String::from_utf8_lossy(written);
    assert!(text.contains(&mark), "{text:?}");
}

#[test]
fn without_the_terminal_view_it_adds_nothing() {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::default())
        .add_plugins(TerminalTitlePlugin);
    app.finish();
    app.update();
    assert!(!app.world().contains_resource::<super::Told>());
}

#[test]
fn added_before_the_view_it_tells_the_terminal() {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::default())
        .insert_resource(rig_harness::prelude::Invoked::default())
        .add_plugins((TerminalTitlePlugin, crate::TuiViewPlugin));
    app.finish();
    assert!(app.world().contains_resource::<super::Told>());
}
