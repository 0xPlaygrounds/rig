use rig_core::completion::Message;
use rig_ecs::agent::{SpawnedBy, TurnOf, TurnOutcome};

use super::{POP_TITLE, PUSH_TITLE};
use crate::TuiTestScreen;
use crate::tests::view_app;

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

const IDLE: [&str; 3] = ["133;A", "133;B", "133;S;activity=reading"];
const BUSY: [&str; 2] = ["133;C", "133;S;activity=busy"];

#[test]
fn the_terminal_is_told_of_each_turn_of_the_main_agent_only() {
    let (mut app, agent) = view_app();
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
    let helper = app.world_mut().spawn(SpawnedBy(agent)).id();
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

    // The title goes back on exit.
    app.world_mut().write_message(bevy_app::AppExit::Success);
    app.update();
    let written = app.world().resource::<TuiTestScreen>().written();
    assert!(written.ends_with(POP_TITLE), "{written:?}");
}
