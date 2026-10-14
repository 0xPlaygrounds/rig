use std::sync::Mutex;

use bevy_time::TimeUpdateStrategy;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// Keeps every notification.
#[derive(Default)]
struct Recording(Mutex<Vec<String>>);

impl Notify for Recording {
    fn notify(&self, text: &str, _: &mut MessageWriter<TerminalWrite>) {
        if let Ok(mut sent) = self.0.lock() {
            sent.push(text.to_owned());
        }
    }
}

impl Recording {
    fn sent(&self) -> Vec<String> {
        self.0.lock().map(|sent| sent.clone()).unwrap_or_default()
    }
}

/// Runs one frame `by` later on the app's clock, then a few with it still.
fn advance(app: &mut App, by: Duration) {
    rig_harness_test_support::advance(app, by);
    for _ in 0..3 {
        app.update();
    }
}

#[test]
fn a_long_turn_and_an_unanswered_picker_notify_but_a_subagent_an_answered_question_and_a_focused_terminal_do_not()
 {
    let recording = Arc::new(Recording::default());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(Settings::from_toml("[notify]\nmin_seconds = 0"))
        .insert_resource(Notifier(recording.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, NotifyPlugin));
    let (top, sub) = (
        MockCompletionModel::from_stream_turns([reply("All done.\nmore")]),
        MockCompletionModel::from_stream_turns([reply("Sub done.")]),
    );
    let lines = connect(&top).zip(connect(&sub));
    assert!(lines.is_some(), "the built-in catalog lists the model");
    let Some((top_line, sub_line)) = lines else {
        return;
    };
    let parent = app.world_mut().spawn((Agent, top_line)).id();
    let child = app
        .world_mut()
        .spawn((Agent, sub_line, SpawnedBy(parent)))
        .id();
    app.finish();
    // The terminal view is the front, and the user is in another window.
    app.insert_resource(TerminalView(true))
        .insert_resource(TerminalFocus(Some(false)));
    app.update();
    for agent in [parent, child] {
        let ask = Deliver::new(agent, "Go.", DeliveryMode::Queue);
        app.world_mut().trigger(ask);
    }
    run_until(&mut app, &wakes, |world| {
        top.request_count() > 0
            && sub.request_count() > 0
            && world.get::<ActiveTurn>(parent).is_none()
            && world.get::<ActiveTurn>(child).is_none()
    });
    // Both turns ended; only the top-level one notified.
    assert_eq!((top.request_count(), sub.request_count()), (1, 1));
    assert_eq!(recording.sent(), ["rig: done — All done."]);

    // A picker left open past the patience asks for the user; a question
    // answered within it does not.
    let picker = Picker::new("Allow `rm -rf build`?");
    app.world_mut().spawn(picker);
    let question = app
        .world_mut()
        .spawn((Ask::yes_no("Commit?"), ForAgent(parent)))
        .id();
    app.update();
    advance(&mut app, Duration::from_secs(1));
    assert_eq!(recording.sent().len(), 1, "not yet: they waited 1 s");
    let yes = Answer::choice("yes").outcome();
    app.world_mut().entity_mut(question).insert(yes);
    advance(&mut app, Duration::from_secs(2));
    let sent = [
        "rig: done — All done.",
        "rig needs you: Allow `rm -rf build`?",
    ];
    assert_eq!(recording.sent(), sent);

    // Back at the terminal, a picker waiting does not notify.
    app.insert_resource(TerminalFocus(Some(true)));
    app.world_mut().spawn(Picker::new("Allow `git push`?"));
    app.update();
    advance(&mut app, Duration::from_secs(3));
    assert_eq!(recording.sent(), sent);
}
