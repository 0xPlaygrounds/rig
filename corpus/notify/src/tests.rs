use std::sync::Mutex;

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// Keeps every notification.
#[derive(Default)]
struct Recording(Mutex<Vec<String>>);

impl Notifier for Recording {
    fn notify(&self, text: &str) {
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
    app.insert_resource(TimeUpdateStrategy::ManualDuration(by));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    for _ in 0..3 {
        app.update();
    }
}

#[test]
fn a_finished_turn_and_an_unanswered_picker_notify_but_a_subagent_does_not() {
    let recording = Arc::new(Recording::default());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(Notifications {
            notifier: recording.clone(),
            min_turn: Duration::ZERO,
            on: true,
        })
        .add_plugins((AgentPlugin, JournalPlugin, NotifyPlugin));
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
    app.update();
    for agent in [parent, child] {
        let ask = Deliver::new(agent, "Go.", DeliveryMode::Queue);
        app.world_mut().trigger(ask.with_origin(Origin::user()));
    }
    run_until(&mut app, &wakes, |world| {
        top.request_count() > 0
            && sub.request_count() > 0
            && world.get::<ActiveTurn>(parent).is_none()
            && world.get::<ActiveTurn>(child).is_none()
    });
    // Both turns ended; only the top-level one notified.
    assert_eq!((top.request_count(), sub.request_count()), (1, 1));
    let idle = |agent| app.world().get::<ActiveTurn>(agent).is_none();
    assert!(idle(parent) && idle(child));
    assert_eq!(recording.sent(), ["rig: done — All done."]);

    // A picker left open past the patience asks for the user.
    app.world_mut().write_message(PickRequest {
        agent: parent,
        title: "Allow `rm -rf build`?".to_owned(),
        items: Vec::new(),
        selected: 0,
    });
    app.update();
    advance(&mut app, Duration::from_secs(1));
    assert_eq!(recording.sent().len(), 1, "not yet: it waited 1 s");
    advance(&mut app, Duration::from_secs(2));
    assert_eq!(
        recording.sent(),
        [
            "rig: done — All done.",
            "rig needs you: Allow `rm -rf build`?"
        ]
    );
}
