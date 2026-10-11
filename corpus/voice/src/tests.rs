use std::sync::{Arc, Mutex};

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};
use rig_tui::KeyBinding;

/// Records a fixed buffer.
struct FakeRecorder;

impl Recorder for FakeRecorder {
    fn start(&self) -> Result<Box<dyn Recording>, String> {
        Ok(Box::new(FakeRecording))
    }
}

struct FakeRecording;

impl Recording for FakeRecording {
    fn stop(self: Box<Self>) -> Result<Audio, String> {
        Ok(Audio {
            samples: vec![1, -1, 2, -2],
            rate: 16_000,
        })
    }
}

/// Hears "list the files" in any recording.
struct FakeTranscriber;

impl Transcriber for FakeTranscriber {
    fn transcribe(&self, audio: &Audio) -> Result<String, String> {
        if audio.samples.is_empty() {
            return Err("nothing recorded".to_owned());
        }
        Ok("list the files".to_owned())
    }
}

/// Keeps what it was asked to say.
#[derive(Clone, Default)]
struct FakeSpeaker(Arc<Mutex<Vec<String>>>);

impl FakeSpeaker {
    fn said(&self) -> Vec<String> {
        self.0.lock().map(|said| said.clone()).unwrap_or_default()
    }
}

impl Speaker for FakeSpeaker {
    fn speak(&self, text: &str) -> Result<(), String> {
        if let Ok(mut said) = self.0.lock() {
            said.push(text.to_owned());
        }
        Ok(())
    }

    fn stop(&self) {}
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

fn draft(world: &World) -> String {
    world.resource::<TuiInput>().text().to_owned()
}

#[test]
fn a_held_key_sends_the_transcript_after_a_moment_and_the_answer_is_spoken_without_code() {
    let speaker = FakeSpeaker::default();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .insert_resource(VoiceBackends {
            recorder: Some(Arc::new(FakeRecorder)),
            transcriber: Some(Arc::new(FakeTranscriber)),
            speaker: Some(Arc::new(speaker.clone())),
        })
        .add_plugins((AgentPlugin, JournalPlugin, VoicePlugin));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply(
        "There are three files.\n\n```\na b c\n```",
    )]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let key = KEY.parse::<Keys>().ok();
    let binding = world
        .query::<(Entity, &KeyBinding)>()
        .iter(world)
        .find(|(_, binding)| Some(&binding.keys) == key.as_ref() && binding.releases)
        .map(|(entity, _)| entity);
    let connection = connect(&model);
    assert!(
        agent.is_some() && binding.is_some() && connection.is_some(),
        "an agent on the catalog's model, and the push-to-talk key bound"
    );
    let (Some(agent), Some(binding), Some(connection)) = (agent, binding, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    // Hold the key, let go: the transcript waits in the input.
    let press = |released| KeyPressed {
        entity: binding,
        agent: Some(agent),
        released,
    };
    app.world_mut().trigger(press(false));
    app.update();
    app.world_mut().trigger(press(true));
    run_until(&mut app, &wakes, |world| !draft(world).is_empty());
    assert_eq!(draft(app.world()), "list the files");
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(
        model.request_count(),
        0,
        "not sent before the moment passed"
    );

    // The moment passes: it is sent, answered, and the answer spoken.
    advance(&mut app, AUTO_SEND);
    run_until(&mut app, &wakes, |_| !speaker.said().is_empty());
    assert_eq!(draft(app.world()), "");
    let sent = model
        .requests()
        .iter()
        .map(|request| format!("{request:?}"))
        .collect::<Vec<_>>();
    assert_eq!(sent.len(), 1);
    assert!(
        sent.iter()
            .all(|request| request.contains("list the files")),
        "{sent:?}"
    );
    assert_eq!(speaker.said(), ["There are three files."]);
}
