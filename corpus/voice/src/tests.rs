use std::sync::{Arc, Mutex};

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

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

/// How many transcripts waited to be sent.
#[derive(Resource, Default)]
struct Waited(usize);

#[test]
fn a_held_key_sends_the_transcript_after_a_moment_and_the_answer_is_spoken_without_code() {
    let speaker = FakeSpeaker::default();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .insert_resource(VoiceBackends {
            recorder: Some(Arc::new(FakeRecorder)),
            transcriber: Some(Arc::new(FakeTranscriber)),
            speaker: Some(Arc::new(speaker.clone())),
            summary_model: None,
        })
        .add_plugins((AgentPlugin, JournalPlugin, VoicePlugin))
        .init_resource::<Waited>()
        .add_observer(
            |_: On<Add<PendingTranscript>>, mut waited: ResMut<Waited>| {
                waited.0 += 1;
            },
        );
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
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent on the catalog's model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    // Press, release: the transcript waits in the place of being sent.
    app.world_mut().trigger(TalkPressed { entity: agent });
    app.update();
    app.world_mut().trigger(TalkReleased { entity: agent });
    run_until(&mut app, &wakes, |world| world.resource::<Waited>().0 > 0);
    assert_eq!(app.world().resource::<Waited>().0, 1);
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
