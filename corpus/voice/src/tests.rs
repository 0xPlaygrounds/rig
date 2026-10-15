use std::sync::{Arc, Mutex};

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{advance, app_on, connect, first_agent, reply, run_until};
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

fn draft(world: &World) -> String {
    let main = world.try_query_filtered::<&rig_tui::Editor, With<rig_tui::MainEditor>>();
    let main =
        main.and_then(|mut main| main.single(world).ok().map(|input| input.text().to_owned()));
    main.unwrap_or_default()
}

#[test]
fn a_held_key_sends_the_transcript_after_a_moment_and_the_answer_is_spoken_without_code() {
    let speaker = FakeSpeaker::default();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(VoiceBackends {
        recorder: Some(Arc::new(FakeRecorder)),
        transcriber: Some(Arc::new(FakeTranscriber)),
        speaker: Some(Arc::new(speaker.clone())),
    })
    .add_plugins((AgentPlugin, SessionLogPlugin, VoicePlugin));
    app.finish();
    // Frames take no time until the test moves the clock.
    advance(&mut app, Duration::ZERO);
    let model = MockCompletionModel::from_stream_turns([reply(
        "There are three files.\n\n```\na b c\n```",
    )]);
    let world = app.world_mut();
    let agent = first_agent(world);
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
