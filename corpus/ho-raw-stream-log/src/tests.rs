use std::path::Path;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, app_in, connect, first_agent, log_ended, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// The raw folder's files, as `(name, text)`, sorted by name.
fn raw_files(dir: &Path) -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (
                name,
                std::fs::read_to_string(entry.path()).unwrap_or_default(),
            )
        })
        .collect();
    files.sort();
    files
}

/// The lines of `text`, as JSON.
fn lines(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

#[test]
fn a_call_is_recorded_raw_and_redacted_and_off_records_nothing() {
    let unknown = json!({ "type": "response.vendor_ping", "seq": 7, "note": "keep me" });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text_start("msg_1", Some(json!({ "phase": "final_answer" }))),
            MockStreamEvent::unknown(unknown.clone()),
            MockStreamEvent::text("Hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::RequestId("req_42".to_owned()),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("again"),
    ]);
    let home = scratch_home("raw-stream-log");
    let id = SessionId::generate();
    let dir = home.session(&id).path().join("raw");
    let (mut app, wakes) = app_in(&home, &id);
    app.add_plugins((AgentPlugin, JournalPlugin, RawStreamLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let connection = connect(&model);
    assert!(connection.is_some(), "the scripted model connects");
    if let Some(connection) = connection {
        world.entity_mut(agent).insert(connection);
    }

    world.trigger(RunCommand::new(agent, "rawlog on"));
    world.trigger(Deliver::new(
        agent,
        "hello sk-test-123",
        DeliveryMode::Steer,
    ));
    // The turn ended and the writer thread wrote the response.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
            && raw_files(&dir)
                .iter()
                .any(|(_, text)| text.contains("\"done\""))
    });
    let files = raw_files(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    let Some((name, text)) = files.first() else {
        return;
    };
    assert!(
        name.ends_with(".jsonl") && name.contains("deepseek"),
        "{name}"
    );
    let lines = lines(text);
    let kinds: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.get("type").and_then(Value::as_str))
        .collect();
    assert_eq!(kinds.first(), Some(&"request"), "{text}");
    assert_eq!(kinds.last(), Some(&"done"), "{text}");
    let request = lines.first().map(Value::to_string).unwrap_or_default();
    assert!(
        request.contains("hello") && request.contains(REDACTED),
        "{request}"
    );
    assert!(!text.contains("sk-test-123"), "{text}");
    // The stream items in order: the unknown item verbatim before the two
    // text fragments, and the request id on the response.
    let unknown_at = lines
        .iter()
        .position(|line| line.get("item") == Some(&unknown));
    let texts: Vec<usize> = (0..lines.len())
        .filter(|&at| {
            lines
                .get(at)
                .is_some_and(|line| line.pointer("/event/event") == Some(&json!("text")))
        })
        .collect();
    assert!(
        unknown_at.is_some_and(|at| texts.iter().all(|&text| at < text)),
        "{text}"
    );
    assert_eq!(texts.len(), 2, "{text}");
    assert!(
        text.contains("final_answer"),
        "the text start's provider params: {text}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.get("request_id") == Some(&json!("req_42"))),
        "{text}"
    );
    let times: Vec<u64> = lines
        .iter()
        .filter_map(|line| line.get("ms").and_then(Value::as_u64))
        .collect();
    assert!(times.len() + 1 == lines.len(), "{text}");
    assert!(
        times.windows(2).all(|pair| pair.first() <= pair.last()),
        "{times:?}"
    );

    // Off: the next turn writes no file.
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "rawlog off"));
    world.resource_mut::<Ended>().0.clear();
    world.trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert!(!app.world().resource::<RawLog>().on);
    assert_eq!(raw_files(&dir), files);
}
