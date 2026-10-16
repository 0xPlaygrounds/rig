use std::path::Path;

use rig_core::catalog::{Catalog, Connector, ModelSpec};
use rig_core::providers::registry::{ConnectOptions, ProviderId};
use rig_core::test_utils::MockStreamingClient;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{Ended, app_in, first_agent, log_ended, run_until, scratch_home};
use serde_json::json;

use super::*;

/// A key the scrubber knows by its shape.
const KEY: &str = "sk-proj-abcdefghijklmnopqrstuvwx";

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

/// An OpenAI Responses stream as the provider sends it: an unknown event
/// between the message's start and its two text deltas, and the response
/// with its id.
fn sse(unknown: &Value) -> String {
    let message = |text: &str| json!({ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{ "type": "output_text", "text": text, "annotations": [] }] });
    let response = |status: &str, output: Value| json!({ "id": "resp_req_42", "object": "response", "created_at": 0, "status": status, "model": "gpt-5.4", "output": output, "tools": [], "usage": { "input_tokens": 3, "output_tokens": 2, "total_tokens": 5 } });
    let delta = |text: &str| json!({ "type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": text });
    [
        json!({ "type": "response.created", "response": response("in_progress", json!([])) }),
        json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] } }),
        unknown.clone(),
        delta("Hel"),
        delta("lo"),
        json!({ "type": "response.output_item.done", "output_index": 0, "item": message("Hello") }),
        json!({ "type": "response.completed", "response": response("completed", json!([message("Hello")])) }),
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect()
}

#[test]
fn a_call_is_recorded_raw_and_redacted_and_off_records_nothing() {
    let unknown = json!({ "type": "response.vendor_ping", "seq": 7, "note": "keep me" });
    let wire = sse(&unknown);
    let mut connector = Connector::new(Catalog::builtin().clone());
    let openai = ProviderId::catalog("openai");
    let reference = connector
        .catalog()
        .iter()
        .find(|spec| spec.provider.vendor() == "openai" && spec.tools)
        .map(ModelSpec::reference);
    assert!(
        openai.is_some() && reference.is_some(),
        "the catalog lists OpenAI"
    );
    let (Some(openai), Some(reference)) = (openai, reference) else {
        return;
    };
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(wire.clone()),
    };
    connector.provide(
        openai,
        ConnectOptions::new()
            .api_key(KEY)
            .base_url("http://scripted.test/v1")
            .http(http),
    );

    let home = scratch_home("raw-stream-log");
    let id = SessionId::generate();
    let dir = home.session(&id).path().join("raw");
    let (mut app, wakes) = app_in(&home, &id);
    app.insert_resource(Models(connector))
        .add_plugins((AgentPlugin, SessionLogPlugin, RawStreamLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let connection = world.resource::<Models>().connect(&reference);
    assert!(
        connection.as_ref().err().is_none(),
        "{:?}",
        connection.as_ref().err()
    );
    if let Ok(connection) = connection {
        world.entity_mut(agent).insert(connection);
    }

    world.trigger(RunCommand::new(agent, "rawlog on"));
    world.trigger(Deliver::new(
        agent,
        format!("hello {KEY}"),
        DeliveryMode::Steer,
    ));
    // The turn ended and the writer thread wrote the body.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
            && raw_files(&dir)
                .iter()
                .any(|(_, text)| text.contains("response.completed"))
    });
    let files = raw_files(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    let Some((name, text)) = files.first() else {
        return;
    };
    // Named on the handler the call was served by, `model:<reference>`.
    let model = rig_core::effect::model_key(&reference);
    assert!(
        name.ends_with(&format!("-1-{}.jsonl", file_safe(model.as_str()))),
        "{name}"
    );
    assert!(!text.contains(KEY), "{text}");
    let lines = lines(text);
    let kinds: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.get("type").and_then(Value::as_str))
        .collect();
    assert_eq!(kinds.first(), Some(&"request"), "{text}");
    assert_eq!(kinds.get(1), Some(&"response"), "{text}");
    assert!(kinds.iter().skip(2).all(|kind| *kind == "chunk"), "{text}");
    let request = lines.first().map(Value::to_string).unwrap_or_default();
    assert!(
        request.contains("hello") && request.contains(redact::REDACTED),
        "{request}"
    );
    // The body exactly as it came: the unknown event verbatim, in its
    // place, and the response id.
    let body: String = lines
        .iter()
        .filter_map(|line| line.get("data").and_then(Value::as_str))
        .collect();
    assert_eq!(body, wire);
    assert!(body.contains(&unknown.to_string()) && body.contains("resp_req_42"));
    assert!(
        lines
            .iter()
            .all(|line| line.get("attempt") == Some(&json!(1)))
    );
    let times: Vec<u64> = lines
        .iter()
        .filter_map(|line| line.get("ms").and_then(Value::as_u64))
        .collect();
    assert_eq!(times.len() + 1, lines.len(), "{text}");
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

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RawStreamLogPlugin);
}
