use std::io::Write;
use std::sync::{Arc, Mutex};

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ReadTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::Value;

use super::*;

/// The lines written, shared with the test.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut lines) = self.0.lock() {
            lines.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Lines {
    /// Each line, parsed; `Null` for one that is not JSON.
    fn parsed(&self) -> Vec<Value> {
        let bytes = self.0.lock().map(|lines| lines.clone()).unwrap_or_default();
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap_or(Value::Null))
            .collect()
    }
}

/// The field `key` of `line`, `Null` without one.
fn at<'a>(line: &'a Value, key: &str) -> &'a Value {
    line.get(key).unwrap_or(&Value::Null)
}

#[test]
fn a_prompt_command_streams_the_turn_as_json_lines() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let lines = Lines::default();
    let (input, rpc) = RpcInput::channel();
    app.insert_resource(JsonMode::Rpc)
        .insert_resource(JsonSink::new(lines.clone()))
        .insert_resource(rpc)
        .add_plugins((AgentPlugin, JournalPlugin, ReadTool, JsonStreamPlugin));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "read", serde_json::json!({ "path": "Cargo.toml" })),
        reply("ok"),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and the catalog's model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    let sent = input.send(r#"{"type":"prompt","id":7,"text":"read a.txt"}"#.to_owned());
    assert!(sent.is_ok());
    let ended = |lines: &Lines| {
        lines
            .parsed()
            .iter()
            .any(|line| at(line, "type") == "turn_end")
    };
    let watched = lines.clone();
    run_until(&mut app, &wakes, move |_| ended(&watched));

    let parsed = lines.parsed();
    assert!(
        parsed.iter().all(|line| at(line, "v") == VERSION),
        "{parsed:?}"
    );
    let response = parsed.iter().find(|line| at(line, "type") == "response");
    assert_eq!(
        response.map(|line| (at(line, "id").clone(), at(line, "ok").clone())),
        Some((7.into(), true.into()))
    );
    let kinds: Vec<&str> = parsed
        .iter()
        .filter_map(|line| at(line, "type").as_str())
        .collect();
    let expected = [
        "turn_start",
        "tool_call_start",
        "tool_call_end",
        "text_delta",
        "turn_end",
    ];
    let mut rest = kinds.iter();
    for kind in expected {
        assert!(
            rest.any(|seen| *seen == kind),
            "{kind} in order in {kinds:?}"
        );
    }
    let start = parsed
        .iter()
        .find(|line| at(line, "type") == "tool_call_start");
    assert_eq!(
        start.map(|line| at(line, "name").clone()),
        Some("read".into())
    );
    let end = parsed
        .iter()
        .find(|line| at(line, "type") == "tool_call_end");
    assert_eq!(
        end.map(|line| at(line, "is_error").clone()),
        Some(false.into())
    );
    let text: String = parsed
        .iter()
        .filter(|line| at(line, "type") == "text_delta")
        .filter_map(|line| at(line, "text").as_str())
        .collect();
    assert_eq!(text, "ok");
    let turn_end = parsed.iter().find(|line| at(line, "type") == "turn_end");
    assert_eq!(
        turn_end.map(|line| at(line, "outcome").clone()),
        Some("answered".into())
    );
}
