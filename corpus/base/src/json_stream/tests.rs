use std::sync::{Arc, Mutex};

use rig_basics::coding_tools::ReadTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

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

    /// The first line of type `kind` for which `pick` holds.
    fn find(&self, kind: &str, pick: impl Fn(&Value) -> bool) -> Option<Value> {
        self.parsed()
            .into_iter()
            .find(|line| at(line, "type") == kind && pick(line))
    }
}

#[test]
fn rpc_commands_stream_the_turn_and_answer_requests_as_json_lines() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let lines = Lines::default();
    let (input, rpc) = RpcInput::channel();
    let flags = [(FRONT.to_owned(), None), (RPC.to_owned(), None)];
    app.insert_resource(Invoked {
        args: Invocation {
            flags: flags.into(),
            ..Invocation::default()
        },
        terminal: false,
    })
    .insert_resource(JsonSink::new(lines.clone()))
    .insert_resource(rpc)
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        PickPlugin,
        ReadTool,
        JsonStreamPlugin,
    ));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "read", json!({ "path": "Cargo.toml" })),
        reply("ok"),
    ]);
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let connection = connect(&model);
    assert!(
        agent.is_ok() && connection.is_some(),
        "an agent and a model"
    );
    let (Ok(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    let prompt = r#"{"type":"prompt","id":7,"text":"read a.txt"}"#;
    assert!(input.try_send(prompt.to_owned()).is_ok());
    let watched = lines.clone();
    run_until(&mut app, &wakes, move |_| {
        watched.find("turn_end", |_| true).is_some()
    });

    let parsed = lines.parsed();
    assert!(
        parsed.iter().all(|line| at(line, "v") == VERSION),
        "{parsed:?}"
    );
    let response = lines.find("response", |_| true);
    assert_eq!(
        response.map(|r| (at(&r, "id").clone(), at(&r, "ok").clone())),
        Some((7.into(), true.into()))
    );
    let kinds: Vec<&str> = parsed
        .iter()
        .filter_map(|line| at(line, "type").as_str())
        .collect();
    let mut rest = kinds.iter();
    for kind in [
        "turn_start",
        "tool_call_start",
        "tool_call_end",
        "text_delta",
        "turn_end",
    ] {
        assert!(
            rest.any(|seen| *seen == kind),
            "{kind} in order in {kinds:?}"
        );
    }
    let start = lines.find("tool_call_start", |_| true);
    assert_eq!(
        start.map(|line| at(&line, "name").clone()),
        Some("read".into())
    );
    let text: String = parsed
        .iter()
        .filter(|line| at(line, "type") == "text_delta")
        .filter_map(|line| at(line, "text").as_str())
        .collect();
    assert_eq!(text, "ok");
    let end = lines.find("turn_end", |_| true);
    assert_eq!(
        end.map(|line| at(&line, "outcome").clone()),
        Some("answered".into())
    );

    // A plugin's question about a subagent becomes a request, with the
    // subagent's parent, and an `answer` answers it.
    let world = app.world_mut();
    let sub = world
        .spawn((Agent, Name::new("sub"), SpawnedBy(agent)))
        .id();
    let ask = world.spawn((Ask::yes_no("Run it?"), ForAgent(sub))).id();
    let watched = lines.clone();
    run_until(&mut app, &wakes, move |_| {
        watched.find("request", |_| true).is_some()
    });
    let request = lines.find("request", |_| true).unwrap_or_default();
    let parent = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
    assert_eq!(
        at(&request, "parent").as_str(),
        parent.as_deref(),
        "{request}"
    );
    let answer =
        json!({ "type": "answer", "id": 8, "request_id": at(&request, "request_id"), "choice": 1 });
    assert!(input.try_send(answer.to_string()).is_ok());
    run_until(&mut app, &wakes, |world| {
        world.get::<Outcome>(ask).is_some()
    });
    let answered = app.world().get::<Outcome>(ask).and_then(Answer::of);
    assert_eq!(answered, Some(Answer::choice("no")));
    let response = lines.find("response", |line| at(line, "id") == 8);
    assert_eq!(
        response.map(|line| at(&line, "ok").clone()),
        Some(true.into())
    );

    // A plugin's picker too; the harness carries out the choice.
    let picker = Picker::new("Branch").item("main", "").item("dev", "");
    let picker = app.world_mut().spawn(picker).id();
    let watched = lines.clone();
    run_until(&mut app, &wakes, move |_| {
        watched
            .find("request", |line| at(line, "kind") == "picker")
            .is_some()
    });
    let request = lines
        .find("request", |line| at(line, "kind") == "picker")
        .unwrap_or_default();
    let answer =
        json!({ "type": "answer", "id": 9, "request_id": at(&request, "request_id"), "choice": 1 });
    assert!(input.try_send(answer.to_string()).is_ok());
    run_until(&mut app, &wakes, |world| world.get_entity(picker).is_err());
    let response = lines.find("response", |line| at(line, "id") == 9);
    assert_eq!(
        response.map(|line| at(&line, "ok").clone()),
        Some(true.into())
    );
}

/// The field `key` of `line`, `Null` without one.
fn at<'a>(line: &'a Value, key: &str) -> &'a Value {
    line.get(key).unwrap_or(&Value::Null)
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(|app: &mut App| {
        if let Some(mut invoked) = app.world_mut().get_resource_mut::<Invoked>() {
            invoked.args.flags.push((FRONT.to_owned(), None));
        }
        app.insert_resource(JsonSink::new(std::io::sink()))
            .add_plugins(JsonStreamPlugin);
    });
}
