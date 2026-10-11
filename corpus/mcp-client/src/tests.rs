use std::io::{BufRead, Write};
use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use serde_json::{Value, json};

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// Set in the environment of the stub server this test binary runs as.
const STUB: &str = "CORPUS_MCP_STUB";

/// Each tool output, rendered.
#[derive(Resource, Default)]
struct Outputs(Vec<String>);

/// The test binary as an MCP server with the tool `add(a, b)`: run by the
/// other test as a child, it answers JSON-RPC lines on stdin. The lines
/// libtest prints on stdout are not JSON, and the client skips them.
#[test]
#[ignore = "the stub MCP server the other test starts"]
fn stub() {
    if std::env::var_os(STUB).is_none() {
        return;
    }
    let mut out = std::io::stdout().lock();
    // libtest left its line about this test open.
    if writeln!(out).is_err() {
        return;
    }
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { return };
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let params = request.get("params").cloned().unwrap_or_default();
        let result = match request.get("method").and_then(Value::as_str) {
            Some("initialize") => json!({
                "protocolVersion": params.get("protocolVersion").cloned().unwrap_or_default(),
                "capabilities": { "tools": { "listChanged": true } },
                "serverInfo": { "name": "stub", "version": "0" },
            }),
            Some("tools/list") => json!({ "tools": [{
                "name": "add",
                "description": "Adds two numbers.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                    "required": ["a", "b"],
                },
                "annotations": { "readOnlyHint": true },
            }]}),
            Some("tools/call") => {
                let number = |key| {
                    params
                        .pointer(&format!("/arguments/{key}"))
                        .and_then(Value::as_f64)
                        .unwrap_or_default()
                };
                let sum = number("a") + number("b");
                json!({ "content": [{ "type": "text", "text": sum.to_string() }] })
            }
            _ => json!({}),
        };
        let answer = json!({ "jsonrpc": "2.0", "id": id, "result": result });
        if writeln!(out, "{answer}")
            .and_then(|()| out.flush())
            .is_err()
        {
            return;
        }
    }
}

/// The names of the tools registered now.
fn tools(world: &mut World) -> Vec<String> {
    let mut defs = world.query::<&ToolDef>();
    defs.iter(world)
        .map(|def| def.0.name.as_str().to_owned())
        .collect()
}

fn has_add(world: &World) -> bool {
    world.try_query::<&ToolDef>().is_some_and(|mut defs| {
        defs.iter(world)
            .any(|def| def.0.name.as_str() == "stub__add")
    })
}

#[test]
fn stub_tools_come_and_go_with_its_server() {
    // The config names the stub: this test binary, running only `stub`.
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|deps| deps.join("corpus-mcp-client")))
        .unwrap_or_else(|| PathBuf::from("corpus-mcp-client"));
    std::fs::create_dir_all(&dir).ok();
    let exe = std::env::current_exe().unwrap_or_default();
    let config = json!({ "mcpServers": { "stub": {
        "command": exe,
        "args": ["tests::stub", "--exact", "--ignored", "--nocapture", "--test-threads=1"],
        "env": { STUB: "1" },
    }}});
    let file = dir.join(format!("mcp-{}.json", std::process::id()));
    std::fs::write(&file, config.to_string()).ok();

    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "stub__add", json!({ "a": 2, "b": 3 })),
        reply("2 + 3 = 5."),
        reply("Nothing to add with."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(McpConfig {
        files: vec![file.clone()],
        // No second attempt while the test runs.
        first_retry: Duration::from_secs(3600),
    })
    .add_plugins((AgentPlugin, JournalPlugin, McpClientPlugin))
    .init_resource::<Outputs>()
    .add_observer(
        |output: On<Insert<ToolOutput>>, calls: Query<&ToolOutput>, mut log: ResMut<Outputs>| {
            if let Ok(output) = calls.get(output.entity) {
                log.0.push(output.0.output().render());
            }
        },
    );
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(agent.is_some() && connection.is_some());
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

    // The agent is usable at once; the tool comes once the server listed it.
    run_until(&mut app, &wakes, has_add);
    assert!(has_add(app.world()), "tools: {:?}", tools(app.world_mut()));
    app.world_mut()
        .trigger(Deliver::new(agent, "Add 2 and 3.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && !world.entity(agent).contains::<ActiveTurn>()
    });
    let offered = |request: usize| {
        model.requests().get(request).is_some_and(|request| {
            request
                .tools
                .iter()
                .any(|tool| tool.name.as_str() == "stub__add")
        })
    };
    assert!(offered(0));
    assert_eq!(app.world().resource::<Outputs>().0, ["5"]);

    // Killed, the server's tools are gone from the next request.
    let world = app.world_mut();
    let pid = world
        .query::<&McpStatus>()
        .iter(world)
        .find_map(|status| status.pid);
    assert!(pid.is_some(), "the stub's process is not known");
    let Some(pid) = pid else { return };
    let killed = std::process::Command::new("kill")
        .arg(pid.to_string())
        .status();
    assert!(killed.is_ok_and(|status| status.success()));
    run_until(&mut app, &wakes, |world| !has_add(world));
    app.world_mut()
        .trigger(Deliver::new(agent, "And now?", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |_| model.request_count() == 3);
    assert_eq!(model.request_count(), 3);
    assert!(!offered(2));
    let world = app.world_mut();
    let status = world.query::<&McpStatus>().iter(world).next().cloned();
    let error = status.and_then(|status| status.last_error);
    assert_eq!(error.as_deref(), Some("server `stub` disconnected"));
    std::fs::remove_file(file).ok();
}
