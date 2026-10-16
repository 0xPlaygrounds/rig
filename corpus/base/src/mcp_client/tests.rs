use std::io::{BufRead, Write};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::MemoryStore;
use serde_json::{Value, json};

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// Set in the environment of the stub server this test binary runs as.
const STUB: &str = "CORPUS_MCP_STUB";

/// Each settled tool call's result, rendered.
#[derive(Resource, Default)]
struct Outputs(Vec<String>);

/// The test binary as an MCP server with the tools `add(a, b)` and `hang`,
/// which never answers: run by the other test as a child, it answers
/// JSON-RPC lines on stdin. The lines
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
            }, {
                "name": "hang",
                "description": "Never answers.",
                "inputSchema": { "type": "object" },
            }]}),
            Some("tools/call") if params.get("name") == Some(&json!("hang")) => continue,
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

/// Whether `stub__add` is registered now.
fn has_add(world: &World) -> bool {
    world.try_query::<&ToolDef>().is_some_and(|mut defs| {
        defs.iter(world)
            .any(|def| def.0.name.as_str() == "stub__add")
    })
}

#[test]
fn stub_tools_come_and_go_with_its_server() {
    // The project's `.rig/mcp.json` names the stub: this test binary,
    // running only `stub`.
    let project = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|deps| deps.join("corpus-mcp-client")))
        .unwrap_or_default()
        .join(std::process::id().to_string());
    std::fs::create_dir_all(project.join(".rig")).ok();
    let exe = std::env::current_exe().unwrap_or_default();
    let config = json!({ "mcpServers": { "stub": {
        "command": exe,
        "args": [test_path("stub"), "--exact", "--ignored", "--nocapture", "--test-threads=1"],
        "env": { STUB: "1" },
    }}});
    std::fs::write(project.join(".rig/mcp.json"), config.to_string()).ok();

    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "stub__add", json!({ "a": 2, "b": 3 })),
        reply("2 + 3 = 5."),
        calls("c2", "stub__hang", json!({})),
        reply("It went away."),
        reply("Nothing to add with."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(project.clone()))
        .insert_resource(RigHome(Home::at(project.join("home"))))
        // No second attempt while the test runs.
        .insert_resource(Settings::from_toml("[mcp]\nfirst_retry_secs = 3600"))
        .add_plugins((
            ServicesPlugin,
            AgentPlugin,
            SessionLogPlugin,
            McpClientPlugin,
        ))
        .init_resource::<Outputs>()
        .add_observer(
            |settled: On<Add<Settled>>,
             calls: Query<&Outcome, With<ToolRequest>>,
             mut log: ResMut<Outputs>| {
                if let Ok(outcome) = calls.get(settled.entity) {
                    let result = outcome.tool_result();
                    let text = result.error().map_or_else(
                        || result.output().render(),
                        |error| error.message().to_owned(),
                    );
                    log.0.push(text);
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
    assert!(has_add(app.world()));
    // The server says `add` only reads, which is only its word: a restart
    // never runs a call of it again.
    let world = app.world_mut();
    let footprint = world
        .query::<(&ToolDef, &Footprint)>()
        .iter(world)
        .find(|(def, _)| def.0.name.as_str() == "stub__add")
        .map(|(_, footprint)| footprint.0.clone());
    assert_eq!(footprint, Some(Touches::Independent));
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

    // Killed while a call runs, the call fails for that, and the server's
    // tools are gone from the next request.
    app.world_mut()
        .trigger(Deliver::new(agent, "Hang.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world
            .try_query_filtered::<&ToolRequest, (With<Started>, Without<Outcome>)>()
            .is_some_and(|mut calls| {
                calls
                    .iter(world)
                    .any(|call| call.call.function.name.as_str() == "stub__hang")
            })
    });
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
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 4 && !world.entity(agent).contains::<ActiveTurn>()
    });
    let outputs = &app.world().resource::<Outputs>().0;
    assert_eq!(outputs, &["5", "server `stub` disconnected"]);
    run_until(&mut app, &wakes, |world| !has_add(world));
    app.world_mut()
        .trigger(Deliver::new(agent, "And now?", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |_| model.request_count() == 5);
    assert_eq!(model.request_count(), 5);
    assert!(!offered(4));
    let world = app.world_mut();
    let status = world.query::<&McpStatus>().iter(world).next().cloned();
    let error = status.and_then(|status| status.last_error);
    assert_eq!(error.as_deref(), Some("server `stub` disconnected"));
    std::fs::remove_dir_all(project).ok();
}

/// The test `name` of this module as its test binary names it: the module's
/// path without the crate's name.
fn test_path(name: &str) -> String {
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    format!("{module}::{name}")
}
