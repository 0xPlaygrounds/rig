#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Instant;

use rig_basics::print::{HeadlessEnded, PrintPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness::host::exit::ExitPlugin;
use rig_harness::rig_memory::DirFiles;
use rig_harness::run::TOOLS;
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::{
    Collected, MemoryStore, app_on, calls, collect, connect, reply, run_until, scripted_models,
};
use serde_json::{Value, json};

use super::*;

/// Set in the environment of the stub server this test binary runs as.
const STUB: &str = "RIG_MCP_STUB";

/// Where the stub, as it starts, writes a file, if set.
const MARKER: &str = "RIG_MCP_MARKER";

/// When set, the stub reads its requests and never answers one.
const SILENT: &str = "RIG_MCP_SILENT";

/// When set, the stub says its tools changed just before it answers each of
/// its first two lists, which lack `mul`; every later list has it.
const CHANGES: &str = "RIG_MCP_CHANGES";

/// When set, the stub offers no tools: it says so as it starts, and answers
/// `tools/list` with an error.
const NO_TOOLS: &str = "RIG_MCP_NO_TOOLS";

/// Where the stub writes the process id of a child it starts and leaves
/// running, if set.
const CHILD: &str = "RIG_MCP_CHILD";

/// When set, the stub says its tools changed twice after its first list,
/// leaves the list that follows the first word unanswered, and has `mul` in
/// the next.
const HANGS: &str = "RIG_MCP_HANGS";

/// When set, the stub writes each answer after a byte-order mark, and its
/// list of tools describes `add` at length.
const LONG: &str = "RIG_MCP_LONG";

/// When set, the stub refuses an `initialize` that offers any protocol
/// version but the latest released one, as a server that knows only that
/// one does.
const RELEASED: &str = "RIG_MCP_RELEASED";

/// When set, the stub answers `initialize` with the older protocol version
/// `2025-06-18`, whatever was offered.
const OLDER: &str = "RIG_MCP_OLDER";

/// The model a headless run chooses.
const MODEL: &str = "test/model";

/// Each settled tool call's result, rendered.
#[derive(Resource, Default)]
struct Outputs(Vec<String>);

/// The test binary as an MCP server with the tools `add(a, b)` and `hang`,
/// which never answers: run by the other tests as a child, it answers
/// JSON-RPC lines on stdin. The lines libtest prints on stdout are not
/// JSON, and the client skips them.
#[test]
#[ignore = "the stub MCP server the other tests start"]
fn stub() {
    if std::env::var_os(STUB).is_none() {
        return;
    }
    if let Some(marker) = std::env::var_os(MARKER) {
        std::fs::write(marker, "started").ok();
    }
    if let Some(file) = std::env::var_os(CHILD) {
        let child = std::process::Command::new("sleep").arg("600").spawn();
        if let Ok(child) = child {
            std::fs::write(file, child.id().to_string()).ok();
        }
    }
    if std::env::var_os(SILENT).is_some() {
        std::io::stdin().lock().lines().for_each(drop);
        return;
    }
    let mut out = std::io::stdout();
    // libtest left its line about this test open.
    if writeln!(out).is_err() {
        return;
    }
    let mut lists = 0;
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
                "protocolVersion": match std::env::var_os(OLDER) {
                    Some(_) => json!("2025-06-18"),
                    None => params.get("protocolVersion").cloned().unwrap_or_default(),
                },
                "capabilities": match std::env::var_os(NO_TOOLS) {
                    Some(_) => json!({ "resources": {} }),
                    None => json!({ "tools": { "listChanged": true } }),
                },
                "serverInfo": { "name": "stub", "version": "0" },
            }),
            Some("tools/list") if std::env::var_os(HANGS).is_some() => {
                lists += 1;
                let tool =
                    |name: &str| json!({ "name": name, "inputSchema": { "type": "object" } });
                match lists {
                    1 => {
                        std::thread::spawn(|| {
                            let changed = json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" });
                            for wait in [300, 2500] {
                                std::thread::sleep(Duration::from_millis(wait));
                                writeln!(std::io::stdout(), "{changed}").ok();
                            }
                        });
                        json!({ "tools": [tool("add")] })
                    }
                    2 => continue,
                    _ => json!({ "tools": [tool("add"), tool("mul")] }),
                }
            }
            Some("tools/list") if std::env::var_os(CHANGES).is_some() => {
                lists += 1;
                let changed =
                    json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" });
                if lists <= 2
                    && writeln!(out, "{changed}")
                        .and_then(|()| out.flush())
                        .is_err()
                {
                    return;
                }
                let tool =
                    |name: &str| json!({ "name": name, "inputSchema": { "type": "object" } });
                match lists {
                    1 | 2 => json!({ "tools": [tool("add")] }),
                    _ => json!({ "tools": [tool("add"), tool("mul")] }),
                }
            }
            Some("tools/list") => json!({ "tools": [{
                "name": "add",
                "description": match std::env::var_os(LONG) {
                    Some(_) => "Adds two numbers. ".repeat(100),
                    None => "Adds two numbers.".to_owned(),
                },
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
        let offered = params.get("protocolVersion").and_then(Value::as_str);
        let answer = match (request.get("method"), std::env::var_os(NO_TOOLS)) {
            (Some(method), _)
                if method == "initialize"
                    && std::env::var_os(RELEASED).is_some()
                    && offered != Some("2025-11-25") =>
            {
                json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32602, "message": format!("Unsupported protocol version {offered:?}") },
                })
            }
            (Some(method), Some(_)) if method == "tools/list" => json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": -32601, "message": "Method not found" },
            }),
            _ => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        };
        let bom = if std::env::var_os(LONG).is_some() {
            "\u{feff}"
        } else {
            ""
        };
        if writeln!(out, "{bom}{answer}")
            .and_then(|()| out.flush())
            .is_err()
        {
            return;
        }
    }
}

/// The stub's server object, with `env` added to its environment.
fn stub_server(env: Value) -> Value {
    let exe = std::env::current_exe().unwrap_or_default();
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    let mut vars = json!({ STUB: "1" });
    if let (Some(vars), Value::Object(env)) = (vars.as_object_mut(), env) {
        vars.extend(env);
    }
    json!({
        "command": exe,
        "args": [format!("{module}::stub"), "--exact", "--ignored", "--nocapture", "--test-threads=1"],
        "env": vars,
    })
}

/// An empty project of the test `name` in this run, beside the test
/// binary, with its `RIG_HOME` in `home`.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let run = format!("rig-mcp-client-{}", std::process::id());
    let project = exe.with_file_name(run).join(name);
    std::fs::remove_dir_all(&project).ok();
    std::fs::create_dir_all(project.join("home")).ok();
    // Its own `.rig` makes it the project, wherever the test runs.
    std::fs::create_dir_all(project.join(".rig")).ok();
    project
}

/// Writes `{"mcpServers": servers}` at `path`.
fn write_servers(path: &Path, servers: Value) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(path, json!({ "mcpServers": servers }).to_string()).ok();
}

/// An app with the plugin started in `project`, its home inside it, `settings`, and
/// the notices it writes collected; `invoked` makes it a headless run.
fn app_in(project: &Path, settings: &str, invoked: Option<Invocation>) -> (App, Receiver<()>) {
    app_with::<McpPlugin>(project, settings, invoked)
}

/// [`app_in`] with the plugin `P`.
fn app_with<P: Plugin + Default>(
    project: &Path,
    settings: &str,
    invoked: Option<Invocation>,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let data = PluginData {
        session: Arc::new(MemoryStore::default()),
        project: Arc::new(DirFiles::new(project.join("home/projects/it"))),
        user: Arc::new(DirFiles::new(project.join("home/data"))),
    };
    app.insert_resource(ProjectRoot(project.to_owned()))
        .insert_resource(RigHome(Home::at(project.join("home"))))
        .insert_resource(Settings::from_toml(settings))
        .insert_resource(data);
    if let Some(invoked) = invoked {
        app.insert_resource(Invoked {
            args: invoked,
            terminal: false,
        });
    }
    app.add_plugins((ServicesPlugin, AgentPlugin, SessionLogPlugin));
    // As the agent loads it, with the entity its tools are attributed to.
    rig_harness::load::<P>(&mut app, "rig-mcp-client", "path plugins/rig-mcp-client");
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    (app, wakes)
}

/// Whether `name` is a registered tool now.
fn has(world: &World, name: &str) -> bool {
    world
        .try_query::<&ToolDef>()
        .is_some_and(|mut defs| defs.iter(world).any(|def| def.0.name.as_str() == name))
}

/// The status of the server `name`.
fn status(world: &mut World, name: &str) -> Option<McpStatus> {
    world
        .query::<(&McpServer, &McpStatus)>()
        .iter(world)
        .find(|(server, _)| server.name == name)
        .map(|(_, status)| status.clone())
}

/// The file the server `name` is from, if there is one.
fn scope(world: &mut World, name: &str) -> Option<Scope> {
    world
        .query::<&McpServer>()
        .iter(world)
        .find(|server| server.name == name)
        .map(|server| server.scope)
}

/// The texts of the notices written so far.
fn notices(world: &World) -> Vec<String> {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

/// Runs `/mcp <args>` for `agent`.
fn mcp(app: &mut App, agent: Entity, args: &str) {
    app.world_mut()
        .trigger(RunCommand::new(agent, format!("mcp {args}")));
    app.update();
}

#[test]
fn stub_tools_come_and_go_with_its_server() {
    let project = project("come-and-go");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "stub": stub_server(json!({})) }),
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "stub__add", json!({ "a": 2, "b": 3 })),
        reply("2 + 3 = 5."),
        calls("c2", "stub__hang", json!({})),
        reply("It went away."),
        reply("Nothing to add with."),
    ]);
    // No second attempt while the test runs.
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.init_resource::<Outputs>().add_observer(
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
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    assert!(notices(app.world()).contains(&"MCP server `stub` connected: 2 tools.".to_owned()));
    // The server says `add` only reads, which is only its word: a restart
    // never runs a call of it again.
    let world = app.world_mut();
    let footprint = world
        .query::<(&ToolDef, &Footprint)>()
        .iter(world)
        .find(|(def, _)| def.0.name.as_str() == "stub__add")
        .map(|(_, footprint)| footprint.0.clone());
    assert_eq!(footprint, Some(Touches::Independent));
    // The tool is the plugin's, as what it spawns as it builds is.
    let mine = world.resource::<PluginEntity<McpPlugin>>().entity();
    let by = world
        .query::<(&ToolDef, &ProvidedBy)>()
        .iter(world)
        .find(|(def, _)| def.0.name.as_str() == "stub__add")
        .map(|(_, by)| by.0);
    assert_eq!(by, Some(mine));
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
    let pid = status(app.world_mut(), "stub").and_then(|status| status.pid);
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
    run_until(&mut app, &wakes, |world| !has(world, "stub__add"));
    app.world_mut()
        .trigger(Deliver::new(agent, "And now?", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |_| model.request_count() == 5);
    assert_eq!(model.request_count(), 5);
    assert!(!offered(4));
    let error = status(app.world_mut(), "stub").and_then(|status| status.last_error);
    assert_eq!(error.as_deref(), Some("server `stub` disconnected"));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_project_server_starts_with_no_question() {
    let project = project("trusted");
    let marker = project.join("started");
    let server = stub_server(json!({ MARKER: marker }));
    write_servers(&project.join(".rig/mcp.json"), json!({ "stub": server }));
    let (mut app, wakes) = app_in(&project, "", None);
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    assert!(marker.exists());
    assert_eq!(scope(app.world_mut(), "stub"), Some(Scope::Project));
    let told = notices(app.world());
    assert!(
        told.contains(&"MCP server `stub` connected: 2 tools.".to_owned()),
        "{told:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

/// [`app_in`] with [`approval::McpApprovalPlugin`] on too.
fn asking_app(project: &Path) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.add_plugins(approval::McpApprovalPlugin);
    app.finish();
    app.update();
    (app, wakes)
}

#[test]
fn with_approval_a_project_server_waits_for_approve_and_a_changed_one_is_asked_again() {
    let project = project("approval");
    let marker = project.join("started");
    let mut server = stub_server(json!({ MARKER: marker }));
    // What approving approves is shown whole, as written.
    if let Some(env) = server.get_mut("env").and_then(Value::as_object_mut) {
        env.insert("TOKEN".to_owned(), json!("${RIG_MCP_TEST_TOKEN:-none}"));
    }
    let whole = server.to_string();
    write_servers(&project.join(".rig/mcp.json"), json!({ "stub": server }));
    let (mut app, wakes) = asking_app(&project);
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    // Time enough for a process to start, were one started.
    let waited = Instant::now();
    while waited.elapsed() < Duration::from_millis(500) {
        app.update();
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!marker.exists(), "an unapproved server started");
    let state = status(app.world_mut(), "stub").map(|status| status.state);
    assert_eq!(state, Some(McpState::Held));
    let told = notices(app.world());
    assert!(told.iter().any(|text| text.contains(&whole)), "{told:?}");
    mcp(&mut app, agent, "");
    let listed =
        format!("stub (project): not approved; /mcp approve stub starts it as written: {whole}");
    let told = notices(app.world());
    assert!(told.iter().any(|text| text == &listed), "{told:?}");
    // A word after the name is refused, and nothing is approved.
    mcp(&mut app, agent, "approve stub --user");
    let state = status(app.world_mut(), "stub").map(|status| status.state);
    assert_eq!(state, Some(McpState::Held));

    mcp(&mut app, agent, "approve stub");
    let approved = format!("Approved `stub` as written: {whole}.");
    assert!(notices(app.world()).iter().any(|text| text == &approved));
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    assert!(marker.exists());
    // Denied, it stops.
    mcp(&mut app, agent, "deny stub");
    run_until(&mut app, &wakes, |world| !has(world, "stub__add"));
    let state = status(app.world_mut(), "stub").map(|status| status.state);
    assert_eq!(state, Some(McpState::Held));
    mcp(&mut app, agent, "approve stub");
    drop(app);

    // The approval is kept, for the server as it was written then.
    let (mut app, wakes) = asking_app(&project);
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    drop(app);
    let changed = stub_server(json!({ MARKER: marker, "MORE": "1" }));
    write_servers(&project.join(".rig/mcp.json"), json!({ "stub": changed }));
    let (mut app, _) = asking_app(&project);
    let state = status(app.world_mut(), "stub").map(|status| status.state);
    assert_eq!(state, Some(McpState::Held));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn with_approval_a_users_server_is_never_held() {
    let project = project("approval-home");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "stub": stub_server(json!({})) }),
    );
    let (mut app, wakes) = asking_app(&project);
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    assert_eq!(scope(app.world_mut(), "stub"), Some(Scope::User));
    std::fs::remove_dir_all(project).ok();
}

/// A policy plugin of a few lines, on the public API: it holds `held`.
fn hold_held(added: On<Add<McpServer>>, servers: Query<&McpServer>, mut commands: Commands) {
    if servers
        .get(added.entity)
        .is_ok_and(|server| server.name == "held")
    {
        let hold = McpHold("held by the test's policy".to_owned());
        commands.entity(added.entity).insert(hold);
    }
}

/// A policy of a few lines that holds a server once it sees one of its
/// tools.
fn hold_on_a_tool(added: On<Add<McpToolOf>>, tools: Query<&McpToolOf>, mut commands: Commands) {
    if let Ok(McpToolOf(server)) = tools.get(added.entity) {
        let hold = McpHold("a tool of it is not allowed".to_owned());
        commands.entity(*server).insert(hold);
    }
}

#[test]
fn a_hold_that_comes_as_the_tools_register_takes_them_all_away() {
    let project = project("policy-tools");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "stub": stub_server(json!({})) }),
    );
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.add_observer(hold_on_a_tool);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        world
            .try_query_filtered::<(), (With<McpServer>, With<McpHold>)>()
            .is_some_and(|mut held| held.iter(world).next().is_some())
    });
    assert!(!has(app.world(), "stub__add") && !has(app.world(), "stub__hang"));
    let state = status(app.world_mut(), "stub").map(|status| status.state);
    assert_eq!(state, Some(McpState::Held));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_policy_plugin_holds_the_server_it_names_until_it_lets_go() {
    let project = project("policy");
    let marker = project.join("started");
    write_servers(
        &project.join("home/mcp.json"),
        json!({
            "held": stub_server(json!({ MARKER: marker })),
            "free": stub_server(json!({})),
        }),
    );
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.add_observer(hold_held);
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "free__add"));
    assert!(!marker.exists() && !has(app.world(), "held__add"));
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    mcp(&mut app, agent, "");
    let told = notices(app.world());
    let listed = "held (user): held by the test's policy";
    assert!(told.iter().any(|text| text.contains(listed)), "{told:?}");
    let world = app.world_mut();
    let held = world
        .query_filtered::<(Entity, &McpServer), With<McpHold>>()
        .iter(world)
        .map(|(entity, server)| (entity, server.name.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        held.iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        ["held"]
    );
    if let Some((entity, _)) = held.first() {
        world.entity_mut(*entity).remove::<McpHold>();
    }
    run_until(&mut app, &wakes, |world| has(world, "held__add"));
    assert!(marker.exists());
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn add_and_remove_save_to_the_users_file_and_keep_the_rest() {
    let project = project("persistence");
    let file = project.join("home/mcp.json");
    let before = "{\n    \"theme\": [1, 2],\n    \"mcpServers\": {\n        \"kept\": { \"command\": \"nope\", \"disabled\": true }\n    }\n}\n";
    std::fs::write(&file, before).ok();
    let (mut app, _) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    app.update();
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);

    mcp(
        &mut app,
        agent,
        "add --user web https://example.invalid/mcp",
    );
    let after = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(
        after,
        "{\n    \"theme\": [1, 2],\n    \"mcpServers\": {\n        \"kept\": { \"command\": \"nope\", \"disabled\": true },\n        \"web\": {\n            \"type\": \"http\",\n            \"url\": \"https://example.invalid/mcp\"\n        }\n    }\n}\n"
    );
    assert!(status(app.world_mut(), "web").is_some());

    mcp(&mut app, agent, "remove web");
    assert_eq!(std::fs::read_to_string(&file).unwrap_or_default(), before);
    assert!(status(app.world_mut(), "web").is_none());

    // A project server the user adds starts as added.
    mcp(
        &mut app,
        agent,
        "add --project local /no/such/server --flag",
    );
    let written = std::fs::read_to_string(project.join(".rig/mcp.json")).unwrap_or_default();
    let written: Value = serde_json::from_str(&written).unwrap_or_default();
    assert_eq!(
        written,
        json!({ "mcpServers": { "local": { "command": "/no/such/server", "args": ["--flag"] } } })
    );
    assert_eq!(scope(app.world_mut(), "local"), Some(Scope::Project));

    // A file that is not JSON is left alone.
    std::fs::write(&file, "{ not json").ok();
    mcp(&mut app, agent, "add other nope");
    assert_eq!(
        std::fs::read_to_string(&file).unwrap_or_default(),
        "{ not json"
    );
    let refused = &app.world().resource::<Collected<Recalled>>().0;
    let why = refused.last().and_then(|recalled| recalled.why.as_deref());
    assert!(
        why.is_some_and(|why| why.contains("is not JSON")),
        "{why:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

/// How the headless run of `prompt` with `--tools tools` on `project`
/// ended, after the plugin connected its servers, and the model.
fn headless(
    project: &Path,
    settings: &str,
    tools: &str,
    model: &MockCompletionModel,
) -> (App, Option<HeadlessEnded>) {
    headless_with(project, settings, tools, model, |_| {})
}

/// [`headless`], with `add` adding to the app first.
fn headless_with(
    project: &Path,
    settings: &str,
    tools: &str,
    model: &MockCompletionModel,
    add: impl FnOnce(&mut App),
) -> (App, Option<HeadlessEnded>) {
    let invoked = Invocation {
        print: Some("Add 2 and 3.".to_owned()),
        flags: vec![(TOOLS.to_owned(), Some(tools.to_owned()))],
        ..Invocation::default()
    };
    let (mut app, wakes) = app_in(project, settings, Some(invoked));
    app.insert_resource(scripted_models([(MODEL, model)]))
        .add_plugins((ExitPlugin, PrintPlugin))
        .init_resource::<Ended>()
        .add_observer(|ended: On<HeadlessEnded>, mut log: ResMut<Ended>| {
            log.0 = Some(ended.event().clone());
        });
    add(&mut app);
    app.finish();
    app.cleanup();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.is_some()
    });
    let ended = app.world().resource::<Ended>().0.clone();
    (app, ended)
}

/// How a headless run ended.
#[derive(Resource, Default)]
struct Ended(Option<HeadlessEnded>);

#[test]
fn a_headless_run_waits_for_the_servers_its_tools_name() {
    let project = project("headless");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "stub": stub_server(json!({})) }),
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "stub__add", json!({ "a": 2, "b": 3 })),
        reply("5"),
    ]);
    let (_app, ended) = headless(&project, "", "stub__add", &model);
    let how = ended.map(|ended| (ended.code, ended.answer, ended.why));
    assert_eq!(how, Some((0, Some("5".to_owned()), None)));
    let first = model.requests().first().map(|request| {
        let names = request.tools.iter().map(|tool| tool.name.to_string());
        names.collect::<Vec<_>>()
    });
    assert_eq!(first, Some(vec!["stub__add".to_owned()]));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_headless_run_waits_for_no_held_server() {
    let project = project("headless-held");
    let marker = project.join("started");
    write_servers(
        &project.join("home/mcp.json"),
        json!({
            "held": stub_server(json!({ MARKER: marker })),
            "stub": stub_server(json!({})),
        }),
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "stub__add", json!({ "a": 2, "b": 3 })),
        reply("5"),
    ]);
    let (_app, ended) = headless_with(&project, "", "stub__add", &model, |app| {
        app.add_observer(hold_held);
    });
    let how = ended.map(|ended| (ended.code, ended.answer));
    assert_eq!(how, Some((0, Some("5".to_owned()))));
    assert!(!marker.exists(), "a held server started");
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_change_said_while_the_first_list_comes_is_in_before_the_run_starts() {
    let project = project("changes");
    let server = stub_server(json!({ CHANGES: "1" }));
    write_servers(&project.join("home/mcp.json"), json!({ "stub": server }));
    let model = MockCompletionModel::from_stream_turns([reply("ok")]);
    let (_app, ended) = headless(&project, "", "stub__mul", &model);
    assert_eq!(ended.map(|ended| ended.code), Some(0));
    let first = model.requests().first().map(|request| {
        let names = request.tools.iter().map(|tool| tool.name.to_string());
        names.collect::<Vec<_>>()
    });
    assert_eq!(first, Some(vec!["stub__mul".to_owned()]));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_list_left_unanswered_holds_back_no_later_change() {
    let project = project("hangs");
    let server = stub_server(json!({ HANGS: "1" }));
    write_servers(&project.join("home/mcp.json"), json!({ "stub": server }));
    let settings = "[mcp]\nconnect_timeout_secs = 1";
    let (mut app, wakes) = app_in(&project, settings, None);
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "stub__mul"));
    let told = notices(app.world());
    let unlisted =
        "its tools could not be listed again: no answer within 1 s (`[mcp] connect_timeout_secs`)";
    assert!(told.iter().any(|text| text.contains(unlisted)), "{told:?}");
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_headless_run_gives_up_on_a_server_that_fails_or_never_answers() {
    let project = project("headless-fails");
    let servers = json!({
        "broken": { "command": "/no/such/mcp-server" },
        "silent": stub_server(json!({ SILENT: "1" })),
    });
    write_servers(&project.join("home/mcp.json"), servers);
    let model = MockCompletionModel::from_stream_turns([reply("never asked")]);
    let started = Instant::now();
    let settings = "[mcp]\nconnect_timeout_secs = 1\nfirst_retry_secs = 3600";
    let (app, ended) = headless(&project, settings, "broken__run", &model);
    assert!(started.elapsed() < Duration::from_secs(10));
    let ended = ended.map(|ended| (ended.code, ended.why));
    let why = ended
        .as_ref()
        .and_then(|(_, why)| why.clone())
        .unwrap_or_default();
    assert_eq!(ended.map(|(code, _)| code), Some(2));
    assert!(
        why.starts_with("--tools: unknown tool `broken__run`"),
        "{why}"
    );
    assert_eq!(model.request_count(), 0);
    let told = notices(app.world());
    let failed = |name: &str| {
        told.iter()
            .any(|text| text.starts_with(&format!("MCP server `{name}` could not connect: ")))
    };
    assert!(failed("broken") && failed("silent"), "{told:?}");
    assert!(
        told.iter()
            .any(|text| text.contains("no answer within 1 s (`[mcp] connect_timeout_secs`)")),
        "{told:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_project_server_takes_the_users_place_until_removed() {
    let project = project("same-name");
    let (user, theirs) = (project.join("home/mcp.json"), project.join(".rig/mcp.json"));
    write_servers(
        &user,
        json!({ "stub": { "command": "/no/such/user-stub" } }),
    );
    write_servers(&theirs, json!({ "stub": stub_server(json!({})) }));
    let users = std::fs::read_to_string(&user).unwrap_or_default();
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    assert_eq!(scope(app.world_mut(), "stub"), Some(Scope::Project));
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    // A word after the name is refused, and nothing is removed.
    mcp(&mut app, agent, "remove stub --project");
    assert!(has(app.world(), "stub__add"));

    // A removal without a file takes the one shown, and the other's comes.
    mcp(&mut app, agent, "remove stub");
    let told = notices(app.world());
    let back = "takes its place; /mcp shows it.";
    assert!(told.iter().any(|text| text.ends_with(back)), "{told:?}");
    assert!(
        !std::fs::read_to_string(&theirs)
            .unwrap_or_default()
            .contains("stub")
    );
    assert_eq!(std::fs::read_to_string(&user).unwrap_or_default(), users);
    run_until(&mut app, &wakes, |world| !has(world, "stub__add"));
    assert_eq!(scope(app.world_mut(), "stub"), Some(Scope::User));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_project_on_another_host_has_no_project_servers_here() {
    let project = project("remote");
    let local = project.join(".rig/mcp.json");
    write_servers(&local, json!({ "here": stub_server(json!({})) }));
    let before = std::fs::read_to_string(&local).unwrap_or_default();
    let (mut app, _) = app_in(&project, "", None);
    let remote = rig_harness::rig_tools::remote::Remote::new(
        "bench",
        ["false"],
        rig_harness::rig_tools::process::Launcher::default,
    );
    assert!(remote.is_ok(), "{remote:?}");
    if let Ok(remote) = remote {
        app.insert_resource(ProjectWorkspace(Arc::new(remote)));
    }
    app.finish();
    app.update();
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    assert!(status(app.world_mut(), "here").is_none());
    mcp(&mut app, agent, "add --project there /no/such/server");
    mcp(&mut app, agent, "remove --project here");
    assert_eq!(std::fs::read_to_string(&local).unwrap_or_default(), before);
    let refused = &app.world().resource::<Collected<Recalled>>().0;
    let why = refused.last().and_then(|recalled| recalled.why.as_deref());
    assert!(
        why.is_some_and(|why| why.contains("on another host")),
        "{why:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

/// Whether the process `pid` runs: there, and no zombie waiting for its
/// new parent, as a container's first process may leave it.
#[cfg(target_os = "linux")]
fn alive(pid: &str) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        let state = stat.rsplit(')').next().unwrap_or_default().trim_start();
        !state.starts_with('Z')
    })
}

#[cfg(target_os = "linux")]
#[test]
fn what_a_server_started_goes_with_it() {
    // As the server is removed, and as the app exits, which waits for no
    // task of the server's.
    for exit in [false, true] {
        let project = project(if exit { "group-exit" } else { "group" });
        let pidfile = project.join("child");
        let server = stub_server(json!({ CHILD: pidfile }));
        write_servers(&project.join("home/mcp.json"), json!({ "stub": server }));
        let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
        app.finish();
        run_until(&mut app, &wakes, |world| has(world, "stub__add"));
        let child = std::fs::read_to_string(&pidfile).unwrap_or_default();
        assert!(alive(&child));
        let agent =
            rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
        if exit {
            app.world_mut().write_message(AppExit::Success);
            app.update();
        } else {
            mcp(&mut app, agent, "remove stub");
        }
        let waited = Instant::now();
        while alive(&child) && waited.elapsed() < Duration::from_secs(10) {
            if !exit {
                app.update();
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let left = alive(&child);
        if left {
            std::process::Command::new("kill").arg(&child).status().ok();
        }
        assert!(!left, "the server's child outlived it (exit: {exit})");
        drop(app);
        std::fs::remove_dir_all(project).ok();
    }
}

/// Started in a folder of the project, rig reads the project's file.
#[test]
fn a_folder_of_a_project_has_the_projects_servers() {
    let project = project("folder");
    let dir = project.join("sub/dir");
    std::fs::create_dir_all(dir.join("home")).ok();
    let server = stub_server(json!({}));
    write_servers(&project.join(".rig/mcp.json"), json!({ "stub": server }));
    let (mut app, _) = app_in(&dir, "", None);
    app.finish();
    app.update();
    assert_eq!(scope(app.world_mut(), "stub"), Some(Scope::Project));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_project_in_the_home_directory_has_no_project_file() {
    // Rig run in the home directory: `~/.rig` is the user's own.
    let project = project("home-dir");
    let home = project.join(".rig");
    write_servers(
        &home.join("mcp.json"),
        json!({ "mine": { "command": "x", "disabled": true } }),
    );
    let (mut app, _) = app_in(&project, "", None);
    // As rig finds it run in the home directory, which is the project.
    let found = Project::find(&project, &home, Some(&project));
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(found);
    app.finish();
    app.update();
    assert_eq!(scope(app.world_mut(), "mine"), Some(Scope::User));
    let agent =
        rig_harness_test_support::first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    mcp(&mut app, agent, "add --project theirs x");
    let refused = &app.world().resource::<Collected<Recalled>>().0;
    let why = refused.last().and_then(|recalled| recalled.why.as_deref());
    assert!(
        why.is_some_and(|why| why.contains(".rig is RIG_HOME, or it is your home folder")),
        "{why:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_server_that_offers_no_tools_connects_with_none() {
    let project = project("no-tools");
    let server = stub_server(json!({ NO_TOOLS: "1" }));
    write_servers(&project.join("home/mcp.json"), json!({ "docs": server }));
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    let connected = |world: &World| {
        world
            .try_query::<(&McpServer, &McpStatus)>()
            .is_some_and(|mut servers| {
                servers
                    .iter(world)
                    .any(|(_, status)| status.state == McpState::Connected)
            })
    };
    run_until(&mut app, &wakes, connected);
    let told = notices(app.world());
    assert!(
        told.contains(&"MCP server `docs` connected: 0 tools.".to_owned()),
        "{told:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_server_that_knows_only_the_released_protocol_connects() {
    let project = project("released");
    let server = stub_server(json!({ RELEASED: "1" }));
    write_servers(&project.join("home/mcp.json"), json!({ "gh": server }));
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        notices(world).iter().any(|notice| notice.contains("`gh`"))
    });
    let told = notices(app.world());
    assert!(
        told.contains(&"MCP server `gh` connected: 2 tools.".to_owned()),
        "{told:?}"
    );
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_server_that_answers_an_older_protocol_is_spoken_to_in_it() {
    let project = project("older");
    let server = stub_server(json!({ OLDER: "1" }));
    write_servers(&project.join("home/mcp.json"), json!({ "old": server }));
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        notices(world).iter().any(|notice| notice.contains("`old`"))
    });
    let told = notices(app.world());
    assert!(
        told.contains(&"MCP server `old` connected: 2 tools.".to_owned()),
        "{told:?}"
    );
    let world = app.world_mut();
    let versions: Vec<_> = world
        .query::<&McpLink>()
        .iter(world)
        .filter_map(|link| link.0.peer().peer_info())
        .map(|info| info.protocol_version.to_string())
        .collect();
    assert_eq!(versions, ["2025-06-18"]);
    std::fs::remove_dir_all(project).ok();
}

#[cfg(target_os = "linux")]
#[test]
fn as_rig_exits_every_server_process_is_killed_and_none_starts() {
    let groups = Groups::default();
    // The stub, silent, as a stdio server is started.
    let stub = || {
        let server = stub_server(json!({ SILENT: "1" }));
        let text = |key: &str| {
            server
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let args = server
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut command = Processes::default().launcher().command(text("command"));
        command
            .args(args.iter().filter_map(Value::as_str))
            .envs([(STUB, "1"), (SILENT, "1")])
            .stdin(Stdio::piped());
        command
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    assert!(runtime.is_ok(), "{runtime:?}");
    let Ok(runtime) = runtime else { return };
    let _entered = runtime.enter();
    let running = groups.spawn(stub());
    let pid = running
        .as_ref()
        .map(|it| it.id().to_string())
        .unwrap_or_default();
    assert!(alive(&pid));
    groups.close();
    let waited = Instant::now();
    while alive(&pid) && waited.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!alive(&pid), "{pid} outlived the exit");
    // Nothing starts once closed.
    let late = groups.spawn(stub());
    assert!(late.is_err_and(|error| error.to_string() == "rig is exiting"));
    drop(running);
}

/// The stub as a client names it, `env` added to its environment.
fn client_stub(name: &str, env: Value) -> AddMcpServer {
    let written = stub_server(env);
    let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
    let list = written.get("args").and_then(Value::as_array);
    let vars = written.get("env").and_then(Value::as_object);
    AddMcpServer {
        name: name.to_owned(),
        spec: Ok(ServerSpec::Stdio {
            command: written.get("command").map(text).unwrap_or_default(),
            args: list.into_iter().flatten().map(text).collect(),
            env: vars
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.clone(), text(v)))
                .collect(),
            cwd: None,
        }),
    }
}

/// Names `servers` for the app as a client does, before it starts.
fn name_clients(app: &mut App, servers: Vec<AddMcpServer>) {
    app.add_systems(PreStartup, move |mut commands: Commands| {
        for server in &servers {
            commands.trigger(server.clone());
        }
    });
}

#[test]
fn a_client_server_is_taken_as_given_and_no_file_is_read_without_the_plugin() {
    let project = project("client");
    let mine = project.join("mine-started");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "mine": stub_server(json!({ MARKER: mine })) }),
    );
    // Taken as given: the stub writes its marker at the path as written.
    let literal = project.join("${NOT_EXPANDED}");
    let (mut app, wakes) = app_with::<McpCorePlugin>(&project, "", None);
    name_clients(
        &mut app,
        vec![client_stub("files", json!({ MARKER: literal }))],
    );
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "files__add"));
    assert!(literal.exists() && !mine.exists());
    assert_eq!(scope(app.world_mut(), "files"), Some(Scope::Client));
    let agent = rig_harness_test_support::first_agent(app.world_mut());
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    mcp(&mut app, agent, "");
    let told = notices(app.world());
    let listed = "files (client): connected, 2 tools".to_owned();
    assert_eq!(told.last(), Some(&listed), "{told:?}");
    mcp(&mut app, agent, "add other /bin/true");
    let refused = &app.world().resource::<Collected<Recalled>>().0;
    let why = refused.last().and_then(|recalled| recalled.why.as_deref());
    assert!(why.is_some_and(|why| why.starts_with("/mcp add changes an mcp.json")));
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_client_server_takes_its_name_over_the_users_and_the_projects() {
    let project = project("client-names");
    let (user, theirs) = (project.join("home/mcp.json"), project.join(".rig/mcp.json"));
    let (mine, ours) = (
        project.join("user-started"),
        project.join("project-started"),
    );
    let (first, second) = (
        project.join("first-started"),
        project.join("second-started"),
    );
    write_servers(
        &user,
        json!({ "files": stub_server(json!({ MARKER: mine })) }),
    );
    write_servers(
        &theirs,
        json!({ "files": stub_server(json!({ MARKER: ours })) }),
    );
    let (mut app, wakes) = app_in(&project, "", None);
    name_clients(
        &mut app,
        vec![
            client_stub("files", json!({ MARKER: first })),
            client_stub("files", json!({ MARKER: second })),
        ],
    );
    app.finish();
    run_until(&mut app, &wakes, |world| has(world, "files__add"));
    let agent = rig_harness_test_support::first_agent(app.world_mut());
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    // A file's server of the name stays stopped whatever /mcp does.
    mcp(&mut app, agent, "remove --user files");
    mcp(&mut app, agent, "");
    assert!(!mine.exists() && !ours.exists());
    assert!(first.exists() && !second.exists());
    assert_eq!(scope(app.world_mut(), "files"), Some(Scope::Client));
    // The project's server stands for the user's, and gives way.
    let told = notices(app.world());
    assert!(
        told.contains(&gives_way("files", Scope::Project)),
        "{told:?}"
    );
    let listed = told.last().cloned().unwrap_or_default();
    assert!(
        listed.contains("files (client): connected, 2 tools"),
        "{listed}"
    );
    let gave_way = "files (project): does not run; the client's of that name does";
    assert!(listed.contains(gave_way), "{listed}");
    let second = "MCP server `files`: the client named two; the first stays.";
    assert!(told.contains(&second.to_owned()), "{told:?}");
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_client_server_that_fails_holds_the_start_no_longer_than_its_attempt() {
    let project = project("client-fails");
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "good__add", json!({ "a": 2, "b": 3 })),
        reply("5"),
    ]);
    let invoked = Invocation {
        print: Some("Add 2 and 3.".to_owned()),
        flags: vec![(TOOLS.to_owned(), Some("good__add".to_owned()))],
        ..Invocation::default()
    };
    let settings = "[mcp]\nconnect_timeout_secs = 1\nfirst_retry_secs = 3600";
    let (mut app, wakes) = app_with::<McpCorePlugin>(&project, settings, Some(invoked));
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .add_plugins((ExitPlugin, PrintPlugin))
        .init_resource::<Ended>()
        .add_observer(|ended: On<HeadlessEnded>, mut log: ResMut<Ended>| {
            log.0 = Some(ended.event().clone());
        });
    let mut broken = client_stub("broken", json!({}));
    if let Ok(ServerSpec::Stdio { command, .. }) = &mut broken.spec {
        "/no/such/mcp-server".clone_into(command);
    }
    let silent = client_stub("silent", json!({ SILENT: "1" }));
    name_clients(
        &mut app,
        vec![broken, silent, client_stub("good", json!({}))],
    );
    let started = Instant::now();
    app.finish();
    app.cleanup();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.is_some()
    });
    assert!(started.elapsed() < Duration::from_secs(10));
    let ended = app.world().resource::<Ended>().0.clone();
    let ended = ended.map(|ended| (ended.code, ended.answer));
    assert_eq!(ended, Some((0, Some("5".to_owned()))));
    let told = notices(app.world());
    let failed = |name: &str| {
        let start = format!("MCP server `{name}` could not connect: ");
        told.iter().any(|text| text.starts_with(&start))
    };
    assert!(failed("broken") && failed("silent"), "{told:?}");
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_client_server_that_fails_or_is_refused_still_keeps_the_users_from_its_name() {
    let project = project("client-fails-name");
    let mine = project.join("user-started");
    let user = project.join("home/mcp.json");
    write_servers(
        &user,
        json!({ "files": stub_server(json!({ MARKER: mine })) }),
    );
    let mut broken = client_stub("files", json!({}));
    if let Ok(ServerSpec::Stdio { command, .. }) = &mut broken.spec {
        "/no/such/mcp-server".clone_into(command);
    }
    let refused = AddMcpServer {
        name: "files".to_owned(),
        spec: Err("not a transport rig speaks".to_owned()),
    };
    for client in [broken, refused] {
        let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
        name_clients(&mut app, vec![client]);
        app.finish();
        run_until(&mut app, &wakes, |world| {
            let servers = world.try_query::<(&McpServer, &McpStatus)>();
            servers.is_some_and(|mut servers| {
                servers.iter(world).any(|(server, status)| {
                    let failed = matches!(status.state, McpState::Retrying | McpState::Invalid);
                    server.name == "files" && failed
                })
            })
        });
        assert_eq!(scope(app.world_mut(), "files"), Some(Scope::Client));
        assert!(!mine.exists());
    }
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_run_with_no_session_starts_no_server_of_the_files() {
    let project = project("sessionless");
    let mine = project.join("user-started");
    let user = project.join("home/mcp.json");
    write_servers(
        &user,
        json!({ "files": stub_server(json!({ MARKER: mine })) }),
    );
    let invoked = Invocation {
        command: Some("stats".to_owned()),
        ..Invocation::default()
    };
    let (mut app, _wakes) = app_in(&project, "", Some(invoked));
    let front = Front::new("stats")
        .sessionless()
        .on_command("stats", "", "Count");
    app.add_front(front);
    assert!(app.runs_front("stats"));
    app.finish();
    for _ in 0..3 {
        app.update();
    }
    assert!(status(app.world_mut(), "files").is_none());
    assert!(!mine.exists());
    std::fs::remove_dir_all(project).ok();
}

/// Done item: `--tools <server>__<tool>` at a terminal waits for the server
/// as a headless run does, where it was refused before the server
/// connected.
#[test]
fn a_terminal_run_can_name_a_servers_tool_in_its_tools() {
    let project = project("terminal");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "stub": stub_server(json!({})) }),
    );
    let model = MockCompletionModel::from_stream_turns([reply("ok")]);
    let invoked = Invocation {
        flags: vec![(TOOLS.to_owned(), Some("stub__add".to_owned()))],
        ..Invocation::default()
    };
    let (mut app, wakes) = app_in(&project, "", Some(invoked));
    app.world_mut().resource_mut::<Invoked>().terminal = true;
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .add_front(Front::new("view").on_terminal())
        .add_plugins((ExitPlugin, PrintPlugin));
    let agent = spawn_root(app.world_mut()).id();
    app.finish();
    app.cleanup();
    run_until(&mut app, &wakes, |world| has(world, "stub__add"));
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: MODEL.to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    let ask = Deliver::new(agent, "Add 2 and 3.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(&mut app, &wakes, |_| model.request_count() > 0);
    assert_eq!(app.should_exit(), None);
    let first = model.requests().first().map(|request| {
        let names = request.tools.iter().map(|tool| tool.name.to_string());
        names.collect::<Vec<_>>()
    });
    assert_eq!(first, Some(vec!["stub__add".to_owned()]));
    std::fs::remove_dir_all(project).ok();
}

/// A server whose list of tools is past the limit, after an answer that
/// starts with a byte-order mark, is lost saying so: the reason survives
/// the list that ended the connection.
#[test]
fn a_tool_list_past_the_limit_loses_the_server_saying_why() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build();
    assert!(runtime.is_ok(), "a tokio runtime");
    let Ok(runtime) = runtime else { return };
    let exe = std::env::current_exe().unwrap_or_default();
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    let env = [(STUB, "1"), (LONG, "1")].map(|(key, value)| (key.to_owned(), value.to_owned()));
    let start = Start {
        spec: ServerSpec::Stdio {
            command: exe.to_string_lossy().into_owned(),
            args: vec![
                format!("{module}::stub"),
                "--exact".to_owned(),
                "--ignored".to_owned(),
                "--nocapture".to_owned(),
                "--test-threads=1".to_owned(),
            ],
            env: env.into_iter().collect(),
            cwd: None,
        },
        processes: Processes::default(),
        groups: Groups::default(),
        root: project("long-list"),
        limit: Duration::from_secs(20),
        line_limit: 1024,
    };
    let (sender, _events) = async_channel::bounded(EVENTS);
    let ran = runtime.block_on(start.run(&Outbox(sender)));
    let why = ran.err().unwrap_or_default();
    assert!(why.contains("more than 1024 bytes"), "{why}");
}

/// A wait of 0 seconds does not read, so the user's settings refuse the
/// run naming the key, rather than it becoming 1.
#[test]
fn a_wait_of_zero_seconds_does_not_read() {
    for key in ["connect_timeout_secs", "first_retry_secs"] {
        let settings = Settings::from_toml(&format!("[mcp]\n{key} = 0"));
        let read = settings.get::<McpSettings>("mcp");
        assert!(
            read.as_ref().is_err_and(|why| why.contains("[mcp]")),
            "{key}: {:?}",
            read.err()
        );
    }
}

/// The header every request to [`Loopback`] must carry, and its value.
const TOKEN_HEADER: &str = "x-rig-token";
const TOKEN: &str = "loopback-secret";

/// An MCP server over streamable HTTP in this process: `add(a, b)` answers
/// with the sum, and `mul` is listed once `grown` is set.
#[derive(Clone, Default)]
struct Loopback {
    grown: Arc<AtomicBool>,
    /// The client of each session, once it said it is ready.
    peers: Arc<Mutex<Vec<rmcp::Peer<rmcp::RoleServer>>>>,
}

impl rmcp::ServerHandler for Loopback {
    fn get_info(&self) -> rmcp::model::ServerConfig {
        let capabilities = rmcp::model::ServerCapabilities::builder()
            .enable_tools()
            .enable_tool_list_changed()
            .build();
        rmcp::model::ServerConfig::new(capabilities)
    }

    async fn on_initialized(&self, context: NotificationContext<rmcp::RoleServer>) {
        let mut peers = self.peers.lock().unwrap_or_else(PoisonError::into_inner);
        peers.push(context.peer);
    }

    async fn list_tools(
        &self,
        _: Option<rmcp::model::PaginatedRequestParams>,
        _: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let object = json!({
            "type": "object",
            "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
        });
        let schema = Arc::new(object.as_object().cloned().unwrap_or_default());
        let mut tools = vec![McpDefinition::new(
            "add",
            "Adds two numbers.",
            schema.clone(),
        )];
        if self.grown.load(Ordering::SeqCst) {
            tools.push(McpDefinition::new("mul", "Multiplies two numbers.", schema));
        }
        Ok(rmcp::model::ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        _: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        let number = |key| {
            let arguments = request.arguments.as_ref();
            let value = arguments.and_then(|arguments| arguments.get(key));
            value.and_then(Value::as_f64).unwrap_or_default()
        };
        let sum = number("a") + number("b");
        let text = rmcp::model::ContentBlock::text(sum.to_string());
        Ok(rmcp::model::CallToolResult::success(vec![text]).into())
    }
}

/// Refuses a request without the token, counting it in `refused`.
async fn token_only(
    refused: Arc<AtomicUsize>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let token = request.headers().get(TOKEN_HEADER);
    if token.is_some_and(|token| token == TOKEN) {
        return next.run(request).await;
    }
    refused.fetch_add(1, Ordering::SeqCst);
    axum::response::IntoResponse::into_response(http::StatusCode::UNAUTHORIZED)
}

/// The plugin uses rmcp's default reqwest client, which reads `HTTP_PROXY`,
/// `ALL_PROXY` and `NO_PROXY` (and their lowercase forms): on a machine with
/// a proxy and no `NO_PROXY` for 127.0.0.1, this test goes through the proxy.
#[test]
fn a_streamable_http_server_connects_lists_answers_and_changes_its_tools() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build();
    assert!(runtime.is_ok(), "{runtime:?}");
    let Ok(runtime) = runtime else { return };
    let server = Loopback::default();
    let refused = Arc::new(AtomicUsize::new(0));
    let factory = server.clone();
    let service = rmcp::transport::StreamableHttpService::new(
        move || Ok(factory.clone()),
        Arc::new(
            rmcp::transport::streamable_http_server::session::local::LocalSessionManager::default(),
        ),
        rmcp::transport::StreamableHttpServerConfig::default(),
    );
    let counted = refused.clone();
    let router =
        axum::Router::new()
            .nest_service("/mcp", service)
            .layer(axum::middleware::from_fn(move |request, next| {
                token_only(counted.clone(), request, next)
            }));
    let listener = runtime.block_on(tokio::net::TcpListener::bind("127.0.0.1:0"));
    let bound = listener.and_then(|listener| Ok((listener.local_addr()?, listener)));
    assert!(bound.is_ok(), "{bound:?}");
    let Ok((address, listener)) = bound else {
        return;
    };
    runtime.spawn(async move { axum::serve(listener, router).await });

    let project = project("http");
    let url = format!("http://{address}/mcp");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "web": { "type": "http", "url": url, "headers": { TOKEN_HEADER: TOKEN } } }),
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "web__add", json!({ "a": 2, "b": 3 })),
        reply("2 + 3 = 5."),
    ]);
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.init_resource::<Outputs>().add_observer(
        |settled: On<Add<Settled>>,
         calls: Query<&Outcome, With<ToolRequest>>,
         mut log: ResMut<Outputs>| {
            if let Ok(outcome) = calls.get(settled.entity) {
                log.0.push(outcome.tool_result().output().render());
            }
        },
    );
    app.finish();
    app.update();
    let agent = rig_harness_test_support::first_agent(app.world_mut());
    let connection = connect(&model);
    assert!(agent.is_some() && connection.is_some());
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

    run_until(&mut app, &wakes, |world| has(world, "web__add"));
    let told = notices(app.world());
    assert!(
        told.contains(&"MCP server `web` connected: 1 tool.".to_owned()),
        "{told:?}"
    );
    app.world_mut()
        .trigger(Deliver::new(agent, "Add 2 and 3.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && !world.entity(agent).contains::<ActiveTurn>()
    });
    assert_eq!(app.world().resource::<Outputs>().0, ["5"]);

    // The server's word that its tools changed, outside any request of the
    // client's, brings the new list.
    server.grown.store(true, Ordering::SeqCst);
    let peers = server
        .peers
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(peers.len(), 1);
    for peer in peers {
        let said = runtime.block_on(peer.notify_tool_list_changed());
        assert!(said.is_ok(), "{said:?}");
    }
    run_until(&mut app, &wakes, |world| has(world, "web__mul"));
    let state = status(app.world_mut(), "web").map(|status| (status.state, status.tools));
    assert_eq!(state, Some((McpState::Connected, 2)));
    // Every request carried the header.
    assert_eq!(refused.load(Ordering::SeqCst), 0);
    drop(app);
    runtime.shutdown_background();
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn a_server_that_refuses_its_client_says_why_in_plain_words() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build();
    assert!(runtime.is_ok(), "{runtime:?}");
    let Ok(runtime) = runtime else { return };
    // Answers as a server that takes OAuth does: 401 to every request, and
    // 403 on `/scoped`, each with a bearer challenge. `/bare` answers 401
    // with no challenge, and `/garbled` a JSON type with HTML in it.
    let router = axum::Router::new()
        .route(
            "/scoped",
            axum::routing::any(|| async {
                let challenge = "Bearer error=\"insufficient_scope\", scope=\"repo\"";
                (
                    http::StatusCode::FORBIDDEN,
                    [(http::header::WWW_AUTHENTICATE, challenge)],
                )
            }),
        )
        .route(
            "/bare",
            axum::routing::any(|| async { http::StatusCode::UNAUTHORIZED }),
        )
        .route(
            "/garbled",
            axum::routing::any(|| async {
                let json = [(http::header::CONTENT_TYPE, "application/json")];
                (json, "<html>oops</html>")
            }),
        )
        .fallback(|| async {
            let challenge = "Bearer resource_metadata=\"http://127.0.0.1/.well-known/oauth-protected-resource\"";
            (
                http::StatusCode::UNAUTHORIZED,
                [(http::header::WWW_AUTHENTICATE, challenge)],
            )
        });
    let listener = runtime.block_on(tokio::net::TcpListener::bind("127.0.0.1:0"));
    let bound = listener.and_then(|listener| Ok((listener.local_addr()?, listener)));
    assert!(bound.is_ok(), "{bound:?}");
    let Ok((address, listener)) = bound else {
        return;
    };
    runtime.spawn(async move { axum::serve(listener, router).await });
    // A port nothing listens on once its listener is dropped.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").and_then(|free| free.local_addr());
    assert!(closed.is_ok(), "{closed:?}");
    let Ok(closed) = closed else { return };

    let project = project("refused");
    let token = json!({ "Authorization": "Bearer expired" });
    let servers = json!({
        "conf": { "type": "http", "url": format!("http://{address}/mcp") },
        "keyed": { "type": "http", "url": format!("http://{address}/mcp"), "headers": token },
        "scoped": { "type": "http", "url": format!("http://{address}/scoped"), "headers": token },
        "bare": { "type": "http", "url": format!("http://{address}/bare") },
        "barekeyed": { "type": "http", "url": format!("http://{address}/bare"), "headers": token },
        "garbled": { "type": "http", "url": format!("http://{address}/garbled") },
        "gone": { "type": "http", "url": format!("http://{closed}/mcp") },
        "readonly": {
            "type": "http",
            "url": format!("http://{address}/mcp"),
            "headers": { "X-MCP-Readonly": "true" },
        },
    });
    write_servers(&project.join("home/mcp.json"), servers);
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    run_until(&mut app, &wakes, |world| notices(world).len() == 8);
    let (gone, mut told): (Vec<_>, Vec<_>) = notices(app.world())
        .into_iter()
        .partition(|notice| notice.starts_with("MCP server `gone`"));
    told.sort();
    let gone = gone.concat();
    assert!(
        gone.starts_with("MCP server `gone` could not connect: ")
            && gone.contains(&format!(
                "error sending request for url (http://{closed}/mcp)"
            ))
            && !gone.contains("rmcp::"),
        "{gone}"
    );
    assert_eq!(
        told,
        [
            "MCP server `bare` could not connect: it needs a token (HTTP 401). A server that \
             takes a token in a header works: see \"Servers: `mcp.json`\" in the rig-mcp-client \
             README.",
            "MCP server `barekeyed` could not connect: it refused the token in its `headers` \
             (HTTP 401).",
            "MCP server `conf` could not connect: it asks for a sign-in (OAuth), which rig \
                 does not support. A server that takes a token in a header works: see \"Servers: \
                 `mcp.json`\" in the rig-mcp-client README.",
            "MCP server `garbled` could not connect: its answer was not MCP JSON (expected \
             value at line 1 column 1: <html>oops</html>).",
            "MCP server `keyed` could not connect: it refused the token in its `headers` \
                 (HTTP 401).",
            "MCP server `readonly` could not connect: it asks for a sign-in (OAuth), which rig \
             does not support. A server that takes a token in a header works: see \"Servers: \
             `mcp.json`\" in the rig-mcp-client README.",
            "MCP server `scoped` could not connect: the token in its `headers` lacks a \
                 permission it needs (HTTP 403, insufficient scope).",
        ],
    );
    drop(app);
    runtime.shutdown_background();
    std::fs::remove_dir_all(project).ok();
}

#[cfg(unix)]
#[test]
fn a_server_that_exits_at_once_says_so() {
    let project = project("exits");
    write_servers(
        &project.join("home/mcp.json"),
        json!({ "quits": { "command": "true" } }),
    );
    let (mut app, wakes) = app_in(&project, "[mcp]\nfirst_retry_secs = 3600", None);
    app.finish();
    run_until(&mut app, &wakes, |world| !notices(world).is_empty());
    assert_eq!(
        notices(app.world()),
        ["MCP server `quits` could not connect: it closed the connection before it answered."],
    );
    std::fs::remove_dir_all(project).ok();
}
