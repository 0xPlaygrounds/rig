use std::io::{BufRead, BufReader, Write};

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::host::exit::ExitPlugin;
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::MemoryStore;
use serde_json::{Value, json};

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// Set in the environment of the fake server this test binary runs as:
/// the file it logs each text it is given to.
const STUB: &str = "CORPUS_FH_LSP_STUB";

/// Each settled call's output (or error), by call id.
#[derive(Resource, Default)]
struct Outputs(Vec<(String, String)>);

impl Outputs {
    fn of(&self, id: &str) -> String {
        let found = self.0.iter().find(|(call, _)| call == id);
        found.map(|(_, text)| text.clone()).unwrap_or_default()
    }
}

/// Reads one `Content-Length` framed message, skipping other header lines.
fn read_framed(reader: &mut impl BufRead) -> Option<Value> {
    let mut length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        match line.trim().split_once(':') {
            Some((name, value)) if name.eq_ignore_ascii_case("content-length") => {
                length = value.trim().parse::<usize>().ok();
            }
            _ if line.trim().is_empty() && length.is_some() => break,
            _ => {}
        }
    }
    let mut body = vec![0; length.unwrap_or_default()];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

/// A fake language server: run by the other test as a child, it answers
/// `Content-Length` framed JSON-RPC on stdin. A line that says `"one"`
/// where a number goes is an error; `foo` at line 3, column 5 is defined
/// at `b.rs:10:4`. Every text it is given is logged to the file `STUB`
/// names, and so is a shutdown.
#[test]
#[ignore = "the fake language server the other test starts"]
fn fake_server() {
    let Some(log) = std::env::var_os(STUB) else {
        return;
    };
    let mut out = std::io::stdout().lock();
    // libtest left its line about this test open.
    if writeln!(out).is_err() {
        return;
    }
    eprintln!("fake-lsp: ready");
    let send = |out: &mut std::io::StdoutLock<'_>, message: Value| {
        let body = message.to_string();
        write!(out, "Content-Length: {}\r\n\r\n{body}", body.len()).and_then(|()| out.flush())
    };
    let mut root = String::new();
    let mut input = BufReader::new(std::io::stdin().lock());
    while let Some(message) = read_framed(&mut input) {
        let params = message.get("params").cloned().unwrap_or_default();
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if method == "shutdown" {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&log);
            if let Ok(file) = &mut file {
                writeln!(file, "shutdown").ok();
            }
        }
        let result = match method {
            "initialize" => {
                root = params
                    .pointer("/rootUri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                json!({ "capabilities": { "textDocumentSync": 1, "definitionProvider": true } })
            }
            "textDocument/didOpen" | "textDocument/didChange" => {
                let uri = params
                    .pointer("/textDocument/uri")
                    .cloned()
                    .unwrap_or_default();
                let text = params
                    .pointer("/textDocument/text")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        params
                            .pointer("/contentChanges/0/text")
                            .and_then(Value::as_str)
                    })
                    .unwrap_or_default()
                    .to_owned();
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&log);
                if let Ok(file) = &mut file {
                    writeln!(file, "{method}\n{text}").ok();
                }
                let diagnostics: Vec<Value> = text
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains("\"one\""))
                    .map(|(at, _)| json!({
                        "range": { "start": { "line": at, "character": 17 }, "end": { "line": at, "character": 22 } },
                        "severity": 1,
                        "message": "mismatched types: expected `i32`, found `&str`",
                    }))
                    .collect();
                let publish = json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": { "uri": uri, "diagnostics": diagnostics },
                });
                if send(&mut out, publish).is_err() {
                    return;
                }
                continue;
            }
            "textDocument/definition"
                if params.get("position") == Some(&json!({ "line": 2, "character": 4 })) =>
            {
                json!({
                    "uri": format!("{root}/b.rs"),
                    "range": { "start": { "line": 9, "character": 3 }, "end": { "line": 9, "character": 6 } },
                })
            }
            "exit" => return,
            _ => Value::Null,
        };
        if let Some(id) = message.get("id").cloned() {
            let answer = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            if send(&mut out, answer).is_err() {
                return;
            }
        }
    }
}

/// Whether the process `pid` exists.
fn exists(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether the child `pid` has exited, every thread of it, and waits to be
/// reaped: its leader can show as a zombie while its other threads still
/// exit.
fn exited(pid: u32) -> bool {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .map(str::trim)
    };
    status.is_empty()
        || (field("State:").is_some_and(|state| state.starts_with('Z'))
            && field("Threads:") == Some("1"))
}

fn running(world: &World) -> Vec<(String, u32)> {
    world.resource::<LspServers>().running()
}

#[test]
fn a_fake_server_answers_follows_edits_restarts_and_shuts_down() {
    let project = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|deps| deps.join("corpus-fh-lsp-tools")))
        .unwrap_or_default()
        .join(std::process::id().to_string());
    std::fs::remove_dir_all(&project).ok();
    std::fs::create_dir_all(project.join(".rig")).ok();
    let project = std::fs::canonicalize(&project).unwrap_or(project);
    let log = project.join("received.log");
    let exe = std::env::current_exe().unwrap_or_default();
    let command = format!(
        "env {STUB}={} {} {} --exact --ignored --nocapture --test-threads=1",
        log.display(),
        exe.display(),
        test_path("fake_server"),
    );
    std::fs::write(
        project.join(".rig/lsp.toml"),
        format!("rs = {}", toml::Value::from(command)),
    )
    .ok();
    std::fs::write(
        project.join("a.rs"),
        "fn main() {\n    let x: i32 = \"one\";\n    foo();\n}\n",
    )
    .ok();

    let both = vec![
        MockStreamEvent::tool_call("c1", "lsp_diagnostics", json!({ "path": "a.rs" })),
        MockStreamEvent::tool_call(
            "c2",
            "lsp_definition",
            json!({ "path": "a.rs", "line": 3, "symbol": "foo" }),
        ),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let fix = json!({ "path": "a.rs", "edits": [{ "old_text": "\"one\"", "new_text": "1" }] });
    let model = MockCompletionModel::from_stream_turns([
        both,
        calls("c3", "edit", fix),
        calls("c4", "lsp_diagnostics", json!({ "path": "a.rs" })),
        reply("Fixed."),
        calls("c5", "lsp_diagnostics", json!({ "path": "a.rs" })),
        calls("c6", "lsp_diagnostics", json!({ "path": "a.rs" })),
        reply("Back."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(project.clone()))
        .insert_resource(RigHome(Home::at(project.join("home"))))
        .add_plugins((ServicesPlugin, AgentPlugin, SessionLogPlugin, ExitPlugin))
        .add_plugins((rig_basics::coding_tools::EditTool, LspToolsPlugin))
        .init_resource::<Outputs>()
        .add_observer(
            |settled: On<Add<Settled>>,
             calls: Query<(&ToolRequest, &Outcome)>,
             mut log: ResMut<Outputs>| {
                if let Ok((request, outcome)) = calls.get(settled.entity) {
                    let result = outcome.tool_result();
                    let text = result.error().map_or_else(
                        || result.output().render(),
                        |error| format!("error: {}", error.message()),
                    );
                    log.0.push((request.call.id.to_string(), text));
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
    app.world_mut()
        .entity_mut(agent)
        .insert((connection, WorkDir(project.clone())));

    // Diagnostics and a definition by symbol; then an edit, after which
    // the server is given the new text before it is asked again.
    app.world_mut()
        .trigger(Deliver::new(agent, "Fix a.rs.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 4 && !world.entity(agent).contains::<ActiveTurn>()
    });
    let outputs = app.world().resource::<Outputs>();
    let first = outputs.of("c1");
    assert!(
        first.contains("a.rs:2:18: error: mismatched types"),
        "{first}"
    );
    assert!(first.contains("let x: i32 = \"one\";"), "{first}");
    let definition = outputs.of("c2");
    assert!(definition.contains("b.rs:10:4"), "{definition}");
    let again = outputs.of("c4");
    assert!(again.contains("no errors or warnings in a.rs"), "{again}");
    let received = std::fs::read_to_string(&log).unwrap_or_default();
    let changed = received
        .split("textDocument/didChange")
        .nth(1)
        .unwrap_or_default();
    assert!(changed.contains("let x: i32 = 1;"), "{received}");

    // Killed between calls: the next call reports the crash with the
    // server's stderr, and the one after runs on a new server.
    let first_pid = running(app.world()).first().map(|(_, pid)| *pid);
    assert!(first_pid.is_some(), "no server runs");
    let Some(first_pid) = first_pid else { return };
    let killed = std::process::Command::new("kill")
        .arg(first_pid.to_string())
        .status();
    assert!(killed.is_ok_and(|status| status.success()));
    // The signal is sent; the server is dead once all of it has exited.
    let started = std::time::Instant::now();
    while !exited(first_pid) && started.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(exited(first_pid));
    app.world_mut()
        .trigger(Deliver::new(agent, "Check again.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 7 && !world.entity(agent).contains::<ActiveTurn>()
    });
    let outputs = app.world().resource::<Outputs>();
    let crash = outputs.of("c5");
    assert!(
        crash.starts_with("error: language server `env` exited"),
        "{crash}"
    );
    assert!(crash.contains("fake-lsp: ready"), "{crash}");
    let restarted = outputs.of("c6");
    assert!(
        restarted.contains("no errors or warnings in a.rs"),
        "{restarted}"
    );
    let second_pid = running(app.world()).first().map(|(_, pid)| *pid);
    assert!(second_pid.is_some(), "no server runs after the restart");
    let Some(second_pid) = second_pid else { return };
    assert_ne!(first_pid, second_pid);

    // The session ends: the exit waits for the server to shut down.
    app.world_mut().trigger(Exit(ExitReason::Quit));
    run_until(&mut app, &wakes, |_| !exists(second_pid));
    assert!(!exists(second_pid));
    let received = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(received.ends_with("shutdown\n"), "{received}");
    std::fs::remove_dir_all(project).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LspToolsPlugin);
}

/// The test `name` of this module as its test binary names it: the module's
/// path without the crate's name.
fn test_path(name: &str) -> String {
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    format!("{module}::{name}")
}
