//! The tests run this test binary again as the agent: `rig acp` and each of
//! its session processes, with scripted models ([`agent`]); a typed ACP
//! client drives it ([`Driven`]). The lines libtest prints on stdout are not
//! ACP, and both the router and the client skip them.

#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_client_protocol::schema::{ProtocolVersion, v1 as acp};
use agent_client_protocol::{Agent, Client, ConnectionTo};
use rig_basics::approval::{ApprovalPlugin, ApprovalSystems, NeedsApproval};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness_test_support::{calls, reply, scripted_models};
use serde_json::{Value, json};

use crate::AcpPlugin;
use crate::router::{SessionCommand, lines};

mod flows;
#[cfg(target_os = "linux")]
mod lifecycle;

/// A model the test agent's catalog has and no key reaches.
const UNREACHABLE: &str = "acp-unreachable/model";

/// Set in the environment of the agent this test binary runs as.
const AGENT: &str = "RIG_ACP_TEST_AGENT";

/// Set in the environment of the MCP server this test binary runs as.
const STUB: &str = "RIG_ACP_TEST_MCP";

/// When set, the MCP server stays once its stdin closed, as a server that
/// ignores the end of its input does.
const STAYS: &str = "RIG_ACP_TEST_MCP_STAYS";

/// Set for the MCP stub to answer `initialize` only after this many
/// milliseconds.
const SLOW: &str = "RIG_ACP_TEST_MCP_SLOW";

/// The arguments that run this binary's ignored test `name` alone.
fn libtest(name: &str) -> Vec<OsString> {
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    [
        format!("{module}::{name}"),
        "--exact".to_owned(),
        "--ignored".to_owned(),
        "--nocapture".to_owned(),
        "--test-threads=1".to_owned(),
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

/// This test binary as rig with rig-acp on, `rig acp` or, with
/// `acp-session` among its arguments, a session's process. Its models are
/// scripted, one per behaviour a test picks with the `model` option.
#[test]
#[ignore = "the agent the other tests start"]
fn agent() {
    if std::env::var_os(AGENT).is_none() {
        return;
    }
    // libtest left its line about this test open on stdout.
    println!();
    let session = std::env::args().any(|arg| arg == crate::session::FRONT);
    let command = match session {
        true => crate::session::FRONT,
        false => "acp",
    };
    let script = |turns: Vec<Vec<_>>| MockCompletionModel::from_stream_turns(turns);
    let replies = script((0..32).map(|_| reply("scripted reply")).collect());
    let read = script(vec![
        calls("r1", "read", json!({ "path": "fixture.txt" })),
        reply("read it"),
    ]);
    // The id `read`'s call has, as a provider may use an id again.
    let again = script(vec![
        calls("r1", "read", json!({ "path": "missing.txt" })),
        reply("not there"),
    ]);
    // The main agent hands the guarded read to a subagent on `test/sub`.
    let task = json!({
        "description": "Read the guarded file",
        "prompt": "Read guarded.txt.",
        "model": "test/sub",
    });
    let delegate = script(vec![
        calls("d1", "task", task),
        reply("delegated"),
        reply("noted"),
        reply("noted"),
    ]);
    let sub = script(vec![
        calls("g1", "read", json!({ "path": "guarded.txt" })),
        reply("read it for you"),
    ]);
    // A provider that fails every call, retries included.
    let fail = script(
        (0..8)
            .map(|_| vec![MockStreamEvent::error("overloaded")])
            .collect(),
    );
    let guarded = script(vec![
        calls("g1", "read", json!({ "path": "guarded.txt" })),
        reply("guarded done"),
    ]);
    let wait = script(vec![
        calls("w1", "shell", json!({ "command": "sleep 120" })),
        reply("waited"),
    ]);
    // A `shell` call that ends at once.
    let quick = script(vec![
        calls("q1", "shell", json!({ "command": "true" })),
        reply("ran"),
    ]);
    let mcp = script(vec![
        calls("m1", "files__add", json!({ "a": 2, "b": 3 })),
        reply("added"),
    ]);
    let mut models = scripted_models([
        ("test/reply", &replies),
        ("test/read", &read),
        ("test/again", &again),
        ("test/fail", &fail),
        ("test/delegate", &delegate),
        ("test/sub", &sub),
        ("test/guarded", &guarded),
        ("test/wait", &wait),
        ("test/mcp", &mcp),
        ("test/shell", &quick),
    ]);
    // A model of the catalog no key reaches.
    if let Ok(nowhere) = ProviderId::declare("acp-unreachable") {
        let spec = ModelSpec::new(nowhere, "model").with_tools(true);
        models.0.catalog_mut().insert(spec);
    }
    // `--tools`, which the router hands each session after `--`.
    let tools = std::env::args().filter_map(|arg| {
        let tools = arg.strip_prefix(&format!("--{}=", rig_harness::run::TOOLS))?;
        Some((rig_harness::run::TOOLS.to_owned(), Some(tools.to_owned())))
    });
    let invoked = Invocation {
        command: Some(command.to_owned()),
        flags: tools.collect(),
        ..Invocation::default()
    };
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: invoked,
        terminal: false,
    })
    .add_plugins((HeadlessPlugins, RigHarnessPlugins))
    .insert_resource(models)
    .add_plugins((
        AcpPlugin,
        ApprovalPlugin,
        rig_basics::models::ModelsPlugin,
        rig_basics::sessions::SessionsPlugin,
        rig_basics::sessions::agents::AgentsPlugin,
        rig_basics::coding_tools::ReadTool,
        rig_basics::coding_tools::ShellTool,
        rig_basics::BasicCommandsPlugin,
        rig_basics::compaction::CompactionPlugin,
        rig_basics::subagents::SubagentsPlugin,
    ))
    .add_command("ask", "Ask a question", ask)
    .add_command("early", "Start a call early", early)
    .add_command("hold", "Wait until stopped", hold)
    .add_observer(asked)
    .add_observer(carry_on)
    .add_observer(release)
    .add_systems(AgentPass, guard.in_set(Stage::Gate).before(ApprovalSystems));
    if !session {
        let program = std::env::current_exe().unwrap_or_default();
        // What the router adds after them is no test's filter.
        let mut args = libtest("agent");
        args.push("--".into());
        app.insert_resource(SessionCommand { program, args });
    }
    let exit = rig_harness::run(app);
    std::process::exit(match exit {
        AppExit::Success => 0,
        AppExit::Error(code) => i32::from(code.get()),
    });
}

/// A question that is no tool's approval: `/ask` asks which of two.
#[derive(Component)]
struct Asked;

fn ask(In(args): In<CommandArgs>, mut commands: Commands) {
    let question = Ask::new("Which one?").choice("a", "A").choice("b", "B");
    commands.spawn((question, Asked, ForAgent(args.agent)));
}

/// Carries on a turn the session restored, as a plugin may.
fn carry_on(restored: On<Add<Restored>>, turns: Query<(), With<TurnOf>>, mut commands: Commands) {
    if turns.contains(restored.entity) {
        commands.entity(restored.entity).try_insert(CarryOn);
    }
}

/// `/early <id>`: a call with the id `id` a plugin starts before the reply
/// that asks for it ended, as one that starts reads early does, withdrawn
/// at once.
fn early(In(args): In<CommandArgs>, mut commands: Commands) {
    let Ok(read) = rig_harness::rig_core::message::ToolName::new("read") else {
        return;
    };
    let function =
        rig_harness::rig_core::message::ToolFunction::new(read, json!({ "path": "fixture.txt" }));
    let call = rig_harness::rig_core::message::ToolCall::from_wire(args.args.trim(), function);
    let mut early = commands.spawn((
        ToolRequest { call },
        Effect::with_parent_id(None),
        ForAgent(args.agent),
    ));
    early.try_despawn();
}

/// Work that is no turn, as a sign-in waiting for its browser: `/hold`
/// waits until its agent is stopped.
#[derive(Component)]
struct Held(Entity);

fn hold(
    In(args): In<CommandArgs>,
    wake: Res<Wake>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let waiting = Running::spawn(pool, &wake, std::future::pending::<()>());
    commands.spawn((Held(args.agent), waiting));
    // Tells the test the work is there.
    std::fs::write(root.0.join("held"), "").ok();
}

/// Ends `/hold`'s work as its agent is stopped.
fn release(stop: On<StopTurn>, held: Query<(Entity, &Held)>, mut commands: Commands) {
    for (work, Held(agent)) in &held {
        if *agent == stop.entity {
            commands.entity(work).despawn();
        }
    }
}

/// Says how `/ask`'s question ended.
fn asked(
    settled: On<Add<Settled>>,
    asks: Query<(&Outcome, &ForAgent), With<Asked>>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok((outcome, &ForAgent(agent))) = asks.get(settled.entity) {
        let text = match &outcome.0 {
            Ok(_) => format!(
                "answered: {:?}",
                Answer::of(outcome).map(|answer| answer.choices)
            ),
            Err(why) => format!("not answered: {why}"),
        };
        notices.write(Notice::info(agent, text));
    }
}

/// A call that reads `guarded.txt` needs the user's approval.
fn guard(calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls)>, mut commands: Commands) {
    for (call, request) in &calls {
        let args = request.call.function.arguments_value();
        if args.get("path") == Some(&json!("guarded.txt")) {
            let reason = NeedsApproval::add("test", "It reads the guarded file.");
            commands.entity(call).queue(reason);
        }
    }
}

/// This test binary as an MCP server with the tool `add(a, b)`, answering
/// JSON-RPC lines on stdin.
#[test]
#[ignore = "the MCP server the other tests name"]
fn mcp_stub() {
    use std::io::{BufRead, Write};

    if std::env::var_os(STUB).is_none() {
        return;
    }
    let mut out = std::io::stdout();
    // libtest left its line about this test open.
    if writeln!(out).is_err() {
        return;
    }
    for line in std::io::stdin().lock().lines() {
        let Ok(request) = line.map(|line| serde_json::from_str::<Value>(&line)) else {
            return;
        };
        let Ok(request) = request else { continue };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let number = |key: &str| {
            let at = format!("/params/arguments/{key}");
            request
                .pointer(&at)
                .and_then(Value::as_f64)
                .unwrap_or_default()
        };
        let result = match request.get("method").and_then(Value::as_str) {
            Some("initialize") => {
                let slow = std::env::var(SLOW).ok().and_then(|ms| ms.parse().ok());
                std::thread::sleep(Duration::from_millis(slow.unwrap_or_default()));
                json!({
                "protocolVersion": request.pointer("/params/protocolVersion"),
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "stub", "version": "0" },
                })
            }
            Some("tools/list") => json!({ "tools": [{
                "name": "add",
                "inputSchema": {
                    "type": "object",
                    "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                },
            }]}),
            Some("tools/call") => {
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
            break;
        }
    }
    if std::env::var_os(STAYS).is_some() {
        std::thread::sleep(Duration::from_secs(120));
    }
}

/// The stub as a client declares it, with `env` added to its environment.
fn stub_with(name: &str, mut env: Vec<acp::EnvVariable>) -> acp::McpServer {
    let program = std::env::current_exe().unwrap_or_default();
    let args = libtest("mcp_stub")
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned());
    env.push(acp::EnvVariable::new(STUB, "1"));
    let stdio = acp::McpServerStdio::new(name, program)
        .args(args.collect())
        .env(env);
    acp::McpServer::Stdio(stdio)
}

/// A scratch place for a test: `home` is its `RIG_HOME`, `project` the
/// client's directory, with a `fixture.txt` and a `guarded.txt`.
struct Place {
    root: PathBuf,
    home: PathBuf,
    project: PathBuf,
    /// The router's `--tools`, if any.
    tools: Option<String>,
}

impl Place {
    fn new(name: &str) -> Self {
        let exe = std::env::current_exe().unwrap_or_default();
        let root = exe.with_file_name(format!("rig-acp-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let (home, project) = (root.join("home"), root.join("project"));
        for dir in [&home, &project] {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(project.join("fixture.txt"), "fixture text\n").ok();
        std::fs::write(project.join("guarded.txt"), "guarded text\n").ok();
        let project = std::fs::canonicalize(&project).unwrap_or(project);
        Self {
            root,
            home,
            project,
            tools: None,
        }
    }

    /// The same place, its router started with `--tools tools`.
    fn with_tools(mut self, tools: &str) -> Self {
        self.tools = Some(tools.to_owned());
        self
    }
}

impl Drop for Place {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).ok();
    }
}

/// How the client answers the agent's questions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answering {
    Allow,
    Deny,
    /// Keeps the question, for the test to answer ([`Seen::later`]).
    Later,
}

/// What the client saw and how it answers.
#[derive(Default)]
struct Seen {
    /// Every update, and every response to a prompt, in order.
    events: Vec<Event>,
    /// The call each permission request was about.
    asked: Vec<acp::ToolCallUpdate>,
    /// The questions kept to answer later.
    later: Vec<agent_client_protocol::Responder<acp::RequestPermissionResponse>>,
    answering: Option<Answering>,
}

/// What reached the client.
#[derive(Clone, Debug)]
enum Event {
    Update(acp::SessionId, acp::SessionUpdate),
    /// The answer to `session/new`, where it came among the updates.
    Opened(acp::SessionId),
    /// The answer to a prompt of the session, where it came among them.
    Answered(acp::SessionId),
}

/// A client connected to `rig acp`.
struct Driven {
    cx: ConnectionTo<Agent>,
    seen: Arc<Mutex<Seen>>,
    project: PathBuf,
    /// The router's process id.
    router: Option<u32>,
}

impl Driven {
    fn seen(&self) -> std::sync::MutexGuard<'_, Seen> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn answering(&self, answering: Answering) {
        self.seen().answering = Some(answering);
    }

    /// The updates of `session` so far.
    fn updates(&self, session: &acp::SessionId) -> Vec<acp::SessionUpdate> {
        let events = self.seen().events.clone();
        let updates = events.into_iter().filter_map(|event| match event {
            Event::Update(id, update) if &id == session => Some(update),
            _ => None,
        });
        updates.collect()
    }

    /// The text of `session`'s agent chunks so far.
    fn said(&self, session: &acp::SessionId) -> String {
        let chunks = self
            .updates(session)
            .into_iter()
            .filter_map(|update| match update {
                acp::SessionUpdate::AgentMessageChunk(chunk) => text(&chunk.content),
                _ => None,
            });
        chunks.collect()
    }

    /// Waits until `done` holds of what was seen, at most 30 seconds.
    async fn until(&self, done: impl Fn(&Seen) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if done(&self.seen()) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        false
    }

    /// A new session in the project, with `servers`.
    async fn new_session(&self, servers: Vec<acp::McpServer>) -> Option<acp::NewSessionResponse> {
        let request = acp::NewSessionRequest::new(&self.project).mcp_servers(servers);
        let (answered, answer) = tokio::sync::oneshot::channel();
        let seen = self.seen.clone();
        let ordered = self
            .cx
            .prepare_request(request)
            .on_receiving_result(async move |result| {
                if let Ok(opened) = &result {
                    let mut seen = seen
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    seen.events.push(Event::Opened(opened.session_id.clone()));
                }
                answered.send(result).ok();
                Ok(())
            });
        ordered.ok()?;
        answer.await.ok()?.ok()
    }

    /// Picks the model `model` for `session`.
    async fn model(&self, session: &acp::SessionId, model: &str) -> bool {
        let value = acp::SessionConfigOptionValue::value_id(model.to_owned());
        let request = acp::SetSessionConfigOptionRequest::new(session.clone(), "model", value);
        self.cx.send_request(request).block_task().await.is_ok()
    }

    /// Sends `text` to `session`, answered once its work settled.
    async fn prompt(
        &self,
        session: &acp::SessionId,
        text: &str,
    ) -> Result<acp::StopReason, agent_client_protocol::Error> {
        self.send(session, text).await
    }

    /// Sends `text` to `session`; the answer, once it comes, noted where it
    /// came among the updates.
    fn send(
        &self,
        session: &acp::SessionId,
        text: &str,
    ) -> impl Future<Output = Result<acp::StopReason, agent_client_protocol::Error>> + use<> {
        let request = acp::PromptRequest::new(session.clone(), vec![text.into()]);
        let (answered, answer) = tokio::sync::oneshot::channel();
        let (seen, id) = (self.seen.clone(), session.clone());
        let ordered = self
            .cx
            .prepare_request(request)
            .on_receiving_result(async move |result| {
                let mut seen = seen
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                seen.events.push(Event::Answered(id));
                answered.send(result).ok();
                Ok(())
            });
        async move {
            ordered?;
            let answer = answer
                .await
                .map_err(|_| agent_client_protocol::Error::internal_error())?;
            answer.map(|answer| answer.stop_reason)
        }
    }
}

/// The tool updates of `session`: each call's id and status, in order.
fn tool_steps(driven: &Driven, session: &acp::SessionId) -> Vec<(String, acp::ToolCallStatus)> {
    let steps = driven
        .updates(session)
        .into_iter()
        .filter_map(|update| match update {
            acp::SessionUpdate::ToolCall(call) => {
                Some((call.tool_call_id.to_string(), call.status))
            }
            acp::SessionUpdate::ToolCallUpdate(update) => {
                let status = update.fields.status?;
                Some((update.tool_call_id.to_string(), status))
            }
            _ => None,
        });
    steps.collect()
}

/// The text of a content block.
fn text(content: &acp::ContentBlock) -> Option<String> {
    match content {
        acp::ContentBlock::Text(text) => Some(text.text.clone()),
        _ => None,
    }
}

/// `rig acp` as a process, as the client sees it.
struct Router {
    process: tokio::process::Child,
}

impl Router {
    /// Starts `rig acp` for `place`.
    fn start(
        place: &Place,
    ) -> Option<(
        Self,
        tokio::process::ChildStdin,
        tokio::process::ChildStdout,
    )> {
        let tools = place.tools.iter().flat_map(|tools| {
            let flag = format!("--{}={tools}", rig_harness::run::TOOLS);
            [OsString::from("--"), OsString::from(flag)]
        });
        let mut process = tokio::process::Command::new(std::env::current_exe().ok()?)
            .args(libtest("agent"))
            .args(tools)
            .env(AGENT, "1")
            .env("RIG_HOME", &place.home)
            .env_remove("RIG_RUN")
            .env_remove("RIG_SESSION")
            .current_dir(&place.project)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(match std::env::var_os("RIG_ACP_TEST_STDERR") {
                Some(_) => Stdio::inherit(),
                None => Stdio::null(),
            })
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let (input, output) = (process.stdin.take()?, process.stdout.take()?);
        Some((Self { process }, input, output))
    }

    fn id(&self) -> Option<u32> {
        self.process.id()
    }
}

/// Runs `test` with a client of a new `rig acp` for `place`, the client
/// declaring `capabilities`; the router, once `test` is done and the client
/// closed its side.
fn drive(
    place: &Place,
    capabilities: acp::ClientCapabilities,
    test: impl AsyncFnOnce(&Driven) + Send,
) -> Option<Router> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .ok()?;
    runtime.block_on(async {
        let (router, input, output) = Router::start(place)?;
        let pid = router.id();
        let seen = Arc::new(Mutex::new(Seen::default()));
        let (updates, permissions) = (seen.clone(), seen.clone());
        let project = place.project.clone();
        let driven = Client
            .builder()
            .on_receive_notification(
                async move |update: acp::SessionNotification, _| {
                    let mut seen = updates
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    seen.events
                        .push(Event::Update(update.session_id, update.update));
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                async move |request: acp::RequestPermissionRequest, responder, _| {
                    let mut seen = permissions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    seen.asked.push(request.tool_call.clone());
                    let answering = seen.answering;
                    if answering == Some(Answering::Later) {
                        seen.later.push(responder);
                        return Ok(());
                    }
                    drop(seen);
                    let chosen = |kind| {
                        let option = request.options.iter().find(|option| option.kind == kind);
                        option.map(|option| option.option_id.clone())
                    };
                    let outcome = match answering {
                        Some(Answering::Allow) => chosen(acp::PermissionOptionKind::AllowOnce),
                        Some(Answering::Deny) => chosen(acp::PermissionOptionKind::RejectOnce),
                        _ => return Ok(()),
                    };
                    let outcome = match outcome {
                        Some(id) => acp::RequestPermissionOutcome::Selected(
                            acp::SelectedPermissionOutcome::new(id),
                        ),
                        None => acp::RequestPermissionOutcome::Cancelled,
                    };
                    responder.respond(acp::RequestPermissionResponse::new(outcome))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_: acp::CreateElicitationRequest, responder, _| {
                    let mut content = std::collections::BTreeMap::new();
                    content.insert(
                        "choice".to_owned(),
                        acp::ElicitationContentValue::from("b".to_owned()),
                    );
                    let accept = acp::ElicitationAcceptAction::new().content(content);
                    responder.respond(acp::CreateElicitationResponse::new(
                        acp::ElicitationAction::Accept(accept),
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(lines(input, output), async move |cx| {
                let init = acp::InitializeRequest::new(ProtocolVersion::V1)
                    .client_capabilities(capabilities);
                cx.send_request(init).block_task().await?;
                let driven = Driven {
                    cx,
                    seen,
                    project,
                    router: pid,
                };
                test(&driven).await;
                Ok(())
            })
            .await;
        driven.ok()?;
        Some(router)
    })
}

/// Whether the process `pid` still runs (a zombie counts as gone).
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        !stat
            .split(") ")
            .nth(1)
            .is_some_and(|rest| rest.starts_with('Z'))
    })
}

/// Waits until `gone` holds, at most 20 seconds.
#[cfg(target_os = "linux")]
fn eventually(gone: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if gone() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// The capabilities of a client that shows forms and notices.
fn full_client() -> acp::ClientCapabilities {
    acp::ClientCapabilities::new()
        .elicitation(
            acp::ElicitationCapabilities::new().form(acp::ElicitationFormCapabilities::new()),
        )
        .session(acp::ClientSessionCapabilities::new().notices(acp::NoticeCapabilities::new()))
}

/// The project directory of `place`, for paths in assertions.
fn project(place: &Place) -> &Path {
    &place.project
}

/// The live processes of `place`: the sessions, and the tools they started.
#[cfg(target_os = "linux")]
fn processes_of(place: &Place) -> Vec<u32> {
    let root = place.root.to_string_lossy().into_owned();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| {
            std::fs::read(format!("/proc/{pid}/environ"))
                .is_ok_and(|environ| String::from_utf8_lossy(&environ).contains(&root))
        })
        .filter(|pid| alive(*pid))
        .collect()
}

/// Whether a tool's `sleep` of `place` runs.
#[cfg(target_os = "linux")]
fn sleeping(place: &Place) -> bool {
    processes_of(place).into_iter().any(|pid| {
        std::fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|line| String::from_utf8_lossy(&line).contains("sleep"))
    })
}
