//! One ACP session in a rig process of its own (`rig acp-session`, which
//! `rig acp` starts in the session's directory): a typed ACP v1 agent on
//! stdin and stdout for the one session `RIG_SESSION` names, driving its
//! main agent. A front nobody sits at: it answers questions through the
//! client, and the client's prompts go to the main agent.
//!
//! The app does not start before the client's request that opens the
//! session arrived, as the client's MCP servers ([`AddMcpServer`]) must be
//! named before `Startup`, when the files' servers start. The request is
//! answered once the session is restored and each MCP server's first attempt
//! ended; a loaded session's history goes first. Updates made before that
//! answer wait for it, as a client drops updates of a session it was not
//! told about yet.

mod options;
mod output;
mod questions;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};

use agent_client_protocol::schema::v1 as acp;
use agent_client_protocol::{Agent, Client, ConnectionTo, Error};
use async_channel::{Receiver, Sender};
use bevy_ecs::system::SystemParam;
use rig_basics::print::{Starting, Work};
use rig_harness::harness_protocol::SessionId;
use rig_harness::host::pipes::Writer;
use rig_harness::host::signals;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::StopTurnsOnExit;
use rig_harness::rig_ecs::session_log::{SessionRestored, SessionStore};
use rig_harness::rig_memory::summarized_in_log;
use rig_harness::services::WorkDir;
use rig_mcp_client::{AddMcpServer, ServerSpec};

use rig_harness::rig_core::completion::message::{Message as Said, StopReason as Stop};

use crate::router::GRACE;
use crate::{Reply, translate};

#[cfg(test)]
mod tests;

/// The command that runs a session, which `rig acp` starts.
pub(crate) const FRONT: &str = "acp-session";

/// How long the process waits for the request that opens its session.
const OPENING: Duration = Duration::from_secs(60);

/// How often that wait looks for the request, and for a signal.
const WAIT: Duration = Duration::from_millis(5);

/// The most requests that wait for the app; the connection waits for room.
const REQUESTS: usize = 64;

/// What the ACP connection hands the app.
pub(crate) enum Request {
    /// The client's capabilities, and the connection to it.
    Init(Box<acp::ClientCapabilities>, ConnectionTo<Client>),
    Open(Opening),
    Prompt(acp::PromptRequest, Reply<acp::PromptResponse>),
    Cancel,
    Config(
        acp::SetSessionConfigOptionRequest,
        Reply<acp::SetSessionConfigOptionResponse>,
    ),
    Close(Reply<acp::CloseSessionResponse>),
    /// The client answered the question on `Entity`, or closed it.
    Answered(Entity, Option<Answer>),
    /// The connection ended.
    Ended,
}

/// The request that opens the session.
pub(crate) enum Opening {
    New(acp::NewSessionRequest, Reply<acp::NewSessionResponse>),
    Load(acp::LoadSessionRequest, Reply<acp::LoadSessionResponse>),
    Resume(acp::ResumeSessionRequest, Reply<acp::ResumeSessionResponse>),
}

impl Opening {
    fn servers(&self) -> &[acp::McpServer] {
        match self {
            Self::New(request, _) => &request.mcp_servers,
            Self::Load(request, _) => &request.mcp_servers,
            Self::Resume(request, _) => &request.mcp_servers,
        }
    }

    /// Answers with the session's options.
    fn answer(self, id: &acp::SessionId, config: Vec<acp::SessionConfigOption>) {
        match self {
            Self::New(_, reply) => {
                reply.send(Ok(
                    acp::NewSessionResponse::new(id.clone()).config_options(config)
                ));
            }
            Self::Load(_, reply) => {
                reply.send(Ok(acp::LoadSessionResponse::new().config_options(config)));
            }
            Self::Resume(_, reply) => {
                reply.send(Ok(acp::ResumeSessionResponse::new().config_options(config)));
            }
        }
    }
}

/// The session as the app serves it.
#[derive(Resource)]
pub(crate) struct Session {
    id: acp::SessionId,
    cx: ConnectionTo<Client>,
    client: acp::ClientCapabilities,
    /// The request that opened it, until answered.
    opening: Option<Opening>,
    /// Whether the main agent was given a model once, before the answer.
    chose: bool,
    /// The options the client was last told of.
    advertised: Vec<acp::SessionConfigOption>,
    /// Updates made before the answer, sent after it.
    held: Vec<acp::SessionUpdate>,
    /// Notes for a client that shows no notices, which open the next
    /// prompt's reply: some clients show agent text only during a prompt.
    notes: Vec<String>,
    prompt: Option<Pending>,
    /// Requests that came before the session opened, carried out first.
    early: VecDeque<Request>,
    /// The client's closes, answered once the prompt they cancelled is.
    closing: Vec<Reply<acp::CloseSessionResponse>>,
    /// The tasks asking the client a question, by the question's entity.
    asking: HashMap<Entity, tokio::task::AbortHandle>,
    /// The ACP id of each running call the client was shown, or sees
    /// settle, by the call's entity.
    calls: HashMap<Entity, String>,
    /// A loaded session's history as it was restored, until it opens.
    replay: Option<Vec<acp::SessionUpdate>>,
    requests: Receiver<Request>,
    sender: Sender<Request>,
    runtime: tokio::runtime::Handle,
    wake: Wake,
    /// The writer of stdout: while it is behind, the agent's work waits.
    writer: Writer,
    /// Dropped once every request is answered, which lets the connection
    /// end, after what it was handed is sent.
    done: Option<tokio::sync::oneshot::Sender<()>>,
    /// The connection, done once it ended.
    served: Option<tokio::task::JoinHandle<()>>,
}

/// The prompt the app works on.
struct Pending {
    reply: Reply<acp::PromptResponse>,
    /// How the main agent's last turn ended.
    ended: Option<TurnOutcome>,
    cancelled: bool,
}

/// On the message a prompt sent.
#[derive(Component)]
struct Prompted;

impl Session {
    /// Sends `update` now, or once the session is open.
    fn send(&mut self, update: acp::SessionUpdate) {
        if self.opening.is_some() {
            self.held.push(update);
        } else {
            let notification = acp::SessionNotification::new(self.id.clone(), update);
            self.cx.send_notification(notification).ok();
        }
    }

    /// Whether the client shows notices outside a reply.
    fn shows_notices(&self) -> bool {
        let session = self.client.session.as_ref();
        session.is_some_and(|session| session.notices.is_some())
    }

    /// Says `text` to the user: in the reply while a prompt runs, else as a
    /// notice, else in the next reply.
    fn note(&mut self, text: String, level: NoticeLevel) {
        if self.prompt.is_some() {
            let chunk = acp::ContentChunk::new(acp::ContentBlock::from(in_reply(&text)));
            self.send(acp::SessionUpdate::AgentMessageChunk(chunk));
        } else if self.shows_notices() {
            let severity = match level {
                NoticeLevel::Info => acp::NoticeSeverity::Info,
                NoticeLevel::Warn => acp::NoticeSeverity::Warning,
                NoticeLevel::Error => acp::NoticeSeverity::Error,
            };
            self.send(acp::SessionUpdate::Notice(acp::Notice::new(severity, text)));
        } else {
            self.notes.push(text);
        }
    }

    /// Hands `request` from a task to the app, once there is room, and wakes
    /// it. A request the app no longer takes is dropped, and its reply
    /// answers that the session ended.
    async fn post(sender: &Sender<Request>, wake: &Wake, request: Request) {
        sender.send(request).await.ok();
        wake.wake();
    }
}

/// A note as a paragraph of the reply, which clients show as Markdown: one
/// of several lines, such as `/help`'s, as preformatted text.
fn in_reply(text: &str) -> String {
    match text.contains('\n') {
        true => format!("\n\n```text\n{text}\n```\n\n"),
        false => format!("\n\n{text}\n\n"),
    }
}

/// Makes this app the session's process: serves ACP on stdio, waits for the
/// request that opens the session, and names its client's MCP servers.
pub(crate) fn install(app: &mut App) {
    let world = app.world();
    let (Some(runtime), Some(wake)) = (
        world.get_resource::<TokioRuntime>(),
        world.get_resource::<Wake>(),
    ) else {
        app.refuse_run("rig acp-session needs rig's tokio runtime".to_owned());
        return;
    };
    let (runtime, wake) = (runtime.0.clone(), wake.clone());
    let id = match SessionId::from_env() {
        Ok(Some(id)) => acp::SessionId::new(id.as_str().to_owned()),
        _ => {
            app.refuse_run(
                "rig acp-session runs the session RIG_SESSION names; rig acp starts it".to_owned(),
            );
            return;
        }
    };
    let (sender, requests) = async_channel::bounded(REQUESTS);
    let interrupted = signals::interruption(app.world());
    let writer = Writer::spawn(
        std::io::stdout(),
        std::io::stderr(),
        wake.clone(),
        interrupted,
    );
    let (done, finished) = tokio::sync::oneshot::channel();
    let served = runtime.spawn(serve(
        sender.clone(),
        wake.clone(),
        writer.clone(),
        finished,
    ));
    // A signal or the router gone ends the wait; the first frame then exits.
    let signalled = signals::interruption(app.world());
    let deadline = Instant::now() + OPENING;
    let (mut init, mut opening, mut early) = (None, None, VecDeque::new());
    while (init.is_none() || opening.is_none()) && Instant::now() < deadline && !signalled() {
        // `finish` runs within rig's runtime, which it cannot block on.
        match requests.try_recv() {
            Ok(Request::Init(client, cx)) => init = Some((client, cx)),
            Ok(Request::Open(open)) => opening = Some(open),
            Ok(Request::Ended) | Err(async_channel::TryRecvError::Closed) => break,
            // Such as a close sent before the session opened: kept for when
            // it has.
            Ok(request) => early.push_back(request),
            Err(async_channel::TryRecvError::Empty) => std::thread::sleep(WAIT),
        }
    }
    let (Some((client, cx)), Some(opening)) = (init, opening) else {
        app.add_systems(Startup, |mut commands: Commands| {
            commands.trigger(Exit(ExitReason::Quit));
        });
        return;
    };
    let declared: Vec<(String, AddMcpServer)> =
        opening.servers().iter().filter_map(mcp_server).collect();
    let renamed: Vec<String> = declared
        .iter()
        .filter(|(asked, server)| *asked != server.name)
        .map(|(asked, server)| {
            let name = &server.name;
            format!("MCP server \"{asked}\" is `{name}` in rig: its tools are `{name}__<tool>`.")
        })
        .collect();
    let servers: Vec<AddMcpServer> = declared.into_iter().map(|(_, server)| server).collect();
    app.insert_resource(Session {
        id,
        cx,
        client: *client,
        opening: Some(opening),
        chose: false,
        advertised: Vec::new(),
        held: Vec::new(),
        notes: Vec::new(),
        prompt: None,
        early,
        closing: Vec::new(),
        asking: HashMap::new(),
        calls: HashMap::new(),
        replay: None,
        requests,
        sender,
        runtime,
        wake,
        writer,
        done: Some(done),
        served: Some(served),
    })
    // The agent's work waits while the router is behind reading it.
    .configure_sets(Update, AgentLoop.run_if(has_room))
    .add_systems(PreStartup, move |mut commands: Commands| {
        for server in &servers {
            commands.trigger(server.clone());
        }
    })
    .add_systems(
        Update,
        (requests_in, open, stop_cancelled)
            .chain()
            .after(PollTasks)
            .before(AgentLoop),
    )
    .add_systems(
        Update,
        (output::notices, options::changed, settle)
            .chain()
            .after(AgentLoop)
            .after(CurrentAgentSystems),
    )
    .init_resource::<output::Read>()
    .add_systems(PostUpdate, output::stream)
    .add_observer(output::call_asked)
    .add_observer(output::started)
    .add_observer(output::settled)
    .add_observer(questions::picker)
    .add_observer(ended)
    .add_observer(questions::forgotten)
    .add_systems(
        Last,
        answer_open
            .after(StopTurnsOnExit)
            .in_set(OnAppExitSystems)
            .run_if(on_message::<AppExit>),
    );
    let mut session = app.world_mut().resource_mut::<Session>();
    for note in renamed {
        session.note(note, NoticeLevel::Info);
    }
    app.world_mut().spawn(Answerer::default());
}

/// One of the client's MCP servers as rig-mcp-client takes it, with the
/// name the client gave it; `None` for one of a kind ACP may add later.
fn mcp_server(server: &acp::McpServer) -> Option<(String, AddMcpServer)> {
    let (name, spec) = match server {
        acp::McpServer::Stdio(stdio) => {
            // An environment's names are alike in any case on Windows.
            let env = stdio.env.iter().map(|var| (&var.name, &var.value));
            let spec = ServerSpec::Stdio {
                command: stdio.command.to_string_lossy().into_owned(),
                args: stdio.args.clone(),
                env: last_of(env, cfg!(windows)),
                cwd: None,
            };
            (stdio.name.clone(), Ok(spec))
        }
        acp::McpServer::Http(http) => {
            let headers = http
                .headers
                .iter()
                .map(|header| (&header.name, &header.value));
            let spec = ServerSpec::Http {
                url: http.url.clone(),
                headers: last_of(headers, true),
            };
            (http.name.clone(), Ok(spec))
        }
        acp::McpServer::Sse(sse) => {
            let why = "rig speaks MCP over stdio and streamable HTTP, not SSE".to_owned();
            (sse.name.clone(), Err(why))
        }
        _ => return None,
    };
    let server = AddMcpServer {
        name: server_name(&name),
        spec,
    };
    Some((name, server))
}

/// A name the client gave a server, for people ("GitHub MCP"), as a name
/// rig gives its tools: letters, digits, `_` and `-`, the rest `-`, no
/// `__`, no `_` at its end, starting with a letter or `_`.
fn server_name(given: &str) -> String {
    let allowed = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let mut name: String = given
        .chars()
        .map(|c| if allowed(c) { c } else { '-' })
        .collect();
    while name.contains("__") {
        name = name.replace("__", "_");
    }
    let name = name.trim_end_matches('_');
    match name.chars().next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => name.to_owned(),
        Some(_) => format!("_{name}"),
        None => "server".to_owned(),
    }
}

/// `pairs` as a map, a later pair of a name replacing an earlier one, names
/// alike in any case when `any_case`.
fn last_of<'a>(
    pairs: impl Iterator<Item = (&'a String, &'a String)>,
    any_case: bool,
) -> BTreeMap<String, String> {
    let mut kept: Vec<(String, String)> = Vec::new();
    for (name, value) in pairs {
        kept.retain(|(had, _)| had != name && !(any_case && had.eq_ignore_ascii_case(name)));
        kept.push((name.clone(), value.clone()));
    }
    kept.into_iter().collect()
}

/// Serves ACP on stdio, handing each request to the app, until the app
/// answered every request and dropped `done`: the end of stdin is a request
/// too ([`Request::Ended`]), so the answers the exit owes still go out.
async fn serve(
    sender: Sender<Request>,
    wake: Wake,
    writer: Writer,
    done: tokio::sync::oneshot::Receiver<()>,
) {
    let post = async |request| Session::post(&sender, &wake, request).await;
    let result = Agent
        .builder()
        .name("rig-acp-session")
        .on_receive_request(
            async |request: acp::InitializeRequest, responder, cx: ConnectionTo<Client>| {
                let client = Box::new(request.client_capabilities.clone());
                post(Request::Init(client, cx)).await;
                responder.respond(crate::initialized())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async |request: acp::NewSessionRequest, responder, _| {
                post(Request::Open(Opening::New(request, Reply::new(responder)))).await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async |request: acp::LoadSessionRequest, responder, _| {
                post(Request::Open(Opening::Load(request, Reply::new(responder)))).await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async |request: acp::ResumeSessionRequest, responder, _| {
                post(Request::Open(Opening::Resume(
                    request,
                    Reply::new(responder),
                )))
                .await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async |request: acp::PromptRequest, responder, _| {
                post(Request::Prompt(request, Reply::new(responder))).await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async |_: acp::CancelNotification, _| {
                post(Request::Cancel).await;
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async |request: acp::SetSessionConfigOptionRequest, responder, _| {
                post(Request::Config(request, Reply::new(responder))).await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async |_: acp::CloseSessionRequest, responder, _| {
                post(Request::Close(Reply::new(responder))).await;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            crate::stdio::stdio(writer, "rig acp-session"),
            async |cx: ConnectionTo<Client>| {
                let read = async {
                    cx.incoming_closed().await;
                    post(Request::Ended).await;
                    std::future::pending::<()>().await;
                };
                tokio::select! {
                    () = read => {}
                    _ = done => {}
                }
                Ok(())
            },
        )
        .await;
    if let Err(error) = result {
        eprintln!("rig acp-session: {error}");
    }
    post(Request::Ended).await;
}

/// Whether the router keeps up reading what the session writes.
fn has_room(session: Res<Session>) -> bool {
    session.writer.has_room()
}

/// The main agent, and what a request reads and changes of the app.
#[derive(SystemParam)]
struct Requested<'w, 's> {
    session: ResMut<'w, Session>,
    primary: MainAgentQuery<'w, 's>,
    agents: Query<'w, 's, (Has<Connection>, Option<&'static ModelRefused>)>,
    commands: Commands<'w, 's>,
}

/// Carries out what the client asked since the last frame. While a change
/// of an option waits for its answer, the requests after it wait too, so
/// they see it.
fn requests_in(
    mut requested: Requested,
    mut options: options::Options,
    changing: Option<Res<options::Changing>>,
) {
    let mut changing = changing.is_some();
    while !changing {
        let early = requested.session.early.pop_front();
        let Some(request) = early.or_else(|| requested.session.requests.try_recv().ok()) else {
            break;
        };
        match request {
            Request::Prompt(prompt, reply) => requested.prompt(prompt, reply),
            Request::Cancel => requested.cancel(),
            Request::Config(change, reply) => match options.set(&change) {
                Ok(asked) => {
                    let asked = options::Changing { reply, asked };
                    requested.commands.insert_resource(asked);
                    changing = true;
                }
                Err(why) => reply.send(Err(why)),
            },
            Request::Close(reply) => {
                requested.cancel();
                // The prompt it cancels is answered first ([`settle`]).
                match requested.session.prompt.is_some() {
                    true => requested.session.closing.push(reply),
                    false => {
                        reply.send(Ok(acp::CloseSessionResponse::new()));
                        requested.commands.trigger(Exit(ExitReason::Quit));
                    }
                }
            }
            Request::Answered(ask, answer) => {
                questions::answer_with(&mut requested.commands, ask, answer);
            }
            Request::Ended => requested.commands.trigger(Exit(ExitReason::Quit)),
            Request::Init(..) | Request::Open(_) => {}
        }
    }
}

impl Requested<'_, '_> {
    /// Sends the prompt to the main agent, or says why not.
    fn prompt(&mut self, prompt: acp::PromptRequest, reply: Reply<acp::PromptResponse>) {
        if self.session.opening.is_some() || self.session.prompt.is_some() {
            let why = "the session is busy: one prompt runs at a time";
            return reply.send(Err(crate::error(Error::invalid_request(), why)));
        }
        let parts = match translate::prompt(&prompt.prompt) {
            Ok(parts) => parts,
            Err(why) => {
                return reply.send(Err(crate::error(Error::invalid_params(), why.to_string())));
            }
        };
        let Some(agent) = main_agent(&self.primary) else {
            return reply.send(Err(crate::error(
                Error::internal_error(),
                "the session has no agent",
            )));
        };
        let mut message = Incoming::new("", DeliveryMode::Steer);
        message.parts = parts;
        // The command as rig reads it: the line's first word after `/`.
        let command = message.command().map(|line| line.split_whitespace().next());
        if command == Some(Some("quit")) {
            let why = "an editor's session ends when the editor closes it, not with /quit";
            return reply.send(Err(crate::error(Error::invalid_request(), why)));
        }
        let (connected, refused) = self.agents.get(agent).unwrap_or((false, None));
        if command.is_none() && !connected {
            let why = refused.map_or(
                "no model can be reached: sign in with `rig` in a terminal (`/login`), then open a \
                 new session here",
                |ModelRefused(why)| why.as_str(),
            );
            return reply.send(Err(crate::error(Error::auth_required(), why.to_owned())));
        }
        for note in std::mem::take(&mut self.session.notes) {
            let chunk = acp::ContentChunk::new(acp::ContentBlock::from(in_reply(&note)));
            self.session
                .send(acp::SessionUpdate::AgentMessageChunk(chunk));
        }
        self.commands.spawn((message, ForAgent(agent), Prompted));
        self.session.prompt = Some(Pending {
            reply,
            ended: None,
            cancelled: false,
        });
    }

    /// Marks the prompt cancelled, which stops its work ([`stop_cancelled`]),
    /// and stops the main agent at once, which also ends work that is no
    /// turn, such as a sign-in.
    fn cancel(&mut self) {
        let Some(prompt) = self.session.prompt.as_mut() else {
            return;
        };
        prompt.cancelled = true;
        if let Some(agent) = main_agent(&self.primary) {
            self.commands.trigger(StopTurn::new(agent));
        }
    }
}

/// What answering the opening request reads.
#[derive(SystemParam)]
struct Opened<'w, 's> {
    session: ResMut<'w, Session>,
    restored: Option<Res<'w, SessionRestored>>,
    starting: Query<'w, 's, (), With<Starting>>,
    primary: MainAgentQuery<'w, 's>,
    agents: Query<
        'w,
        's,
        (
            &'static AgentId,
            &'static Conversation,
            Option<&'static ModelChoice>,
            Option<&'static WorkDir>,
        ),
    >,
    store: Option<Res<'w, SessionStore>>,
    root: Res<'w, ProjectRoot>,
    slash: Query<'w, 's, (&'static Name, &'static SlashCommand)>,
    models: Res<'w, Models>,
    options: options::Options<'w, 's>,
    commands: Commands<'w, 's>,
}

/// Answers the opening request once the session is restored and no MCP
/// server's first attempt is still out: a loaded session's history first,
/// then the answer, the updates held meanwhile, the commands and options.
fn open(mut opened: Opened) {
    let restored = opened.restored.is_some() && opened.session.opening.is_some();
    let agent = main_agent(&opened.primary).filter(|_| restored);
    let Some(agent) = agent else {
        return;
    };
    let Ok((id, conversation, chosen, work)) = opened.agents.get(agent) else {
        return;
    };
    // The history as restored, before a turn the session carries on adds
    // to it: what that turn adds reaches the client as it happens.
    if opened.session.replay.is_none() {
        let replay = match &opened.session.opening {
            Some(Opening::Load(..)) => {
                let replaced = opened
                    .store
                    .as_ref()
                    .and_then(|store| summarized_in_log(&*store.0, &id.0).ok())
                    .map(|summarized| summarized.replaced)
                    .unwrap_or_default();
                let cwd = translate::cwd_of(work, &opened.root);
                translate::replay(&id.0, conversation, &replaced, &cwd)
            }
            _ => Vec::new(),
        };
        opened.session.replay = Some(replay);
    }
    if !opened.starting.is_empty() {
        return;
    }
    // A new session's agent gets the first model it can reach, as a
    // headless run's does; the answer waits a frame, so its options show it.
    if !opened.session.chose {
        opened.session.chose = true;
        let reachable = opened.models.0.reachable();
        if let (None, Some(model)) = (chosen, reachable.first()) {
            let model = model.reference();
            opened.commands.trigger(SetModel {
                entity: agent,
                model,
            });
            opened.session.wake.wake();
            return;
        }
    }
    let replay = opened.session.replay.take().unwrap_or_default();
    let config = opened.options.of(agent);
    let mut listed: Vec<_> = opened
        .slash
        .iter()
        .filter_map(|(name, command)| {
            let name = name.as_str().strip_prefix('/')?;
            let shown = !LEFT_OUT.contains(&name);
            shown.then(|| acp::AvailableCommand::new(name.to_owned(), command.help.clone()))
        })
        .collect();
    listed.sort_by(|a, b| a.name.cmp(&b.name));
    let session = &mut *opened.session;
    let Some(opening) = session.opening.take() else {
        return;
    };
    for update in replay {
        let notification = acp::SessionNotification::new(session.id.clone(), update);
        session.cx.send_notification(notification).ok();
    }
    let id = session.id.clone();
    session.advertised = config.clone();
    opening.answer(&id, config);
    for update in std::mem::take(&mut session.held) {
        session.send(update);
    }
    let commands = acp::AvailableCommandsUpdate::new(listed);
    session.send(acp::SessionUpdate::AvailableCommandsUpdate(commands));
}

/// The commands the client is not offered: they start another session,
/// reload rig or end it, which an ACP session does not do, or show another
/// agent, while ACP shows the main one.
const LEFT_OUT: [&str; 7] = [
    "new", "resume", "reload", "plugins", "quit", "fork", "agents",
];

/// Stops a cancelled prompt's work: its message, declined if no turn took
/// it yet, and while any work goes on, the main agent and those below it.
fn stop_cancelled(
    session: Res<Session>,
    primary: MainAgentQuery,
    work: Work,
    mut messages: Query<&mut Decision, (With<Prompted>, Without<Started>)>,
    mut commands: Commands,
) {
    let cancelled = session
        .prompt
        .as_ref()
        .is_some_and(|prompt| prompt.cancelled);
    let Some(agent) = main_agent(&primary).filter(|_| cancelled) else {
        return;
    };
    for mut decision in &mut messages {
        decision.decide(Decider::plugin("rig-acp"), Outcome::custom("cancelled"));
    }
    // Stopping the main agent stops the agents below it, idle or not, and
    // work that is no turn, such as a sign-in, however late it started.
    if !work.idle() {
        commands.trigger(StopTurn::new(agent));
    }
}

/// Notes how the main agent's turn ended, for the prompt it answers.
fn ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    primary: MainAgentQuery,
    mut session: ResMut<Session>,
) {
    let Ok((outcome, ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    if Some(*agent) == main_agent(&primary)
        && let Some(prompt) = session.prompt.as_mut()
    {
        prompt.ended = Some(outcome.clone());
    }
}

/// How a prompt ended, from how the main agent's last turn since it did:
/// cancelled by the client, or by a stop that gave no reason; a failure as
/// an error; else the end of the answer, `max_tokens` when the output limit
/// cut it. A turn a plugin stopped for a reason, as `/compact` does, is no
/// cancel.
fn stop_reason(cancelled: bool, ended: Option<&TurnOutcome>) -> Result<acp::StopReason, Error> {
    match (cancelled, ended) {
        (true, _) | (false, Some(TurnOutcome::Stopped(None))) => Ok(acp::StopReason::Cancelled),
        (false, Some(TurnOutcome::Failed(why))) => Err(crate::error(Error::internal_error(), why)),
        (false, Some(TurnOutcome::Answered(Said::Assistant(reply))))
            if reply.stop == Some(Stop::Length) =>
        {
            Ok(acp::StopReason::MaxTokens)
        }
        (false, _) => Ok(acp::StopReason::EndTurn),
    }
}

/// As the app exits, after the exit stopped the turns (each cut call's
/// update is out already): sends the text streamed, answers the prompt
/// `cancelled` unless its turn ended otherwise, every close, and every
/// other request owed or still queued with why it was not carried out;
/// then lets the connection end, waits (within [`GRACE`]) until what it
/// was handed is sent, and until the writer wrote it.
fn answer_open(mut out: output::Out, changing: Option<Res<options::Changing>>) {
    out.stream();
    let session = &mut *out.session;
    if let Some(prompt) = session.prompt.take() {
        let answer = match &prompt.ended {
            None | Some(TurnOutcome::Stopped(_)) => Ok(acp::StopReason::Cancelled),
            ended => stop_reason(prompt.cancelled, ended.as_ref()),
        };
        prompt.reply.send(answer.map(acp::PromptResponse::new));
    }
    for close in session.closing.drain(..) {
        close.send(Ok(acp::CloseSessionResponse::new()));
    }
    for (_, asking) in session.asking.drain() {
        asking.abort();
    }
    if let Some(changing) = changing {
        changing.reply.send(Err(crate::ended()));
    }
    session.requests.close();
    let queued = std::iter::from_fn(|| session.requests.try_recv().ok());
    let left: Vec<Request> = session.early.drain(..).chain(queued).collect();
    for request in left {
        match request {
            Request::Prompt(_, reply) => reply.send(Err(crate::ended())),
            Request::Config(_, reply) => reply.send(Err(crate::ended())),
            Request::Close(reply) => reply.send(Ok(acp::CloseSessionResponse::new())),
            // Each answers, dropped, that the session ended.
            _ => {}
        }
    }
    session.opening.take();
    session.done.take();
    if let Some(served) = session.served.take() {
        session
            .runtime
            .block_on(async { tokio::time::timeout(GRACE, served).await.ok() });
    }
    session.writer.finish();
}

/// Answers the prompt once the work it started settled.
fn settle(work: Work, mut out: output::Out, mut commands: Commands) {
    if !work.idle() || out.session.prompt.is_none() {
        return;
    }
    // The turn's last text goes before its answer.
    out.stream();
    let Some(prompt) = out.session.prompt.take() else {
        return;
    };
    let answer = stop_reason(prompt.cancelled, prompt.ended.as_ref());
    prompt.reply.send(answer.map(acp::PromptResponse::new));
    let closing = std::mem::take(&mut out.session.closing);
    for close in &closing {
        close.send(Ok(acp::CloseSessionResponse::new()));
    }
    if !closing.is_empty() {
        commands.trigger(Exit(ExitReason::Quit));
    }
}
