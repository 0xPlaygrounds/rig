//! rig-agents: helper agents for the model. Each helper is a child session
//! of the rig agent server (`rigd`), started, steered, waited on, answered
//! and stopped through the same channel and methods (`rig/session/*`) an
//! outside driver uses, as this session's principal. A helper runs in its
//! own worker process, under a grant never wider than its parent's (and
//! sandboxed when narrower), and its asks go to its parent, escalating to
//! the user what the parent may not approve itself.
//!
//! The tools: [`START_AGENT`], [`MESSAGE_AGENT`], [`WAIT_AGENT`],
//! [`ANSWER_AGENT`], [`STOP_AGENT`], [`LIST_AGENTS`]. `wait_agent` is an
//! open call settled by a system when a helper's journal shows its report,
//! an ask or its end (see [`helpers`]); no thread blocks on it.

pub mod helpers;
pub mod link;
pub mod view;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig_harness::prelude::*;
use rig_protocol::event::{AskKind, TurnStatus};
use rig_protocol::methods::{
    Answer, Context, List, Send as SendMethod, SendMode, Start, Started, Stop, StopScope,
};
use rig_protocol::{AskId, DirGrant, Grant, Key, Seq, SessionId};
use schemars::JsonSchema;
use serde::Deserialize;

use helpers::{DeliveredReports, Helper, HelperFeed, HelperJournal, Status};
use link::{AgentsLink, Channel, Feed, call};

pub const START_AGENT: &str = "start_agent";
pub const MESSAGE_AGENT: &str = "message_agent";
pub const WAIT_AGENT: &str = "wait_agent";
pub const ANSWER_AGENT: &str = "answer_agent";
pub const STOP_AGENT: &str = "stop_agent";
pub const LIST_AGENTS: &str = "list_agents";

/// Every tool the plugin adds.
pub const TOOLS: &[&str] = &[
    START_AGENT,
    MESSAGE_AGENT,
    WAIT_AGENT,
    ANSWER_AGENT,
    STOP_AGENT,
    LIST_AGENTS,
];

const START_DESCRIPTION: &str = "Start a helper agent on a task and return its id at once; it works \
in parallel while you go on. It works in your folder (or `cwd`, or on `host`), with your tools and \
permissions or fewer (`tools`, `dirs`), and knows only `task` and the `facts` you pass, so put in \
what it needs: the goal, the files, how to check its work, and what you learned (how to reach the \
remote, missing toolchains, conventions). Its final answer is its report: get it with `wait_agent`. \
Use helpers for independent work that would take you several steps; small lookups are faster to do \
yourself.";

const MESSAGE_DESCRIPTION: &str = "Send a running or idle helper a correction or more work. A \
message is only a message: it never ends the helper's task, and its report on it still comes from \
`wait_agent`. It reads it before its next model call, after the tool it is running finishes; \
`interrupt: true` stops that tool first, for an urgent correction.";

const WAIT_DESCRIPTION: &str = "Wait until one of the helpers in `ids` answers (its report on the \
last message you gave it), needs your approval or an answer, fails or stops, and return that. \
Progress messages never end a wait. Each report is returned once; waiting on helpers that already \
reported returns at once. `timeout_s` (default 600) ends the wait with each helper's progress.";

const ANSWER_DESCRIPTION: &str = "Answer a helper's ask that `wait_agent` returned: an approval \
with `allow` (true or false) or one of its `option` ids, a question with `text`. You may approve \
only what your own permissions allow without asking; anything wider goes to the user, and this \
tool says so.";

const STOP_DESCRIPTION: &str = "Stop a helper and every helper it started. Its work so far stays \
in its transcript; its last report, if any, is what `wait_agent` returns.";

const LIST_DESCRIPTION: &str = "List your helpers, and theirs, with what each is doing.";

/// The helper-agent tools.
#[derive(Default)]
pub struct AgentsPlugin;

impl Plugin for AgentsPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let options = || ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_setting::<view::AgentsSettings>("agents")
            .insert_resource(Adopt { sender, receiver })
            .register_type::<DeliveredReports>()
            .register_required_components::<Agent, DeliveredReports>()
            .keeps_saved::<DeliveredReports>()
            .add_open_tool(START_AGENT, START_DESCRIPTION, options(), on_start)
            .add_open_tool(MESSAGE_AGENT, MESSAGE_DESCRIPTION, options(), on_message)
            .add_open_tool(WAIT_AGENT, WAIT_DESCRIPTION, options(), on_wait)
            .add_open_tool(ANSWER_AGENT, ANSWER_DESCRIPTION, options(), on_answer)
            .add_open_tool(STOP_AGENT, STOP_DESCRIPTION, options(), on_stop)
            .add_open_tool(LIST_AGENTS, LIST_DESCRIPTION, options(), on_list)
            .add_systems(PostStartup, connect_link)
            .add_systems(
                PostUpdate,
                helpers::show_status.in_set(rig_harness::status::StatusSystems::Show),
            )
            .add_systems(
                Update,
                (
                    take_link,
                    recover.run_if(resource_added::<AgentsLink>),
                    adopt,
                    recovered,
                    release_held,
                    helpers::read_feeds,
                    settle_waits,
                )
                    .chain()
                    .after(PollTasks)
                    .before(AgentLoop),
            );
    }
}

/// In a worker of `rigd`: connects this session's own client, with its
/// worker token, off the main thread.
fn connect_link(
    config: Option<Res<rig_harness::worker::WorkerConfig>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(config) = config else {
        return;
    };
    commands.insert_resource(LinkPending);
    let (config, hook) = (config.clone(), wake.clone());
    let future = async move {
        blocking(move || Ok(link::connect(&config, hook)))
            .await
            .unwrap_or_else(|e| Err(e.to_string()))
    };
    commands.spawn((
        Name::new("helpers link"),
        Running::spawn(
            AsyncComputeTaskPool::get_or_init(TaskPool::default),
            &wake,
            future,
        ),
    ));
}

/// Inserts the link once connected; without one the tools refuse, saying
/// why.
fn take_link(made: Query<(Entity, &Done<Result<AgentsLink, String>>)>, mut commands: Commands) {
    for (entity, done) in &made {
        match &done.0 {
            // Calls go on held until its helpers are recovered too.
            Ok(link) => commands.insert_resource(link.clone()),
            Err(why) => {
                bevy_log::warn!("helpers are off: {why}");
                commands.remove_resource::<LinkPending>();
                commands.insert_resource(LinkFailed(why.clone()));
            }
        }
        commands.entity(entity).despawn();
    }
}

/// While this worker's link to `rigd` is being made, and its helpers
/// recovered: tool calls wait for it ([`AwaitsLink`]).
#[derive(Resource, Default)]
pub struct LinkPending;

/// Why this worker has no link to `rigd`.
#[derive(Resource)]
struct LinkFailed(String);

/// On a tool call that came before the link: the call again, once it is
/// there.
#[derive(Component)]
struct AwaitsLink(Option<Box<dyn FnOnce(&mut World) + std::marker::Send + Sync>>);

/// Holds `called` until the link is made, or refuses it when there will be
/// none. True when held.
fn hold_for_link<A: Clone + std::marker::Send + Sync + 'static>(
    called: &ToolCalled<A>,
    pending: bool,
    failed: Option<&LinkFailed>,
    commands: &mut Commands,
) {
    if pending {
        let again = called.clone();
        commands
            .entity(called.call)
            .insert(AwaitsLink(Some(Box::new(move |world: &mut World| {
                world.trigger(again);
            }))));
    } else {
        commands
            .entity(called.call)
            .insert(Outcome::from(no_link(failed.map(|f| f.0.as_str()))));
    }
}

/// Calls each held tool call again, once the link is made (or refused).
fn release_held(world: &mut World) {
    if world.contains_resource::<LinkPending>() {
        return;
    }
    let held: Vec<Entity> = world
        .query_filtered::<Entity, With<AwaitsLink>>()
        .iter(world)
        .collect();
    for call in held {
        let again = world
            .get_mut::<AwaitsLink>(call)
            .and_then(|mut held| held.0.take());
        world.entity_mut(call).remove::<AwaitsLink>();
        if let Some(again) = again {
            again(world);
        }
    }
}

/// Helpers whose start (or recovery) finished off the main thread, for
/// [`adopt`] to spawn.
#[derive(Resource)]
struct Adopt {
    sender: Sender<Adopted>,
    receiver: Receiver<Adopted>,
}

struct Adopted {
    id: SessionId,
    name: String,
    feed: Box<dyn Feed>,
    /// Its events already read from `feed`.
    seen: Vec<rig_protocol::EventRecord>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct StartArgs {
    /// The task, complete: the helper knows nothing else of your conversation.
    task: String,
    /// A short name to refer to it by.
    #[serde(default)]
    name: Option<String>,
    /// The folder it starts in; default: yours.
    #[serde(default)]
    cwd: Option<String>,
    /// A `[remote.hosts]` host to run it on, with that host's paths.
    #[serde(default)]
    host: Option<String>,
    /// Only these of your tools.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Only these folders (and below), writable; it can't touch anything else.
    #[serde(default)]
    dirs: Option<Vec<String>>,
    /// What you learned that it needs: one fact per item.
    #[serde(default)]
    facts: Option<Vec<String>>,
    /// Kinds of call it must ask about before running them (you, or the user, answer):
    /// "shell", "edits", or a tool's name. It always asks at least what you must.
    #[serde(default)]
    ask: Option<Vec<String>>,
    /// Open a live view of it where the user can watch, with the view command the user set (`[agents] view`).
    #[serde(default)]
    view: Option<bool>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MessageArgs {
    /// The helper's id or name.
    id: String,
    text: String,
    /// Stop the tool it is running first.
    #[serde(default)]
    interrupt: Option<bool>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WaitArgs {
    /// The helpers' ids or names; it returns when the first of them settles.
    ids: Vec<String>,
    #[serde(default)]
    timeout_s: Option<u64>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AnswerArgs {
    /// The helper's id or name.
    id: String,
    /// The ask's id, as `wait_agent` gave it.
    ask: String,
    #[serde(default)]
    allow: Option<bool>,
    #[serde(default)]
    option: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct StopArgs {
    /// The helper's id or name.
    id: String,
}

#[derive(Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ListArgs {}

/// The longest a wait may be asked to last.
const MAX_WAIT: Duration = Duration::from_secs(3600);
const DEFAULT_WAIT: Duration = Duration::from_secs(600);

/// On an open `wait_agent` call: the helpers it waits on, until when.
#[derive(Component)]
struct AwaitsHelpers {
    helpers: Vec<Entity>,
    until: SystemTime,
}

/// The idempotency key of a tool call's command: the same call sent again
/// after a restart is the same command.
fn key(me: &SessionId, call: &ToolCall) -> Key {
    Key(format!("{}:{}", me.0, call.id))
}

fn failure(why: impl Into<String>) -> ToolResult {
    ToolResult::failed(ToolExecutionError::other(why.into()))
}

fn success(text: impl Into<String>) -> ToolResult {
    ToolResult::success(text.into().into())
}

/// The refusal of a tool used where no agent server runs.
fn no_link(failed: Option<&str>) -> ToolResult {
    failure(match failed {
        Some(why) => format!(
            "Helpers are off: this session can't reach the rig server ({why}). Do the work yourself."
        ),
        None => "Helpers need the rig agent server: this session runs without one (an old-style \
                 session). Do the work yourself."
            .to_owned(),
    })
}

/// Runs `work` off the main thread for `call`; its text answers the call.
fn answer_with(
    commands: &mut Commands,
    wake: &Wake,
    call: Entity,
    work: impl FnOnce() -> Result<String, String> + std::marker::Send + 'static,
) {
    let future = async move {
        match blocking(move || Ok(work())).await {
            Ok(Ok(text)) => success(text),
            Ok(Err(why)) => failure(why),
            Err(error) => failure(error.to_string()),
        }
    };
    commands.entity(call).insert(Running::answering(
        AsyncComputeTaskPool::get_or_init(TaskPool::default),
        wake,
        future,
    ));
}

/// The refusal's text, with what to do next.
fn refusal(error: &rig_protocol::RpcError) -> String {
    match error.data.as_ref().and_then(|d| d.next.as_deref()) {
        Some(next) => format!("{} (next: {next})", error.message),
        None => error.message.clone(),
    }
}

fn absolute(base: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// The host and folder a `start_agent` call names: `host` and `cwd`, or a
/// `cwd` of `HOST:PATH` (a host name of two or more characters before any
/// `/`) when no `host` is given.
fn remote_place<'a>(
    host: Option<&'a str>,
    cwd: Option<&'a str>,
) -> (Option<&'a str>, Option<&'a str>) {
    if host.is_some() {
        return (host, cwd);
    }
    let Some((name, path)) = cwd.and_then(|c| c.split_once(':')) else {
        return (None, cwd);
    };
    let is_host = name.len() > 1
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'));
    if is_host {
        (Some(name), Some(path))
    } else {
        (None, cwd)
    }
}

/// The start a `start_agent` call asks `rigd` for.
fn start_params(link: &AgentsLink, args: &StartArgs, key: Key) -> Start {
    let (host, cwd) = remote_place(args.host.as_deref(), args.cwd.as_deref());
    // A remote helper's paths are its host's, resolved there (`~` too);
    // a local one's are under this session's folder.
    let place = |path: &str| match host {
        Some(_) => PathBuf::from(path),
        None => absolute(&link.cwd, path),
    };
    let cwd = match (&host, cwd) {
        (_, Some(cwd)) => place(cwd),
        (Some(_), None) => PathBuf::from("~"),
        (None, None) => link.cwd.clone(),
    };
    // Only what it asks to narrow: anything else inherits, server side.
    let grant =
        (args.tools.is_some() || args.dirs.is_some() || args.ask.is_some()).then(|| Grant {
            tools: args.tools.clone(),
            dirs: args.dirs.as_ref().map(|dirs| {
                dirs.iter()
                    .map(|d| DirGrant {
                        path: place(d),
                        write: true,
                    })
                    .collect()
            }),
            net: link.grant.as_ref().is_none_or(|g| g.net),
            // The parent's own up-front rules: answers it would give anyway.
            approve: link
                .grant
                .as_ref()
                .map(|g| g.approve.clone())
                .unwrap_or_default(),
            ask: args.ask.clone().unwrap_or_default(),
            ..Grant::default()
        });
    let mut facts = args.facts.clone().unwrap_or_default();
    if let Some(host) = host {
        facts.push(format!(
            "You run on `{host}`, not on your parent's machine: your paths are `{host}`'s."
        ));
    }
    Start {
        cwd,
        prompt: Some(args.task.clone()),
        name: args.name.clone(),
        host: host.map(str::to_owned).or_else(|| link.host.clone()),
        build: None,
        grant,
        policy: None,
        budget: None,
        context: Some(Context { facts }),
        unload: None,
        view: args.view.unwrap_or(false),
        key,
    }
}

fn on_start(
    called: On<ToolCalled<StartArgs>>,
    link: Option<Res<AgentsLink>>,
    pending: Option<Res<LinkPending>>,
    failed: Option<Res<LinkFailed>>,
    adopt: Res<Adopt>,
    settings: Res<view::AgentsSettings>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(link) = link else {
        hold_for_link(&called, pending.is_some(), failed.as_deref(), &mut commands);
        return;
    };
    let start = start_params(&link, &called.args, key(&link.me, &called.tool_call));
    let name = called
        .args
        .name
        .clone()
        .unwrap_or_else(|| helpers::clip(&called.args.task, 24));
    let wants_view = called.args.view.unwrap_or(false);
    let view = (wants_view && link.unrestricted()).then(|| settings.view.clone());
    let (channel, adopted) = (link.channel.clone(), adopt.sender.clone());
    answer_with(&mut commands, &wake, called.call, move || {
        let Started { session_id, .. } =
            call::<Start>(channel.as_ref(), &start).map_err(|e| refusal(&e))?;
        let mut feed = channel
            .subscribe(&session_id, Seq(0))
            .map_err(|e| refusal(&e))?;
        // A remote helper has started only once its host says so.
        let (seen, confirmed) = match &start.host {
            Some(_) => remote_started(feed.as_mut()),
            None => (Vec::new(), Ok(())),
        };
        adopted
            .send(Adopted {
                id: session_id.clone(),
                name: name.clone(),
                feed,
                seen,
            })
            .ok();
        if let Err(why) = confirmed {
            return Err(format!(
                "The helper `{name}` ({}) did not start on `{}`: {why} (next: check the host's \
                 folder and that `rig` runs there, or do this part yourself)",
                session_id.0,
                start.host.as_deref().unwrap_or_default()
            ));
        }
        let mut text = format!(
            "Started helper `{name}` ({}) in {}. Its report comes from wait_agent; \
             message_agent steers it.",
            session_id.0,
            start.cwd.display()
        );
        if let Some(command) = &view {
            text.push_str(&view::open(command, &name, &session_id));
        } else if wants_view {
            text.push_str(" (No live view: a restricted session doesn't open one.)");
        }
        Ok(text)
    });
}

/// How long a remote start may take to be confirmed.
const REMOTE_START: Duration = Duration::from_secs(60);

/// Reads a remote helper's first events until its host confirms it (its
/// worker is ready) or reports it failed; the events read, and how it went.
/// Unconfirmed in [`REMOTE_START`]: taken as started, its wait settles on
/// whatever comes later.
fn remote_started(feed: &mut dyn Feed) -> (Vec<rig_protocol::EventRecord>, Result<(), String>) {
    use rig_protocol::event::Event;
    let until = std::time::Instant::now() + REMOTE_START;
    let mut seen = Vec::new();
    while std::time::Instant::now() < until {
        let Some(next) = feed.try_next() else {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        };
        let Ok(record) = next else {
            break;
        };
        let event = Event::decode(&record);
        seen.push(record);
        match event {
            Some(Event::WorkerReady(_)) => return (seen, Ok(())),
            Some(Event::SessionFaulted(faulted)) => return (seen, Err(faulted.reason)),
            _ => {}
        }
    }
    (seen, Ok(()))
}

/// Spawns each helper whose start finished.
fn adopt(adopt: Res<Adopt>, helpers: Query<&Helper>, mut commands: Commands) {
    for adopted in adopt.receiver.try_iter() {
        if helpers.iter().any(|h| h.id == adopted.id) {
            continue;
        }
        commands.spawn((
            Name::new(format!("helper {}", adopted.name)),
            Helper {
                id: adopted.id,
                name: adopted.name,
                journal: adopted.seen.iter().fold(
                    HelperJournal::default(),
                    |mut journal, record| {
                        journal.apply(record);
                        journal
                    },
                ),
            },
            HelperFeed(adopted.feed),
        ));
    }
}

/// On a worker's start: follows again every helper this session started,
/// from its journal's start, so what each did is known after a restart.
fn recover(link: Res<AgentsLink>, adopt: Res<Adopt>, wake: Res<Wake>, mut commands: Commands) {
    let (channel, me, adopted) = (link.channel.clone(), link.me.clone(), adopt.sender.clone());
    let future = async move {
        blocking(move || {
            let sessions = call::<List>(
                channel.as_ref(),
                &List {
                    tree: true,
                    all: false,
                },
            )
            .map_err(|e| ToolExecutionError::other(e.message))?;
            for child in sessions.iter().filter(|s| s.parent.as_ref() == Some(&me)) {
                if let Ok(feed) = channel.subscribe(&child.session_id, Seq(0)) {
                    let name = child
                        .name
                        .clone()
                        .unwrap_or_else(|| child.session_id.0.clone());
                    adopted
                        .send(Adopted {
                            id: child.session_id.clone(),
                            name,
                            feed,
                            seen: Vec::new(),
                        })
                        .ok();
                }
            }
            Ok(())
        })
        .await
        .ok();
    };
    commands.spawn((
        Name::new("helpers recovery"),
        Recovery,
        Running::spawn(
            AsyncComputeTaskPool::get_or_init(TaskPool::default),
            &wake,
            future,
        ),
    ));
}

/// On the task recovering this worker's helpers.
#[derive(Component)]
struct Recovery;

/// Ends the wait for the link once its helpers are recovered: the recovery
/// task is done, and every helper it found adopted.
fn recovered(done: Query<Entity, (With<Recovery>, Without<Running>)>, mut commands: Commands) {
    for task in &done {
        commands.entity(task).despawn();
        commands.remove_resource::<LinkPending>();
    }
}

/// The helper `id` names: its session id or its name.
fn find<'a>(
    helpers: impl Iterator<Item = (Entity, &'a Helper)>,
    id: &str,
) -> Option<(Entity, &'a Helper)> {
    let all: Vec<_> = helpers.collect();
    all.iter()
        .find(|(_, h)| h.id.0 == id)
        .or_else(|| all.iter().find(|(_, h)| h.name == id))
        .copied()
}

fn unknown(id: &str, helpers: &Query<(Entity, &Helper)>) -> ToolResult {
    let names: Vec<String> = helpers
        .iter()
        .map(|(_, h)| format!("`{}` ({})", h.name, h.id.0))
        .collect();
    let have = if names.is_empty() {
        "You have no helpers.".to_owned()
    } else {
        format!("Your helpers: {}.", names.join(", "))
    };
    failure(format!("`{id}` is not one of your helpers. {have}"))
}

fn on_message(
    called: On<ToolCalled<MessageArgs>>,
    link: Option<Res<AgentsLink>>,
    pending: Option<Res<LinkPending>>,
    failed: Option<Res<LinkFailed>>,
    helpers: Query<(Entity, &Helper)>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(link) = link else {
        hold_for_link(&called, pending.is_some(), failed.as_deref(), &mut commands);
        return;
    };
    let Some((_, helper)) = find(helpers.iter(), &called.args.id) else {
        let refused = unknown(&called.args.id, &helpers);
        commands.entity(called.call).insert(Outcome::from(refused));
        return;
    };
    let interrupt = called.args.interrupt.unwrap_or(false);
    let send = SendMethod {
        session_id: helper.id.clone(),
        text: Some(called.args.text.clone()),
        content: None,
        mode: Some(if interrupt {
            SendMode::Interrupt
        } else {
            SendMode::Steer
        }),
        key: key(&link.me, &called.tool_call),
    };
    let (channel, name) = (link.channel.clone(), helper.name.clone());
    answer_with(&mut commands, &wake, called.call, move || {
        let sent = call::<SendMethod>(channel.as_ref(), &send).map_err(|e| refusal(&e))?;
        let when = match sent.disposition {
            rig_protocol::methods::Disposition::Started => "It starts on it now.",
            rig_protocol::methods::Disposition::Steered if interrupt => {
                "It stopped its running step and reads it now."
            }
            rig_protocol::methods::Disposition::Steered => {
                "It reads it before its next model call."
            }
            rig_protocol::methods::Disposition::Queued => "It reads it when its current turn ends.",
        };
        Ok(format!(
            "Sent to `{name}`. {when} Its report on it comes from wait_agent."
        ))
    });
}

fn on_wait(
    called: On<ToolCalled<WaitArgs>>,
    helpers: Query<(Entity, &Helper)>,
    pending: Option<Res<LinkPending>>,
    mut commands: Commands,
) {
    if pending.is_some() {
        hold_for_link(&called, true, None, &mut commands);
        return;
    }
    if called.args.ids.is_empty() {
        let refused = failure("Name the helpers to wait on in `ids`.");
        commands.entity(called.call).insert(Outcome::from(refused));
        return;
    }
    let mut waits = Vec::new();
    for id in &called.args.ids {
        match find(helpers.iter(), id) {
            Some((entity, _)) => waits.push(entity),
            None => {
                let refused = unknown(id, &helpers);
                commands.entity(called.call).insert(Outcome::from(refused));
                return;
            }
        }
    }
    let wait = called
        .args
        .timeout_s
        .map_or(DEFAULT_WAIT, Duration::from_secs)
        .min(MAX_WAIT);
    let until = SystemTime::now() + wait;
    commands.entity(called.call).insert((
        AwaitsHelpers {
            helpers: waits,
            until,
        },
        WakeAt(until),
    ));
}

/// Settles each open `wait_agent` call whose helpers' journals say so.
fn settle_waits(
    waits: Query<(Entity, &AwaitsHelpers, &ForAgent), Without<Outcome>>,
    helpers: Query<&Helper>,
    mut delivered: Query<&mut DeliveredReports>,
    mut commands: Commands,
) {
    let now = SystemTime::now();
    for (call, wait, ForAgent(agent)) in &waits {
        let Ok(mut reports) = delivered.get_mut(*agent) else {
            continue;
        };
        let waited: Vec<&Helper> = wait
            .helpers
            .iter()
            .filter_map(|&h| helpers.get(h).ok())
            .collect();
        let settled = waited.iter().find_map(|h| {
            let status = h.journal.status(reports.of(&h.id));
            status.settles().then_some((*h, status))
        });
        let text = match settled {
            Some((helper, status)) => {
                if let Status::Report(ended) = &status {
                    let _ = ended;
                    reports
                        .0
                        .insert(helper.id.0.clone(), helper.journal.asked.0);
                }
                let mut text = settled_text(helper, &status);
                let others: Vec<String> = waited
                    .iter()
                    .filter(|h| h.id != helper.id)
                    .map(|h| helpers::line(h, reports.of(&h.id)))
                    .collect();
                if !others.is_empty() {
                    text.push_str("\n\nThe others: ");
                    text.push_str(&others.join("; "));
                }
                text
            }
            None if waited
                .iter()
                .all(|h| h.journal.status(reports.of(&h.id)) == Status::Reported) =>
            {
                let lines: Vec<String> = waited
                    .iter()
                    .map(|h| helpers::line(h, reports.of(&h.id)))
                    .collect();
                format!(
                    "Nothing to wait for: {}. Give one more work with message_agent.",
                    lines.join("; ")
                )
            }
            None if now >= wait.until => {
                let lines: Vec<String> = waited
                    .iter()
                    .map(|h| helpers::line(h, reports.of(&h.id)))
                    .collect();
                format!(
                    "Still waiting: {}. Wait again, or steer one with message_agent.",
                    lines.join("; ")
                )
            }
            None => continue,
        };
        commands.entity(call).insert(Outcome::from(success(text)));
    }
}

/// A settled helper as the model reads it.
fn settled_text(helper: &Helper, status: &Status) -> String {
    let who = format!("`{}` ({})", helper.name, helper.id.0);
    match status {
        Status::Report(ended) => {
            let answer = ended.answer.clone().unwrap_or_default();
            match ended.status {
                TurnStatus::Answered => format!("{who} answered:\n{answer}"),
                TurnStatus::Failed => format!(
                    "{who} failed: {}\n{answer}",
                    ended.error.clone().unwrap_or_default()
                ),
                TurnStatus::Stopped => format!("{who} was stopped. What it had:\n{answer}"),
            }
        }
        Status::Asks(ask) => {
            let options: Vec<String> = ask
                .options
                .iter()
                .map(|o| format!("`{}` ({})", o.option_id, o.label))
                .collect();
            match ask.kind {
                AskKind::Approval => format!(
                    "{who} needs approval (ask `{}`): {}\n{}\nOptions: {}. Answer with \
                     answer_agent {{\"id\": \"{}\", \"ask\": \"{}\", \"allow\": true}} or false. \
                     Beyond what you may approve yourself, the user answers it instead.",
                    ask.ask_id.0,
                    ask.title,
                    ask.body,
                    options.join(", "),
                    helper.id.0,
                    ask.ask_id.0
                ),
                AskKind::Question => format!(
                    "{who} asks you (ask `{}`): {}\n{}\nReply with answer_agent {{\"id\": \"{}\", \
                     \"ask\": \"{}\", \"text\": \"…\"}}.",
                    ask.ask_id.0, ask.title, ask.body, helper.id.0, ask.ask_id.0
                ),
            }
        }
        Status::Stopped => format!("{who} stopped."),
        Status::Faulted(reason) => format!("{who} failed and stopped: {reason}"),
        Status::Reported | Status::Working => helpers::line(helper, rig_protocol::CommandSeq(0)),
    }
}

fn on_answer(
    called: On<ToolCalled<AnswerArgs>>,
    link: Option<Res<AgentsLink>>,
    pending: Option<Res<LinkPending>>,
    failed: Option<Res<LinkFailed>>,
    helpers: Query<(Entity, &Helper)>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(link) = link else {
        hold_for_link(&called, pending.is_some(), failed.as_deref(), &mut commands);
        return;
    };
    let Some((_, helper)) = find(helpers.iter(), &called.args.id) else {
        let refused = unknown(&called.args.id, &helpers);
        commands.entity(called.call).insert(Outcome::from(refused));
        return;
    };
    let args = &called.args;
    let ask = helper.journal.asks.iter().find(|a| a.ask_id.0 == args.ask);
    let option = args.option.clone().or_else(|| {
        let allow = args.allow?;
        let wanted = if allow { "allow" } else { "reject" };
        let by_kind = ask.and_then(|a| {
            a.options
                .iter()
                .find(|o| o.kind.starts_with(wanted) || o.option_id.starts_with(wanted))
        });
        Some(by_kind.map_or_else(
            || if allow { "allow" } else { "deny" }.to_owned(),
            |o| o.option_id.clone(),
        ))
    });
    if option.is_none() && args.text.is_none() {
        let refused = failure("Say `allow`, an `option` id, or a `text` answer.");
        commands.entity(called.call).insert(Outcome::from(refused));
        return;
    }
    let answer = Answer {
        session_id: helper.id.clone(),
        ask_id: AskId(args.ask.clone()),
        option_id: option,
        text: args.text.clone(),
        key: key(&link.me, &called.tool_call),
    };
    let (channel, name) = (link.channel.clone(), helper.name.clone());
    answer_with(&mut commands, &wake, called.call, move || {
        match call::<Answer>(channel.as_ref(), &answer) {
            Ok(_) => Ok(format!("Answered `{name}`; it goes on.")),
            Err(e) if e.code == rig_protocol::codes::UNAUTHORIZED => Err(format!(
                "That is beyond what you may approve yourself, so the user answers it. Wait on \
                 `{name}` again, or do something else meanwhile. ({})",
                e.message
            )),
            Err(e) if e.code == rig_protocol::codes::ALREADY_RESOLVED => {
                Err(format!("That ask was already answered: {}", e.message))
            }
            Err(e) => Err(refusal(&e)),
        }
    });
}

fn on_stop(
    called: On<ToolCalled<StopArgs>>,
    link: Option<Res<AgentsLink>>,
    pending: Option<Res<LinkPending>>,
    failed: Option<Res<LinkFailed>>,
    helpers: Query<(Entity, &Helper)>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(link) = link else {
        hold_for_link(&called, pending.is_some(), failed.as_deref(), &mut commands);
        return;
    };
    let Some((_, helper)) = find(helpers.iter(), &called.args.id) else {
        let refused = unknown(&called.args.id, &helpers);
        commands.entity(called.call).insert(Outcome::from(refused));
        return;
    };
    let stop = Stop {
        session_id: helper.id.clone(),
        scope: StopScope::Session,
        key: key(&link.me, &called.tool_call),
    };
    let (channel, name) = (link.channel.clone(), helper.name.clone());
    answer_with(&mut commands, &wake, called.call, move || {
        let stopping = call::<Stop>(channel.as_ref(), &stop).map_err(|e| refusal(&e))?;
        let n = stopping.stopping.len().saturating_sub(1);
        Ok(match n {
            0 => format!("Stopping `{name}`."),
            n => format!("Stopping `{name}` and {n} helper(s) it started."),
        })
    });
}

fn on_list(
    called: On<ToolCalled<ListArgs>>,
    link: Option<Res<AgentsLink>>,
    pending: Option<Res<LinkPending>>,
    failed: Option<Res<LinkFailed>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(link) = link else {
        hold_for_link(&called, pending.is_some(), failed.as_deref(), &mut commands);
        return;
    };
    let (channel, me) = (link.channel.clone(), link.me.clone());
    answer_with(&mut commands, &wake, called.call, move || {
        let sessions = call::<List>(
            channel.as_ref(),
            &List {
                tree: true,
                all: false,
            },
        )
        .map_err(|e| refusal(&e))?;
        let lines = helpers::tree_lines(&sessions, &me);
        Ok(if lines.is_empty() {
            "You have no helpers.".to_owned()
        } else {
            lines.join("\n")
        })
    });
}

/// A link for tests and embedders: `channel` as session `me` in `cwd`.
pub fn link(
    channel: Arc<dyn Channel>,
    me: SessionId,
    cwd: PathBuf,
    grant: Option<Grant>,
) -> AgentsLink {
    AgentsLink {
        channel,
        me,
        cwd,
        host: None,
        grant,
    }
}

#[cfg(test)]
mod tests;
