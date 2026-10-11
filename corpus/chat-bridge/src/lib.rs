//! Plugin corpus: 47. Chat bot bridge (Slack/Matrix) (spec `47-chat-bridge.md`).
//!
//! Each chat thread that mentions the bot gets a top-level agent of its
//! own. Later messages in the thread are queued to that agent, each turn's
//! final answer is posted back (split when long), tool activity is one
//! edited "working…" post, and a question another plugin asks with a
//! picker is posted as "Reply `yes` or `no`". The thread an agent belongs
//! to is a saved component, so a restart keeps the conversations.
//!
//! The network is a [`ChatTransport`] the app inserts with
//! [`ChatBridge::new`]; posts it cannot send are kept and retried with
//! backoff.

use std::collections::{BTreeMap, VecDeque};

use bevy_log::info;
use rig_harness::prelude::*;
use rig_tui::{PickItem, PickRequest};

#[cfg(test)]
mod tests;

/// The longest post; a longer answer is split.
const MAX_POST: usize = 3500;
/// The longest wait between attempts to send a kept post.
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// A message the transport received.
#[derive(Clone, Debug)]
pub struct Incoming {
    /// The channel it was posted in.
    pub channel: String,
    /// The thread it belongs to: a top-level message starts its own.
    pub thread: String,
    /// Who wrote it.
    pub user: String,
    /// Its text, without the mention of the bot.
    pub text: String,
    /// Whether it mentions the bot.
    pub mentions_bot: bool,
}

/// A post to a thread, or an edit of an earlier one with the same `id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Post {
    /// The channel.
    pub channel: String,
    /// The thread.
    pub thread: String,
    /// The bridge's id of the post, which the transport maps to its own.
    pub id: u64,
    /// Whether it replaces the text of the post `id`.
    pub edit: bool,
    /// The text.
    pub text: String,
}

/// The transport could not send: it is reconnecting.
#[derive(Clone, Debug)]
pub struct Disconnected(pub String);

/// A chat network (Slack socket mode, Matrix, or a fake in tests). It runs
/// its own connection and reconnects by itself.
pub trait ChatTransport: Send + Sync + 'static {
    /// Starts it; it calls `wake` whenever a message arrives.
    fn start(&mut self, wake: Wake);
    /// The messages received since the last call, without waiting.
    fn receive(&mut self) -> Vec<Incoming>;
    /// Sends `post`, or says it is disconnected; the bridge then keeps it.
    fn send(&mut self, post: &Post) -> Result<(), Disconnected>;
}

/// The transport, and the posts not sent yet.
#[derive(Resource, Default)]
pub struct ChatBridge {
    transport: Option<Box<dyn ChatTransport>>,
    outbox: VecDeque<Post>,
    next_id: u64,
    failures: u32,
    paused: bool,
}

impl ChatBridge {
    /// A bridge on `transport`.
    pub fn new(transport: impl ChatTransport) -> Self {
        Self {
            transport: Some(Box::new(transport)),
            ..Self::default()
        }
    }

    /// Keeps a new post of `text` to `thread`, and returns its id.
    fn post(&mut self, thread: &ChatThread, text: impl Into<String>) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.send(thread, id, false, text.into());
        id
    }

    fn send(&mut self, thread: &ChatThread, id: u64, edit: bool, text: String) {
        self.outbox.push_back(Post {
            channel: thread.channel.clone(),
            thread: thread.thread.clone(),
            id,
            edit,
            text,
        });
    }
}

/// Who may talk to the bot, and where; from `RIG_CHAT_CHANNELS` and
/// `RIG_CHAT_USERS` (comma-separated ids).
#[derive(Resource, Clone, Debug, Default)]
pub struct ChatConfig {
    /// The channel ids.
    pub channels: Vec<String>,
    /// The user ids.
    pub users: Vec<String>,
}

impl ChatConfig {
    /// The configuration in the environment.
    pub fn from_env() -> Self {
        let list = |name| {
            std::env::var(name)
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect()
        };
        Self {
            channels: list("RIG_CHAT_CHANNELS"),
            users: list("RIG_CHAT_USERS"),
        }
    }

    fn allows(&self, message: &Incoming) -> bool {
        self.channels.contains(&message.channel) && self.users.contains(&message.user)
    }
}

/// The chat thread an agent answers. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct ChatThread {
    /// The channel id.
    pub channel: String,
    /// The thread id.
    pub thread: String,
}

/// The tool calls of the running turn, by kind, and their "working…" post.
#[derive(Component, Default)]
struct Working {
    post: Option<u64>,
    counts: BTreeMap<&'static str, u32>,
}

/// A question asked of the thread, answered by `yes` or `no`.
#[derive(Component)]
struct Asking(Vec<PickItem>);

/// Messages for agents, delivered once the agents spawned this frame are
/// ready for them.
#[derive(Resource, Default)]
struct Arrivals(Vec<(Entity, String)>);

/// Bridges chat threads to agents.
#[derive(Default)]
pub struct ChatBridgePlugin;

impl Plugin for ChatBridgePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<ChatConfig>() {
            app.insert_resource(ChatConfig::from_env());
        }
        app.init_resource::<ChatBridge>()
            .init_resource::<Arrivals>()
            .register_type::<ChatThread>()
            .register_required_components::<ChatThread, Working>()
            .add_message::<PickRequest>()
            .add_systems(Startup, start)
            .add_systems(Update, (receive, deliver).chain())
            .add_systems(PostUpdate, (ask, show_work, report_recalled, flush).chain())
            .add_observer(count_tool_call)
            .add_observer(post_answer);
    }
}

fn start(mut bridge: ResMut<ChatBridge>, wake: Res<Wake>) {
    if let Some(transport) = &mut bridge.transport {
        transport.start(wake.clone());
    }
}

/// Routes each message received: to its thread's agent, as the answer to
/// its question, or to a new agent when it mentions the bot.
fn receive(
    mut bridge: ResMut<ChatBridge>,
    config: Res<ChatConfig>,
    threads: Query<(Entity, &ChatThread, Option<&Asking>)>,
    primaries: PrimaryQuery,
    models: Query<(Option<&ModelChoice>, &Effort, Option<&Connection>)>,
    mut arrivals: ResMut<Arrivals>,
    mut commands: Commands,
) {
    let Some(transport) = &mut bridge.transport else {
        return;
    };
    for message in transport.receive() {
        if !config.allows(&message) {
            info!(
                "chat bridge: ignored a message of {} in {}",
                message.user, message.channel
            );
            continue;
        }
        let thread = ChatThread {
            channel: message.channel,
            thread: message.thread,
        };
        let bound = threads.iter().find(|(_, bound, _)| **bound == thread);
        if let Some((agent, _, Some(Asking(items)))) = bound
            && let Some(item) = choice(items, &message.text)
        {
            commands.entity(agent).remove::<Asking>();
            commands.trigger(RunCommand {
                entity: agent,
                line: item.command.clone(),
            });
            bridge.post(&thread, format!("Chose: {}.", item.label));
            continue;
        }
        let agent = match bound {
            Some((agent, ..)) => agent,
            None if message.mentions_bot => {
                let parent = primary(&primaries).and_then(|agent| models.get(agent).ok());
                spawn_agent(&mut commands, thread, parent)
            }
            None => continue,
        };
        arrivals.0.push((agent, message.text));
    }
}

/// The item a reply chooses: `yes` the first that does not deny, `no` the
/// first that does.
fn choice<'a>(items: &'a [PickItem], reply: &str) -> Option<&'a PickItem> {
    let denies = |item: &&PickItem| item.label.to_lowercase().starts_with("deny");
    match reply.trim().to_lowercase().as_str() {
        "yes" | "y" => items.iter().find(|item| !denies(item)),
        "no" | "n" => items.iter().find(denies),
        _ => None,
    }
}

/// A top-level agent for `thread`, on the model of the agent the user
/// talks to.
fn spawn_agent(
    commands: &mut Commands,
    thread: ChatThread,
    parent: Option<(Option<&ModelChoice>, &Effort, Option<&Connection>)>,
) -> Entity {
    let name = Name::new(format!("chat {}", thread.thread));
    let mut agent = commands.spawn((name, thread));
    if let Some((choice, effort, connection)) = parent {
        // Before the model choice, so connecting it keeps this connection.
        if let Some(connection) = connection {
            agent.insert(connection.clone());
        }
        agent.insert(*effort);
        if let Some(choice) = choice {
            agent.insert(choice.clone());
        }
    }
    agent.insert(Agent);
    agent.id()
}

/// Sends what arrived, queued behind a running turn.
fn deliver(mut arrivals: ResMut<Arrivals>, mut commands: Commands) {
    for (agent, text) in arrivals.0.drain(..) {
        commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).literal());
    }
}

/// Counts each tool call of an agent with a thread.
fn count_tool_call(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    mut working: Query<&mut Working>,
) {
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    if let Ok(TurnOf(agent)) = turns.get(*turn)
        && let Ok(mut working) = working.get_mut(*agent)
    {
        *working
            .counts
            .entry(kind(run.call.function.name.as_str()))
            .or_default() += 1;
    }
}

/// What a tool's calls are counted as.
fn kind(tool: &str) -> &'static str {
    match tool {
        "read" => "read",
        "edit" | "write" => "edited",
        "shell" => "ran",
        "search" => "searched",
        _ => "called",
    }
}

/// "read 3 files, ran 1 command".
fn summary(counts: &BTreeMap<&'static str, u32>) -> String {
    let said: Vec<String> = counts
        .iter()
        .map(|(kind, n)| {
            let (one, many) = match *kind {
                "read" | "edited" => ("file", "files"),
                "ran" => ("command", "commands"),
                "searched" => ("time", "times"),
                _ => ("tool", "tools"),
            };
            format!("{kind} {n} {}", if *n == 1 { one } else { many })
        })
        .collect();
    said.join(", ")
}

/// Posts, then edits, each thread's "working…" line as its calls change.
fn show_work(
    mut agents: Query<(&ChatThread, &mut Working), Changed<Working>>,
    mut bridge: ResMut<ChatBridge>,
) {
    for (thread, mut working) in &mut agents {
        if working.counts.is_empty() {
            continue;
        }
        let text = format!("working… {}", summary(&working.counts));
        match working.post {
            Some(id) => bridge.send(thread, id, true, text),
            None => working.bypass_change_detection().post = Some(bridge.post(thread, text)),
        }
    }
}

/// Posts each turn's final answer, or its failure, to the agent's thread.
fn post_answer(
    ended: On<TurnEnded>,
    mut agents: Query<(&ChatThread, &mut Working)>,
    mut bridge: ResMut<ChatBridge>,
) {
    let Ok((thread, mut working)) = agents.get_mut(ended.entity) else {
        return;
    };
    *working.bypass_change_detection() = Working::default();
    let text = match &ended.outcome {
        TurnOutcome::Answered(message) => final_answer(message).unwrap_or_default(),
        TurnOutcome::Failed(why) => format!("The turn failed: {why}"),
        TurnOutcome::Stopped => "Stopped.".to_owned(),
    };
    for part in split(&text, MAX_POST) {
        bridge.post(thread, part);
    }
}

/// `text` in parts of at most `max` characters, split at line ends when
/// it can be.
fn split(text: &str, max: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut part = String::new();
    for line in text.split_inclusive('\n') {
        let mut line = line;
        while !line.is_empty() {
            let room = max.saturating_sub(part.chars().count());
            if line.chars().count() <= room {
                part.push_str(line);
                break;
            }
            if part.is_empty() {
                let at = line
                    .char_indices()
                    .nth(room)
                    .map_or(line.len(), |(at, _)| at);
                let (head, rest) = line.split_at(at);
                part.push_str(head);
                line = rest;
            }
            parts.push(std::mem::take(&mut part));
        }
    }
    parts.push(part);
    parts.retain(|part| !part.trim().is_empty());
    parts
}

/// Posts each question another plugin asks an agent with a thread.
fn ask(
    mut picks: MessageReader<PickRequest>,
    threads: Query<&ChatThread>,
    mut bridge: ResMut<ChatBridge>,
    mut commands: Commands,
) {
    for pick in picks.read() {
        if let Ok(thread) = threads.get(pick.agent) {
            bridge.post(thread, format!("{}\nReply `yes` or `no`.", pick.title));
            commands
                .entity(pick.agent)
                .insert(Asking(pick.items.clone()));
        }
    }
}

/// Says in the thread which messages a turn that stopped did not send.
fn report_recalled(
    mut recalled: MessageReader<Recalled>,
    threads: Query<&ChatThread>,
    mut bridge: ResMut<ChatBridge>,
) {
    for recalled in recalled.read() {
        if let Ok(thread) = threads.get(recalled.agent) {
            bridge.post(
                thread,
                format!("Not sent, send it again: {}", recalled.text),
            );
        }
    }
}

/// Sends the kept posts in order; on a disconnect, keeps them and tries
/// again after a wait that doubles up to [`MAX_BACKOFF`].
fn flush(mut bridge: ResMut<ChatBridge>, mut commands: Commands) {
    let bridge = &mut *bridge;
    let Some(transport) = &mut bridge.transport else {
        return;
    };
    if bridge.paused {
        return;
    }
    while let Some(post) = bridge.outbox.front() {
        if let Err(Disconnected(why)) = transport.send(post) {
            bridge.failures += 1;
            bridge.paused = true;
            let wait = Duration::from_secs(1 << bridge.failures.min(6)).min(MAX_BACKOFF);
            info!("chat bridge: disconnected ({why}); trying again in {wait:?}");
            commands
                .delayed()
                .duration(wait)
                .queue(|world: &mut World| {
                    if let Some(mut bridge) = world.get_resource_mut::<ChatBridge>() {
                        bridge.paused = false;
                    }
                });
            return;
        }
        bridge.failures = 0;
        bridge.outbox.pop_front();
    }
}
