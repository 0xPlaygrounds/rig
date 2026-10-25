//! Plugin corpus: 47. Chat bot bridge (Slack/Matrix) (spec `47-chat-bridge.md`).
//!
//! Each chat thread that mentions the bot gets a top-level agent of its
//! own. Later messages in the thread are queued to that agent, each turn's
//! final answer is posted back (split when long), tool activity is one
//! edited "working…" post, and the questions other plugins ask the agent
//! ([`Ask`]) are posted as "Reply `yes` or `no`": the thread is the
//! agent's [`Answerer`]. The thread an agent belongs to is a saved
//! component, so a restart keeps the conversations.
//!
//! The network is a [`ChatTransport`] the app inserts with
//! [`ChatBridge::new`]; posts it cannot send are kept and retried with
//! backoff. `rig --chat` makes the bridge the run's front, a bot with no
//! terminal; without it, it runs beside the terminal view.

use std::collections::{BTreeMap, VecDeque};

use bevy_log::info;
use rig_harness::pick::PickPlugin;
use rig_harness::prelude::*;
use rig_harness::rig_core::redact;
use rig_harness::rig_tools::{Edit, Read, Search, Shell, Write};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The longest post; a longer answer is split.
const MAX_POST: usize = 3500;
/// The longest wait between attempts to send a kept post.
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// A message the transport received.
#[derive(Clone, Debug)]
pub struct ChatMessage {
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
/// its own connection, with its credentials from the environment, and
/// reconnects by itself.
pub trait ChatTransport: Send + Sync + 'static {
    /// Starts it; it calls `wake` whenever a message arrives.
    fn start(&mut self, wake: Wake);
    /// The messages received since the last call, without waiting.
    fn receive(&mut self) -> Vec<ChatMessage>;
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

/// Who may talk to the bot, and where: the `[chat]` table of the settings.
///
/// ```toml
/// [chat]
/// channels = ["C0123"]
/// users = ["U0456"]
/// ```
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatSettings {
    /// The channel ids.
    pub channels: Vec<String>,
    /// The user ids.
    pub users: Vec<String>,
}

impl ChatSettings {
    fn allows(&self, message: &ChatMessage) -> bool {
        self.channels.contains(&message.channel) && self.users.contains(&message.user)
    }
}

/// The chat thread an agent answers. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
#[require(Working)]
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
    /// The counts changed since the post last showed them.
    stale: bool,
}

impl Working {
    /// Posts the "working…" line, or edits it, when the counts changed
    /// since it last showed them.
    fn show(&mut self, thread: &ChatThread, bridge: &mut ChatBridge) {
        if !std::mem::take(&mut self.stale) {
            return;
        }
        let text = format!("working… {}", summary(&self.counts));
        match self.post {
            Some(id) => bridge.send(thread, id, true, text),
            None => self.post = Some(bridge.post(thread, text)),
        }
    }
}

/// Bridges chat threads to agents.
#[derive(Default)]
pub struct ChatBridgePlugin;

impl Plugin for ChatBridgePlugin {
    fn build(&self, app: &mut App) {
        // It answers questions with `Pick`, which this carries out.
        if !app.is_plugin_added::<PickPlugin>() {
            app.add_plugins(PickPlugin);
        }
        app.add_setting::<ChatSettings>("chat")
            .add_front(Front::new("chat").on_flag("chat", "Answer a Slack or Matrix channel"))
            .init_resource::<ChatBridge>()
            .register_type::<ChatThread>()
            .add_systems(Startup, start)
            .add_systems(Update, receive.before(AgentLoop))
            .add_systems(PostUpdate, (show_work, report_recalled, flush).chain())
            .add_observer(answer_for_thread)
            .add_observer(post_question)
            .add_observer(count_tool_call)
            .add_observer(post_answer);
    }
}

fn start(mut bridge: ResMut<ChatBridge>, wake: Res<Wake>) {
    if let Some(transport) = &mut bridge.transport {
        transport.start(wake.clone());
    }
}

/// A thread answers its agent's questions, from when it is bound or
/// restored.
fn answer_for_thread(bound: On<Insert<ChatThread>>, mut commands: Commands) {
    commands.entity(bound.entity).insert(Answerer {
        agent: Some(bound.entity),
    });
}

/// The questions not answered yet, for the agent each is about.
type OpenAsks<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Ask,
        &'static ForAgent,
        &'static Effect,
        &'static Decision,
    ),
    (PluginEffects, Without<Outcome>, Without<Settled>),
>;

/// Routes each message received: as the answer to its thread's oldest open
/// question (a [`Pick`], as any front answers), to its thread's agent, or
/// to a new agent when it mentions the bot.
fn receive(
    mut bridge: ResMut<ChatBridge>,
    settings: Res<ChatSettings>,
    threads: Query<(Entity, &ChatThread)>,
    asks: OpenAsks,
    primaries: MainAgentQuery,
    models: Query<(Option<&ModelChoice>, &ReasoningSetting, Option<&Connection>)>,
    mut commands: Commands,
) {
    let Some(transport) = &mut bridge.transport else {
        return;
    };
    let mut spawned: Vec<(Entity, ChatThread)> = Vec::new();
    let mut answered = Vec::new();
    for message in transport.receive() {
        if !settings.allows(&message) {
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
        let bound = threads
            .iter()
            .chain(spawned.iter().map(|(agent, bound)| (*agent, bound)))
            .find(|(_, bound)| **bound == thread)
            .map(|(agent, _)| agent);
        let agent = match bound {
            Some(agent) => agent,
            None if message.mentions_bot => {
                let parent = main_agent(&primaries).and_then(|agent| models.get(agent).ok());
                let agent = spawn_agent(&mut commands, thread.clone(), parent);
                spawned.push((agent, thread));
                agent
            }
            None => continue,
        };
        let open = asks
            .iter()
            .filter(|(ask, _, to, _, decision)| {
                to.0 == agent && !decision.is_decided() && !answered.contains(ask)
            })
            .min_by_key(|(.., effect, _)| effect.id());
        if let Some((ask, question, ..)) = open
            && let Some(index) = choice(question, &message.text)
        {
            answered.push(ask);
            let chosen = Some(Chosen::item(index));
            commands.trigger(Pick::person(ask, chosen));
            continue;
        }
        // A question that takes text is answered by this line, by the
        // harness, before it reaches the model. The author keeps who
        // wrote it.
        let author = Author::user().titled(message.user);
        let deliver = Deliver::new(agent, message.text, DeliveryMode::Queue);
        commands.trigger(deliver.with_author(author).literal());
    }
}

/// The position of the choice a reply names, by id or label.
fn choice(ask: &Ask, reply: &str) -> Option<usize> {
    let reply = reply.trim();
    ask.choices.iter().position(|choice| {
        choice.id.eq_ignore_ascii_case(reply) || choice.label.eq_ignore_ascii_case(reply)
    })
}

/// A top-level agent for `thread`, on the model of the agent the user
/// talks to.
fn spawn_agent(
    commands: &mut Commands,
    thread: ChatThread,
    parent: Option<(Option<&ModelChoice>, &ReasoningSetting, Option<&Connection>)>,
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

/// Posts each question asked about an agent with a thread.
fn post_question(
    asked: On<Add<Ask>>,
    asks: Query<(&Ask, &ForAgent)>,
    threads: Query<&ChatThread>,
    mut bridge: ResMut<ChatBridge>,
) {
    let Ok((ask, &ForAgent(agent))) = asks.get(asked.entity) else {
        return;
    };
    let Ok(thread) = threads.get(agent) else {
        return;
    };
    let ids: Vec<String> = ask
        .choices
        .iter()
        .map(|choice| format!("`{}`", choice.id))
        .collect();
    let mut text = ask.title.clone();
    if !ask.body.is_empty() {
        text.push_str(&format!("\n```\n{}\n```", ask.body));
    }
    if !ids.is_empty() {
        text.push_str(&format!("\nReply {}.", ids.join(" or ")));
    }
    bridge.post(thread, text);
}

/// Counts each tool call of an agent with a thread, once: not again when a
/// restart runs it again.
fn count_tool_call(
    call: On<Add<ToolRequest>>,
    calls: Query<(&ToolRequest, &ForAgent), Without<Restored>>,
    mut working: Query<&mut Working>,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(call.entity)
        && let Ok(mut working) = working.get_mut(agent)
    {
        let kind = kind(request.call.function.name.as_str());
        *working.counts.entry(kind).or_default() += 1;
        working.stale = true;
    }
}

/// What a tool's calls are counted as.
fn kind(tool: &str) -> &'static str {
    match tool {
        Read::NAME => "read",
        Edit::NAME | Write::NAME => "edited",
        Shell::NAME => "ran",
        Search::NAME => "searched",
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
        working.bypass_change_detection().show(thread, &mut bridge);
    }
}

/// Posts each turn's final answer, or why it ended without one, to its
/// agent's thread, after its last "working…" line, which a turn that ended
/// in the frame its calls started would otherwise never show.
fn post_answer(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut agents: Query<(&ChatThread, &mut Working)>,
    mut bridge: ResMut<ChatBridge>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((thread, mut working)) = agents.get_mut(agent) else {
        return;
    };
    let working = working.bypass_change_detection();
    working.show(thread, &mut bridge);
    *working = Working::default();
    let text = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => format!("The turn failed: {why}"),
        TurnOutcome::Stopped(why) => format!(
            "Stopped{}.",
            why.as_ref()
                .map(|why| format!(": {why}"))
                .unwrap_or_default()
        ),
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

/// Says in the thread which messages a turn that ended did not send.
fn report_recalled(
    mut recalled: MessageReader<Recalled>,
    threads: Query<&ChatThread>,
    mut bridge: ResMut<ChatBridge>,
) {
    for recalled in recalled.read() {
        if let Ok(thread) = threads.get(recalled.agent) {
            let text = redact::text(&recalled.text);
            let text = format!("Not sent, send it again: {text}");
            bridge.post(thread, text);
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
