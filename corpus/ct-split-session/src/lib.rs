//! CT-10: split a tangent into its own session (conversation-over-time
//! fresh held-out spec).
//!
//! `/split <from-turn> [<to-turn>] [--name <name>]` moves a range of the
//! agent's turns (to the end by default) into a new saved session of the
//! same project, and `/split undo` puts the last split's turns back while
//! neither session changed since.
//!
//! Turns are the conversation's turn records ([`TurnRecord`]) whose first
//! message is still in it, numbered from 1. A range boundary inside a tool
//! exchange moves to the nearest turn start outside one, and the user is
//! told. A split asked for while a turn runs waits until it ended; while
//! one runs, the agent's new messages wait on a hold.
//!
//! Two model calls of the plugin's own (the `[split] model`, else the
//! agent's) summarize what came before the range, as the new session's
//! context, and what the range concluded, for the note left behind. The
//! harness's [`ForkSession`] then copies the session as it was before the
//! first message after the range, so every saved value of a plugin comes
//! along as of the end of the range and the copy records its parent
//! ([`ForkedFrom`]). In the copy the messages before the range become one
//! context message; in the original the range becomes one note, a message
//! by this plugin the model reads too, followed by an aside holding the
//! moved entries for `/split undo`. Each session's `/resume` line says
//! "split from" or "split off" ([`ResumeLabel`]).

use std::borrow::Cow;
use std::collections::BTreeMap;

use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::pending_calls;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_memory::{SessionMeta, Summarizer, SummaryLimits, SummaryPrompts, title_of};
use rig_sessions::{NameSession, ResumeLabel, SessionsPlugin};
use serde::{Deserialize, Serialize};

/// Who this plugin is, as an author and a decider, and the purpose of its
/// model calls in usage.
pub const NAME: &str = "split-session";

/// The kind of the aside after a split's note that keeps what it moved.
pub const MOVED: &str = "split-session.moved";

/// The file, in a split-off session's plugin data, naming where it came
/// from.
const FROM_FILE: &str = "split-session/from.json";

/// The same file as the session's store names it ([`PluginData::session`]
/// is the store's `plugins/`).
const FROM_IN_STORE: &str = "plugins/split-session/from.json";

/// The tag of this plugin's `/resume` label.
const LABEL: &str = "split";

/// What the summarizing model is told.
const SYSTEM: &str = "You summarize part of a coding conversation for a reader who has not seen \
    it. Do not continue the conversation. Output only the summary.";

/// rig-memory's summarizer with this plugin's prompts.
const fn summarizer(ask: &'static str, format: &'static str) -> Summarizer {
    Summarizer {
        prompts: SummaryPrompts {
            system: Cow::Borrowed(SYSTEM),
            initial: Cow::Borrowed(ask),
            update: Cow::Borrowed(ask),
            format: Cow::Borrowed(format),
        },
        limits: SummaryLimits {
            summary_tokens: 1_000,
            ..SummaryLimits::DEFAULT
        },
    }
}

/// Condenses what came before the range into the new session's context.
const CONTEXT: Summarizer = summarizer(
    "The turns after this conversation move to a session of their own. Summarize what a reader \
     of those turns needs from it: the goal, decisions, files and names.",
    "\n\nAnswer in at most five short lines.",
);

/// Says what the moved turns concluded, for the note left behind.
const CONCLUSIONS: Summarizer = summarizer(
    "These turns move to a session of their own. Say what they found and concluded.",
    "\n\nStart with one line `Topic: <two to five words>`, then the conclusions in at most three \
     short lines.",
);

/// `/split` and `/split undo`.
#[derive(Default)]
pub struct SplitSessionPlugin;

impl Plugin for SplitSessionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.world_mut()
            .spawn((SplitLabel, ResumeLabel::new(LABEL, "")));
        app.add_setting::<SplitSettings>("split")
            .add_message::<SessionForked>()
            .init_resource::<SplitFrom>()
            .add_command(
                "split",
                "Move turns into a new session: /split <from> [<to>] [--name <name>], or /split undo",
                split,
            )
            .add_observer(undo)
            .add_observer(summarized)
            .add_observer(wake_pending)
            .add_observer(follow_successor::<SplitCall>)
            .add_systems(Startup, read_from)
            .add_systems(Update, (begin.before(AgentLoop), label))
            .add_systems(AgentPass, hold_messages.in_set(Stage::Gate).after(AskSystems))
            .add_systems(Last, finish.after(ForkSessions));
    }
}

/// The `[split]` settings table: `model = "vendor/model"` summarizes on
/// that (cheap) catalog model rather than the agent's own.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct SplitSettings {
    /// The summarizing model.
    pub model: Option<String>,
}

/// What `/split` asked for, on its agent until no turn runs.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct PendingSplit {
    /// The first turn moved, from 1.
    pub from: usize,
    /// The last turn moved; `None` for the last turn.
    pub to: Option<usize>,
    /// The new session's name.
    pub name: Option<String>,
}

/// On an agent while a split of its conversation runs: the split.
#[derive(Component, Clone, Copy, Debug)]
pub struct Splitting(pub Entity);

/// A split under way, until its new session is written.
#[derive(Component, Clone, Debug)]
struct Split {
    agent: Entity,
    name: Option<String>,
    /// The turns moved, as numbered when the split began.
    turns: (usize, usize),
    /// The first message moved.
    first: MessageId,
    /// The first message after the range; `None` when it runs to the end.
    end: Option<MessageId>,
    context: Option<String>,
    conclusions: Option<String>,
    forking: bool,
}

/// Which summary a call of a split writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Context,
    Conclusions,
}

/// On a split's summary call (and on a successor sending it again).
#[derive(Component, Clone, Copy, Debug)]
pub struct SplitCall {
    split: Entity,
    part: Part,
}

/// A hold on a message to an agent being split.
#[derive(Component)]
struct SplitHold(Entity);

/// On a message a split holds: held once.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct HeldForSplit;

/// The entity of this plugin's `/resume` label.
#[derive(Component)]
struct SplitLabel;

/// Where this session was split from, if it was.
#[derive(Resource, Default, Clone, Debug, PartialEq, Eq)]
struct SplitFrom(Option<Origin>);

/// A split-off session's record of its origin.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct Origin {
    /// The session it was split from.
    session: String,
    /// The turns it took, as numbered there.
    turns: (usize, usize),
}

/// What a split moved, kept in an aside after its note in the original.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Moved {
    /// The new session.
    session: String,
    /// Its name.
    name: String,
    /// The note that replaced the turns.
    note: MessageId,
    /// The entries moved, as they were.
    entries: Vec<Entry>,
    /// This conversation's revision right after the split.
    revision: u64,
    /// The new session's main conversation's revision once written.
    copied: u64,
}

/// Asks for `/split undo` on an agent.
#[derive(Event, Clone, Copy, Debug)]
pub struct UndoSplit {
    /// The agent.
    pub agent: Entity,
}

/// What a `/split` line says.
fn parse(args: &str) -> Result<PendingSplit, String> {
    let mut words = args.split_whitespace();
    let mut numbers = Vec::new();
    let mut name = None;
    while let Some(word) = words.next() {
        if word == "--name" {
            let rest: Vec<&str> = words.by_ref().collect();
            name = Some(rest.join(" ")).filter(|name| !name.is_empty());
            if name.is_none() {
                return Err("--name needs a name".to_owned());
            }
        } else {
            let turn = word
                .parse::<usize>()
                .ok()
                .filter(|turn| *turn > 0)
                .ok_or_else(|| format!("`{word}` is no turn number"))?;
            numbers.push(turn);
        }
    }
    match numbers.as_slice() {
        [from] => Ok(PendingSplit {
            from: *from,
            to: None,
            name,
        }),
        [from, to] if from <= to => Ok(PendingSplit {
            from: *from,
            to: Some(*to),
            name,
        }),
        [_, _] => Err("the first turn comes after the last".to_owned()),
        _ => Err("usage: /split <from> [<to>] [--name <name>], or /split undo".to_owned()),
    }
}

/// `/split …`: waits for a running turn, or begins on the next frame.
fn split(
    In(args): In<CommandArgs>,
    agents: Query<(Has<ActiveTurn>, Has<Splitting>)>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if args.args.trim() == "undo" {
        commands.trigger(UndoSplit { agent });
        return;
    }
    let pending = match parse(&args.args) {
        Ok(pending) => pending,
        Err(why) => {
            notices.write(Notice::error(agent, format!("/split: {why}.")));
            return;
        }
    };
    let Ok((busy, splitting)) = agents.get(agent) else {
        return;
    };
    if splitting {
        notices.write(Notice::error(agent, "A split is running already."));
        return;
    }
    if busy {
        notices.write(Notice::info(agent, "Splitting once the turn ends."));
    }
    commands.entity(agent).insert(pending);
    wake.wake();
}

/// Wakes the loop when a turn of an agent with a split waiting ends, so
/// the split begins without waiting for other work.
fn wake_pending(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<TurnOutcome>>,
    pending: Query<(), With<PendingSplit>>,
    wake: Res<Wake>,
) {
    if turns
        .get(settled.entity)
        .is_ok_and(|&ForAgent(agent)| pending.contains(agent))
    {
        wake.wake();
    }
}

/// The turns of `conversation` whose first message is still in it: each
/// one's start, oldest first.
fn turn_starts(conversation: &Conversation) -> Vec<usize> {
    let mut starts: Vec<usize> = conversation
        .turns()
        .filter_map(|(_, record)| conversation.index_of(record.start))
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Whether a cut before message `at` leaves every tool call with its
/// result on its side.
fn whole(messages: &[Message], at: usize) -> bool {
    messages
        .get(..at)
        .is_some_and(|before| pending_calls(before).is_empty())
}

/// `at`, else the turn start (or the end) nearest it that cuts no tool
/// exchange, the earlier on a tie, within `allowed`.
fn nearest_whole(
    messages: &[Message],
    starts: &[usize],
    at: usize,
    allowed: impl Fn(usize) -> bool,
) -> Option<usize> {
    if whole(messages, at) {
        return Some(at);
    }
    starts
        .iter()
        .copied()
        .chain([messages.len()])
        .filter(|&candidate| allowed(candidate) && whole(messages, candidate))
        .min_by_key(|&candidate| (candidate.abs_diff(at), candidate))
}

/// The turn number (from 1) that begins at `at`, or the last before it.
fn turn_at(starts: &[usize], at: usize) -> usize {
    starts.iter().filter(|&&start| start <= at).count().max(1)
}

/// The messages `pending` moves of `conversation`, `from..to`, with
/// whole tool exchanges, and what the user is told about a boundary moved.
fn range(
    conversation: &Conversation,
    pending: &PendingSplit,
) -> Result<(usize, usize, Option<String>), String> {
    let starts = turn_starts(conversation);
    let messages = conversation.messages();
    let count = starts.len();
    let last = pending.to.unwrap_or(count);
    if pending.from > count || last > count {
        return Err(format!("there are {count} turns"));
    }
    let start = starts
        .get(pending.from.saturating_sub(1))
        .copied()
        .ok_or("no such turn")?;
    let end = starts.get(last).copied().unwrap_or(messages.len());
    let from = nearest_whole(messages, &starts, start, |at| at < end)
        .ok_or("no whole turn starts the range")?;
    let to = nearest_whole(messages, &starts, end, |at| at > from)
        .ok_or("no whole turn ends the range")?;
    let moved = (from != start || to != end).then(|| {
        let first = turn_at(&starts, from);
        let last = if to >= messages.len() {
            count
        } else {
            turn_at(&starts, to).saturating_sub(1).max(first)
        };
        format!(
            "A boundary fell inside a tool exchange, so the split moves whole turns {first}–{last}."
        )
    });
    Ok((from, to, moved))
}

/// A summary call of `split` over `messages`, on the `[split]` model or
/// the agent's own.
fn summary_call(
    (split, part): (Entity, Part),
    messages: &[Message],
    own: Option<&Connection>,
    (settings, models): (&SplitSettings, &Models),
) -> Result<impl Bundle, String> {
    let chosen = match settings.model.as_deref() {
        Some(model) => Some(models.connect(model).map_err(|why| why.to_string())?),
        None => None,
    };
    let spec = chosen
        .as_ref()
        .or(own)
        .map(|connection| connection.spec.clone())
        .ok_or("the agent has no model")?;
    let summarizer = match part {
        Part::Context => &CONTEXT,
        Part::Conclusions => &CONCLUSIONS,
    };
    let request = summarizer
        .request(messages, "", "", &spec)
        .map_err(|why| why.to_string())?;
    let request = match chosen {
        Some(connection) => ModelRequest::new(request).on(connection),
        None => ModelRequest::new(request),
    };
    Ok((
        Name::new("split summary"),
        SplitCall { split, part },
        request,
        Effect::default().with_purpose(NAME),
    ))
}

/// Begins each split asked for of an agent with no turn running: its
/// range, then its first summary call.
fn begin(
    agents: Query<
        (Entity, &PendingSplit, &Conversation, Option<&Connection>),
        (Without<ActiveTurn>, Without<Splitting>),
    >,
    (settings, models): (Res<SplitSettings>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, pending, conversation, own) in &agents {
        commands.entity(agent).remove::<PendingSplit>();
        let (from, to, moved) = match range(conversation, pending) {
            Ok(range) => range,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot split: {why}.")));
                continue;
            }
        };
        if let Some(moved) = moved {
            notices.write(Notice::info(agent, moved));
        }
        let (Some(first), end) = (conversation.id(from), conversation.id(to)) else {
            continue;
        };
        let starts = turn_starts(conversation);
        let last = if to >= conversation.messages().len() {
            starts.len()
        } else {
            turn_at(&starts, to).saturating_sub(1)
        };
        let split = commands.spawn_empty().id();
        let (part, messages) = match from {
            0 => (Part::Conclusions, conversation.messages().get(from..to)),
            _ => (Part::Context, conversation.messages().get(..from)),
        };
        let call = summary_call(
            (split, part),
            messages.unwrap_or_default(),
            own,
            (&settings, &models),
        );
        let call = match call {
            Ok(call) => call,
            Err(why) => {
                commands.entity(split).despawn();
                notices.write(Notice::error(agent, format!("Cannot split: {why}.")));
                continue;
            }
        };
        commands.spawn((call, ForAgent(agent)));
        commands.entity(split).insert(Split {
            agent,
            name: pending.name.clone(),
            turns: (turn_at(&starts, from), last.max(1)),
            first,
            end,
            context: (from == 0).then(String::new),
            conclusions: None,
            forking: false,
        });
        commands.entity(agent).insert(Splitting(split));
    }
}

/// Holds each new message to an agent being split until the split ends,
/// after the lines a command or a question takes.
fn hold_messages(
    messages: Query<
        (Entity, &ForAgent, &Decision),
        (Undecided, IncomingMessages, Without<HeldForSplit>),
    >,
    agents: Query<&Splitting>,
    mut commands: Commands,
) {
    for (message, &ForAgent(agent), decision) in &messages {
        if decision.is_decided() {
            continue;
        }
        if let Ok(&Splitting(split)) = agents.get(agent) {
            commands.spawn((
                Name::new("waiting for /split"),
                HoldOf(message),
                SplitHold(split),
            ));
            commands.entity(message).insert(HeldForSplit);
        }
    }
}

/// Ends `split` of `agent`: lets its held messages go.
fn end_split(
    commands: &mut Commands,
    holds: &Query<(Entity, &SplitHold)>,
    split: Entity,
    agent: Entity,
) {
    for (hold, _) in holds.iter().filter(|(_, hold)| hold.0 == split) {
        commands.entity(hold).despawn();
    }
    commands.entity(split).despawn();
    commands.entity(agent).remove::<Splitting>();
}

/// The text of a settled summary call.
fn summary_of(outcome: &Outcome) -> Result<String, String> {
    match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(response)) => {
            Summarizer::summary_text(response).map_err(|why| why.to_string())
        }
    }
}

/// Takes each settled summary: after the context, asks for the
/// conclusions; after both, asks the harness to copy the session as of
/// the end of the range.
fn summarized(
    settled: On<Add<Settled>>,
    calls: Query<(&SplitCall, &Outcome)>,
    mut splits: Query<&mut Split>,
    agents: Query<(&AgentId, &Conversation, Option<&Connection>)>,
    holds: Query<(Entity, &SplitHold)>,
    (settings, models): (Res<SplitSettings>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&SplitCall { split, part }, outcome)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut state) = splits.get_mut(split) else {
        return;
    };
    let agent = state.agent;
    let fail = |why: String, commands: &mut Commands, notices: &mut MessageWriter<Notice>| {
        notices.write(Notice::error(agent, format!("Split failed: {why}.")));
        end_split(commands, &holds, split, agent);
    };
    let text = match summary_of(outcome) {
        Ok(text) => text,
        Err(why) => return fail(why, &mut commands, &mut notices),
    };
    let Ok((id, conversation, own)) = agents.get(agent) else {
        return;
    };
    match part {
        Part::Context => {
            state.context = Some(text);
            let from = conversation.index_of(state.first);
            let to = state
                .end
                .and_then(|end| conversation.index_of(end))
                .unwrap_or(conversation.messages().len());
            let moved = from.and_then(|from| conversation.messages().get(from..to));
            let Some(moved) = moved else {
                return fail("its turns changed".to_owned(), &mut commands, &mut notices);
            };
            match summary_call((split, Part::Conclusions), moved, own, (&settings, &models)) {
                Ok(call) => {
                    commands.spawn((call, ForAgent(agent)));
                }
                Err(why) => fail(why, &mut commands, &mut notices),
            }
        }
        Part::Conclusions => {
            state.conclusions = Some(text);
            state.forking = true;
            commands.trigger(ForkSession {
                from: None,
                cut: state
                    .end
                    .map_or(ForkCut::Whole, |end| ForkCut::Before(id.clone(), end)),
                asker: Some(split),
            });
        }
    }
}

/// The topic and conclusions a conclusions summary gives.
fn topic_and_conclusions(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();
    let (first, rest) = trimmed.split_once('\n').unwrap_or((trimmed, ""));
    match first.trim().strip_prefix("Topic:") {
        Some(topic) => (Some(topic.trim().to_owned()), rest.trim().to_owned()),
        None => (None, trimmed.to_owned()),
    }
}

/// A session name from a topic.
fn slug(topic: &str) -> String {
    let words: Vec<String> = topic
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() {
        "split".to_owned()
    } else {
        words.join("-")
    }
}

/// Writes the new session `to` from the copy the harness made: the
/// messages before the range become one context message. Its main
/// conversation's revision once written.
fn write_copy(
    (stores, home): (&SessionStores, &Home),
    (from, to): (&SessionId, &SessionId),
    agent: &AgentId,
    split: &Split,
    name: &str,
) -> Result<u64, String> {
    let store = stores
        .open(&home.session(to))
        .map_err(|failure| failure.to_string())?;
    let mut session = Session::read(&*store).map_err(|failure| failure.to_string())?;
    let saved = session
        .agent_mut(&agent.0)
        .ok_or("the copy lacks the agent")?;
    let conversation = &mut saved.conversation;
    let first = conversation
        .index_of(split.first)
        .ok_or("the copy lacks the moved turns")?;
    let context = split.context.clone().unwrap_or_default();
    let context = (first > 0).then(|| {
        let text = format!(
            "[Context from session `{from}` before turn {}, condensed]\n{context}",
            split.turns.0
        );
        Entry::new(Message::user(text)).by(Author::plugin(NAME))
    });
    conversation.splice(..first, context, Author::plugin(NAME));
    session
        .save(&*store)
        .map_err(|failure| failure.to_string())?;
    let copied = session
        .agent(&agent.0)
        .map(|saved| saved.conversation.revision())
        .unwrap_or_default();
    let origin = Origin {
        session: from.to_string(),
        turns: split.turns,
    };
    let bytes = serde_json::to_vec(&origin).map_err(|failure| failure.to_string())?;
    store
        .write_file(FROM_IN_STORE, &bytes)
        .map_err(|failure| failure.to_string())?;
    let labels = BTreeMap::from([(LABEL.to_owned(), format!("split from {from}"))]);
    let meta = SessionMeta {
        title: name.to_owned(),
        labels,
        updated: now_ms(),
        ..SessionMeta::default()
    };
    meta.write(&*store).map_err(|failure| failure.to_string())?;
    Ok(copied)
}

/// Replaces the moved range of `conversation` with one note, followed by
/// the aside that keeps what it moved.
fn leave_note(
    conversation: &mut Conversation,
    split: &Split,
    (to, name, copied): (&SessionId, &str, u64),
) -> Result<(), String> {
    let from = conversation
        .index_of(split.first)
        .ok_or("its turns changed")?;
    let end = split
        .end
        .and_then(|end| conversation.index_of(end))
        .unwrap_or(conversation.messages().len());
    let entries: Vec<Entry> = (from..end)
        .filter_map(|at| conversation.entry(at))
        .collect();
    let (topic, conclusions) =
        topic_and_conclusions(split.conclusions.as_deref().unwrap_or_default());
    let topic = topic.unwrap_or_else(|| name.to_owned());
    let (a, b) = split.turns;
    let turns = if a == b {
        format!("Turn {a}")
    } else {
        format!("Turns {a}–{b}")
    };
    let text =
        format!("[{turns} ({topic}) were moved to session `{name}`.]\nConclusions:\n{conclusions}");
    let note = Entry::new(Message::user(text)).by(Author::plugin(NAME));
    conversation.splice(from..end, [note], Author::plugin(NAME));
    let note = conversation.id(from).ok_or("the note went missing")?;
    let moved = Moved {
        session: to.to_string(),
        name: name.to_owned(),
        note,
        entries,
        revision: conversation.revision().saturating_add(1),
        copied,
    };
    let value = serde_json::to_value(&moved).map_err(|failure| failure.to_string())?;
    conversation.add_aside(Aside::new(MOVED, Author::plugin(NAME), value).after(Some(note)));
    Ok(())
}

/// Finishes each split whose copy the harness wrote this frame: writes
/// the new session and leaves the note behind; a split whose copy failed
/// ends with the harness's error.
fn finish(
    mut forked: MessageReader<SessionForked>,
    splits: Query<(Entity, &Split)>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    holds: Query<(Entity, &SplitHold)>,
    (stores, home, wake): (Res<SessionStores>, Res<RigHome>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let forks: Vec<SessionForked> = forked.read().cloned().collect();
    for (entity, split) in splits.iter().filter(|(_, split)| split.forking) {
        let agent = split.agent;
        let fork = forks.iter().find(|fork| fork.asker == Some(entity));
        let done = fork.ok_or_else(|| "the session could not be copied".to_owned());
        let done = done.and_then(|fork| {
            let (id, mut conversation) = agents.get_mut(agent).map_err(|_| "no agent")?;
            let (topic, _) =
                topic_and_conclusions(split.conclusions.as_deref().unwrap_or_default());
            let name = split
                .name
                .clone()
                .unwrap_or_else(|| slug(topic.as_deref().unwrap_or("split")));
            let written = write_copy((&stores, &home.0), (&fork.from, &fork.to), id, split, &name)
                .and_then(|copied| leave_note(&mut conversation, split, (&fork.to, &name, copied)));
            if let Err(why) = written {
                commands.trigger(DeleteSession(fork.to.clone()));
                return Err(why);
            }
            commands.trigger(NameSession {
                session: fork.to.clone(),
                name: name.clone(),
            });
            Ok(format!(
                "Moved turns {}–{} to session `{name}`; /split undo puts them back.",
                split.turns.0, split.turns.1
            ))
        });
        match done {
            Ok(text) => notices.write(Notice::info(agent, text)),
            Err(why) => notices.write(Notice::error(agent, format!("Split failed: {why}."))),
        };
        end_split(&mut commands, &holds, entity, agent);
        wake.wake();
    }
}

/// `/split undo`: puts the last split's turns back in place of its note
/// and removes its session, while neither changed since.
fn undo(
    asked: On<UndoSplit>,
    mut agents: Query<(&AgentId, &mut Conversation), Without<ActiveTurn>>,
    (stores, home): (Res<SessionStores>, Res<RigHome>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = asked.agent;
    let Ok((id, mut conversation)) = agents.get_mut(agent) else {
        notices.write(Notice::turn_running(agent));
        return;
    };
    let last = conversation
        .asides()
        .rev()
        .filter(|(_, aside)| aside.kind == MOVED)
        .filter_map(|(aside, value)| Some((aside, Moved::deserialize(&value.value).ok()?)))
        .find(|(_, moved)| conversation.index_of(moved.note).is_some());
    let Some((aside, moved)) = last else {
        notices.write(Notice::error(agent, "No split to undo."));
        return;
    };
    let unchanged = |moved: &Moved| -> Result<SessionId, String> {
        if conversation.revision() != moved.revision {
            return Err("this session changed since".to_owned());
        }
        let session: SessionId = moved.session.parse().map_err(|_| "a bad session id")?;
        let dir = home.0.session(&session);
        if dir.runs_elsewhere() {
            return Err(format!("session `{}` is running", moved.name));
        }
        let store = stores.open(&dir).map_err(|failure| failure.to_string())?;
        let copy = Session::read(&*store).map_err(|failure| failure.to_string())?;
        let revision = copy.agent(&id.0).map(|saved| saved.conversation.revision());
        if revision != Some(moved.copied) {
            return Err(format!("session `{}` changed since", moved.name));
        }
        Ok(session)
    };
    let session = match unchanged(&moved) {
        Ok(session) => session,
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Cannot undo the split: {why}."),
            ));
            return;
        }
    };
    let Some(at) = conversation.index_of(moved.note) else {
        notices.write(Notice::error(
            agent,
            "Cannot undo the split: its note is gone.",
        ));
        return;
    };
    conversation.remove_aside(aside);
    conversation.restore(at..=at, moved.entries, Author::plugin(NAME));
    commands.trigger(DeleteSession(session));
    notices.write(Notice::info(
        agent,
        format!("Put the turns back; session `{}` is removed.", moved.name),
    ));
}

/// Reads where this session was split from, if it was.
fn read_from(data: Option<Res<PluginData>>, mut from: ResMut<SplitFrom>) {
    let read = data
        .and_then(|data| data.session.read_file(FROM_FILE).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    from.0 = read;
}

/// Keeps this session's `/resume` label: where it was split from and the
/// sessions split off it.
fn label(
    from: Res<SplitFrom>,
    conversations: Query<Ref<Conversation>>,
    mut labels: Query<&mut ResumeLabel, With<SplitLabel>>,
) {
    if !from.is_changed()
        && !conversations
            .iter()
            .any(|conversation| conversation.is_changed())
    {
        return;
    }
    let mut parts = Vec::new();
    if let Some(from) = &from.0 {
        parts.push(format!("split from {}", from.session));
    }
    let off: Vec<String> = conversations
        .iter()
        .flat_map(|conversation| {
            conversation
                .asides()
                .filter(|(_, aside)| aside.kind == MOVED)
                .filter_map(|(_, aside)| Moved::deserialize(&aside.value).ok())
                .filter(|moved| conversation.index_of(moved.note).is_some())
                .map(|moved| moved.name)
                .collect::<Vec<_>>()
        })
        .collect();
    if !off.is_empty() {
        parts.push(format!("split off {}", off.join(", ")));
    }
    let text = title_of(&parts.join("; "));
    for mut label in &mut labels {
        label.set_if_neq(ResumeLabel::new(LABEL, text.clone()));
    }
}

#[cfg(test)]
mod tests;
