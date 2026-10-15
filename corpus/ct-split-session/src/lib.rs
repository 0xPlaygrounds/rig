//! CT-10: split a tangent into its own session (conversation-over-time
//! fresh held-out spec).
//!
//! `/split <from-turn> [<to-turn>] [--name <name>]` moves a range of the
//! agent's turns (to the end by default) into a new saved session of the
//! same project, and `/split undo` puts the last split's turns back while
//! neither session changed since. It takes the `/split` name from the
//! default rig-splits plugin.
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
//! harness's [`ForkSession`] with a [`ForkCut::Line`] move then copies the
//! agent's line through the range into a new session, takes the range out
//! of this one and leaves a split pointer where it was; the copy records
//! its parent, so session listings say "split from", and values that
//! follow the conversation come along as of the end of the range. In the
//! copy the messages before the range become one context message; in the
//! original a note by this plugin, which the model reads too, joins the
//! pointer. `/split undo` merges the copy's turns back at the pointer
//! ([`Conversation::merge_line`](Line::merge_line)) and deletes the copy.

use std::borrow::Cow;

use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::pending_calls;
use rig_harness::rig_core::conversation::{Conversation as Line, MergeAt};
use rig_harness::rig_core::message::Message;
use rig_harness::rig_memory::{SPLIT, SplitPointer, Summarizer, SummaryLimits, SummaryPrompts};
use rig_sessions::{NameSession, SessionsPlugin};
use serde::{Deserialize, Serialize};

/// Who this plugin is, as an author and a decider, and the purpose of its
/// model calls in usage.
pub const NAME: &str = "split-session";

/// The kind of the aside after a split's note that keeps what `/split
/// undo` checks.
pub const MOVED: &str = "split-session.moved";

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
        let world = app.world_mut();
        let taken: Vec<Entity> = world
            .query_filtered::<(Entity, &Name), With<SlashCommand>>()
            .iter(world)
            .filter(|(_, name)| name.as_str() == "/split")
            .map(|(entity, _)| entity)
            .collect();
        for entity in taken {
            world.entity_mut(entity).insert(Disabled);
        }
        app.add_setting::<SplitSettings>("split")
            .add_message::<SessionForked>()
            .add_message::<SessionForkFailed>()
            .add_command(
                "split",
                "Move turns into a new session: /split <from> [<to>] [--name <name>], or /split undo",
                split,
            )
            .add_observer(undo)
            .add_observer(summarized)
            .add_observer(wake_pending)
            .add_observer(follow_successor::<SplitCall>)
            .add_systems(Update, begin.before(AgentLoop))
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
    /// The last message moved.
    last: MessageId,
    context: Option<String>,
    conclusions: Option<String>,
    /// The new session's name, once the harness was asked to copy.
    forking: Option<String>,
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

/// What `/split undo` checks, kept in an aside after a split's note.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Moved {
    /// The new session.
    session: String,
    /// Its name.
    name: String,
    /// The note left in the moved turns' place.
    note: MessageId,
    /// Whether the copy begins with a context message of this plugin.
    context: bool,
    /// This conversation's revision right after the split.
    revision: u64,
    /// The copy's conversation's revision once written.
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

/// The messages `pending` moves of `conversation`, `from..to`, with whole
/// tool exchanges; the turns they are; what the user is told about a
/// boundary moved.
fn range(
    conversation: &Conversation,
    pending: &PendingSplit,
) -> Result<(usize, usize, (usize, usize), Option<String>), String> {
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
    let first = turn_at(&starts, from);
    let last = if to >= messages.len() {
        count
    } else {
        turn_at(&starts, to).saturating_sub(1).max(first)
    };
    let moved = (from != start || to != end).then(|| {
        format!(
            "A boundary fell inside a tool exchange, so the split moves whole turns {first}–{last}."
        )
    });
    Ok((from, to, (first, last), moved))
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
        let (from, to, turns, moved) = match range(conversation, pending) {
            Ok(range) => range,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot split: {why}.")));
                continue;
            }
        };
        if let Some(moved) = moved {
            notices.write(Notice::info(agent, moved));
        }
        let ids = (
            conversation.id(from),
            to.checked_sub(1).and_then(|last| conversation.id(last)),
        );
        let (Some(first), Some(last)) = ids else {
            continue;
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
            turns,
            first,
            last,
            context: None,
            conclusions: None,
            forking: None,
        });
        commands.entity(agent).insert(Splitting(split));
    }
}

/// Holds each new message to an agent being split until the split ends,
/// after the lines a command or a question takes.
fn hold_messages(
    messages: Query<
        (Entity, &ForAgent, &Decision),
        (ReadyToStart, IncomingMessages, Without<HeldForSplit>),
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
/// conclusions; after both, asks the harness to move the range into a new
/// session.
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
            let to = conversation.index_of(state.last);
            let moved = from
                .zip(to)
                .and_then(|(from, to)| conversation.messages().get(from..=to));
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
            let (topic, _) = topic_and_conclusions(&text);
            let name = state
                .name
                .clone()
                .unwrap_or_else(|| slug(topic.as_deref().unwrap_or("split")));
            state.conclusions = Some(text);
            state.forking = Some(name.clone());
            commands.write_message(ForkSession {
                from: None,
                cut: ForkCut::Line {
                    agent: id.clone(),
                    line: None,
                    through: state.last,
                    move_from: Some(state.first),
                    title: name,
                },
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

/// The split pointer in `conversation` to the session `to`, not merged
/// yet, and how many entries it took out.
fn pointer_to(conversation: &Line, to: &SessionId) -> Option<(AsideId, usize)> {
    let to = to.to_string();
    conversation
        .asides()
        .filter(|(_, aside)| aside.kind == SPLIT)
        .find_map(|(id, aside)| {
            let pointer = SplitPointer::deserialize(&aside.value).ok()?;
            (pointer.session == to && !pointer.merged).then_some((id, pointer.moved))
        })
}

/// The saved session `to` of `home`: its conversation of `agent`.
fn copy_of(
    (stores, home): (&StoreOpener, &Home),
    to: &SessionId,
    agent: &AgentId,
) -> Result<Line, String> {
    let store = stores
        .open(&home.session(to))
        .map_err(|failure| failure.to_string())?;
    let session = Session::read(&*store).map_err(|failure| failure.to_string())?;
    let saved = session
        .agent(&agent.0)
        .ok_or("the split-off session lacks the agent")?;
    Ok(saved.conversation.clone())
}

/// Puts the turns `copy` (the split session `to`'s conversation, its
/// context message taken off) holds back at their pointer in
/// `conversation`, with their ids, and removes the pointer.
fn put_back(conversation: &mut Line, to: &SessionId, copy: &Line) -> Result<(), String> {
    let (pointer, moved) = pointer_to(conversation, to).ok_or("its pointer is gone")?;
    conversation
        .merge_line(copy, MergeAt::Back { pointer, moved }, Author::plugin(NAME))
        .map_err(|failure| failure.to_string())?;
    conversation.remove_aside(pointer);
    Ok(())
}

/// Turns the messages before the range in the new session `to` into one
/// context message. The copy's revision once written, and whether it
/// begins with that message.
fn write_copy(
    (stores, home, running): (&StoreOpener, &Home, &SessionDir),
    (from, to): (&SessionId, &SessionId),
    agent: &AgentId,
    split: &Split,
) -> Result<(u64, bool), String> {
    let mut written = (0, false);
    edit_stopped(stores, running, &home.session(to), |session| {
        let conversation = &mut session
            .agent_mut(&agent.0)
            .ok_or("the copy lacks the agent")?
            .conversation;
        let first = conversation
            .index_of(split.first)
            .ok_or("the copy lacks the moved turns")?;
        if first > 0 {
            let text = format!(
                "[Context from session `{from}` before turn {}, condensed]\n{}",
                split.turns.0,
                split.context.as_deref().unwrap_or_default()
            );
            let context = Entry::new(Message::user(text)).by(Author::plugin(NAME));
            conversation.splice(..first, [context], Author::plugin(NAME));
        }
        written = (conversation.revision(), first > 0);
        Ok(())
    })?;
    Ok(written)
}

/// Leaves the note at the pointer to `to` in `conversation`, followed by
/// the aside `/split undo` reads.
fn leave_note(
    conversation: &mut Line,
    split: &Split,
    (to, name): (&SessionId, &str),
    (copied, context): (u64, bool),
) -> Result<(), String> {
    let (pointer, _) = pointer_to(conversation, to).ok_or("its pointer is gone")?;
    let at = conversation
        .aside(pointer)
        .map(|pointer| conversation.aside_position(pointer))
        .ok_or("its pointer is gone")?;
    let (topic, conclusions) =
        topic_and_conclusions(split.conclusions.as_deref().unwrap_or_default());
    let topic = topic.unwrap_or_else(|| name.to_owned());
    let turns = match split.turns {
        (a, b) if a == b => format!("Turn {a}"),
        (a, b) => format!("Turns {a}–{b}"),
    };
    let text =
        format!("[{turns} ({topic}) were moved to session `{name}`.]\nConclusions:\n{conclusions}");
    let note = Entry::new(Message::user(text)).by(Author::plugin(NAME));
    conversation.splice(at..at, [note], Author::plugin(NAME));
    let note = conversation.id(at).ok_or("the note went missing")?;
    let moved = Moved {
        session: to.to_string(),
        name: name.to_owned(),
        note,
        context,
        // The aside added next is one more record.
        revision: conversation.revision().saturating_add(1),
        copied,
    };
    let value = serde_json::to_value(&moved).map_err(|failure| failure.to_string())?;
    conversation.add_aside(Aside::new(MOVED, Author::plugin(NAME), value).after(Some(note)));
    Ok(())
}

/// Finishes each split the harness moved (or failed to) this frame:
/// writes the new session's context and leaves the note behind. When that
/// fails, the turns go back and the copy is deleted.
fn finish(
    (mut forked, mut failed): (
        MessageReader<SessionForked>,
        MessageReader<SessionForkFailed>,
    ),
    splits: Query<&Split>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    holds: Query<(Entity, &SplitHold)>,
    (stores, home, paths, wake): (Res<StoreOpener>, Res<RigHome>, Res<SessionPaths>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let forks = forked
        .read()
        .filter_map(|fork| Some((fork.asker?, Ok((fork.from.clone(), fork.to.clone())))));
    let failures = failed
        .read()
        .filter_map(|failure| Some((failure.asker?, Err(failure.why.clone()))));
    let outcomes: Vec<_> = forks.chain(failures).collect();
    for (entity, outcome) in outcomes {
        let Ok(
            split @ Split {
                agent,
                forking: Some(name),
                ..
            },
        ) = splits.get(entity)
        else {
            continue;
        };
        let agent = *agent;
        let done = outcome.and_then(|(from, to)| {
            let (id, mut conversation) = agents.get_mut(agent).map_err(|_| "no agent")?;
            let written = write_copy((&stores, &home.0, &paths), (&from, &to), id, split)
                .and_then(|copied| leave_note(&mut conversation, split, (&to, name), copied));
            if let Err(why) = written {
                let back = copy_of((&stores, &home.0), &to, id)
                    .and_then(|copy| put_back(&mut conversation, &to, &copy));
                return match back {
                    Ok(()) => {
                        commands.trigger(DeleteSession(to));
                        Err(format!("{why}; the turns are back"))
                    }
                    Err(back) => Err(format!("{why}; they stay in session {to}: {back}")),
                };
            }
            commands.trigger(NameSession {
                session: to,
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
    (stores, home): (Res<StoreOpener>, Res<RigHome>),
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
    let unchanged = || -> Result<(SessionId, Line), String> {
        if conversation.revision() != moved.revision {
            return Err("this session changed since".to_owned());
        }
        let session: SessionId = moved.session.parse().map_err(|_| "a bad session id")?;
        if home.0.session(&session).runs_elsewhere() {
            return Err(format!("session `{}` is running", moved.name));
        }
        let mut copy = copy_of((&stores, &home.0), &session, id)?;
        if copy.revision() != moved.copied {
            return Err(format!("session `{}` changed since", moved.name));
        }
        if moved.context {
            copy.splice(..1, [], Author::plugin(NAME));
        }
        Ok((session, copy))
    };
    let back = unchanged().and_then(|(session, copy)| {
        put_back(&mut conversation, &session, &copy)?;
        Ok(session)
    });
    let session = match back {
        Ok(session) => session,
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Cannot undo the split: {why}."),
            ));
            return;
        }
    };
    conversation.remove_aside(aside);
    if let Some(note) = conversation.index_of(moved.note) {
        conversation.splice(note..=note, [], Author::plugin(NAME));
    }
    commands.trigger(DeleteSession(session));
    notices.write(Notice::info(
        agent,
        format!("Put the turns back; session `{}` is removed.", moved.name),
    ));
}

#[cfg(test)]
mod tests;
