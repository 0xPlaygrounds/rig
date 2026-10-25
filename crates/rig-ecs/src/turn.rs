//! The turn loop. A user message spawns a turn entity, [`TurnOf`] its
//! agent; its model and tool calls are effect entities [`CallOf`] the turn.
//!
//! The turn's own model call is a [`TurnCall`]: the kernel builds its
//! [`ModelRequest`] from the agent when it is admitted, plugins edit it in
//! Prepare, and Start fits it to its model and sends it. The reply's tool
//! calls are [`ToolRequest`] effects that run side by side unless their
//! [`Footprint`]s conflict ([`Waiting`]); once all settled, their results go
//! back in call order and the next `TurnCall` is spawned.
//!
//! A failed call gets the [`RETRY_POLICY`]'s [`RetryVerdict`] in Arrive,
//! which Review may edit or act on; Commit then backs off ([`Backoff`]) and
//! retries, or ends the turn, and a call refused before it was sent ends
//! it: the user's unanswered message goes back to the views. Every end is
//! a [`TurnOutcome`] proposed on the turn; [`StopTurn`] ends it at once.

use std::borrow::Cow;
use std::fmt;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_log::{info, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{IoTaskPool, TaskPool};
use bevy_time::DelayedCommandsExt;
use crossbeam_channel::{Receiver, Sender};
use futures::StreamExt;
use rig_core::completion::message::{turn_delivered_no_answer, turn_failure};
use rig_core::completion::{
    self, CompletionRequest, CompletionResponse, GenerationOptions, Message, Section,
    UnsupportedOption,
};
use rig_core::conversation::pairing::close_pending_with;
use rig_core::conversation::{Author, AuthorKind, Halt, TurnEnd, attached_file};
use rig_core::effect::{self, EffectId, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, StopReason, ToolCall, UserContent,
};
use rig_core::now_ms;
use rig_core::serve::{Reply, stream_truncated};
use rig_core::streaming::{Item, Relayed, StreamEvent};
use rig_core::tool::{ToolErrorKind, ToolExecutionError, ToolResult, feedback};
use rig_memory::Anchor;
use web_time::Instant;

use super::agent::{
    ActiveTurn, Agent, AgentId, CacheKey, CallOf, Calls, Conversation, EffectParent, LastUsage,
    Notice, Partial, Spawned, StopTurn, SystemPrompt, ToolAccess, Turn, TurnOf, TurnOutcome,
};
use super::inbox::{DeliveryMode, Inboxes, Unanswered, carry_on, deliver};
use super::model::{Connection, ReasoningSetting};
use super::prompt::{PromptSection, SectionOf};
use super::restore::{Restored, commit_results, mark_settled};
use super::retry::{RetryPolicy, Verdict};
use super::session_log::{ConversationChanged, SessionLog, save_conversation};
#[cfg(doc)]
use super::stage::Stage;
use super::stage::{
    Admitted, Effect, ForAgent, HoldOf, Holds, IfStopped, Outcome, Settled, Started, SupersededBy,
    Waiting, cancel, release,
};
use super::tasks::{Running, Wake};
use super::tools::{Footprint, ToolDef, ToolRequest, offered};

/// The notice when the agent has no model to call.
pub(crate) const NO_MODEL: &str = "No model is connected; pick one first.";

/// How failed model calls are retried: rig-core's default, four retries in
/// a row per turn (a reply resets the count).
pub const RETRY_POLICY: RetryPolicy = RetryPolicy::DEFAULT;

/// A turn's recovery so far: the failed calls retried since its last reply.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct Recovery {
    /// Retries since the last reply.
    pub retries: u32,
}

/// The wait before a failed [`TurnCall`] is sent again, on the hold the
/// kernel puts on it ([`HoldOf`]). Once the wait passed on Bevy's clock, a
/// delayed command despawns the hold and spawns a new call that
/// supersedes the failed one ([`SupersededBy`]), unless the hold is gone by then, as
/// interrupting the turn makes it.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, from_reflect = false)]
pub struct Backoff {
    /// Which retry this wait is for, from 1 to [`RETRY_POLICY`]'s `max_retries`.
    pub attempt: u32,
    /// When the call is sent again.
    #[reflect(ignore)]
    pub until: Instant,
}

impl Backoff {
    /// The wait for retry `attempt`, ending `delay` from now.
    pub fn after(attempt: u32, delay: Duration) -> Self {
        Self {
            attempt,
            until: Instant::now() + delay,
        }
    }

    /// Whole seconds left to wait, rounded up.
    pub fn seconds_left(&self) -> u64 {
        let left = self.until.saturating_duration_since(Instant::now());
        left.as_secs() + u64::from(left.subsec_nanos() > 0)
    }
}

/// A model call: an effect entity whose request is `request`, sent to
/// `connection`, else to the model of the agent it is [`ForAgent`], as its
/// [`Adapters`] change that model. Its [`Outcome`] is rig-core's
/// `Outcome::Completion` (see [`Outcome::completion`]). Prepare may change
/// the request, the connection and the adapters; Start then takes the
/// model it goes to, applies the adapters to it, fits the request
/// to the result (an option the model cannot take is left out, as
/// none) and sends it streamed, recording on the call where it went
/// ([`SentTo`]). A [`TurnCall`] or [`SideCall`] has its
/// request built by the kernel when it is admitted, with its system prompt
/// in parts ([`ModelRequest::prompt`]) and its conversation
/// ([`ModelRequest::history`]) beside it; a plugin's own call, such as a
/// summary, brings its own with [`ModelRequest::new`].
#[derive(Component, Reflect, Clone)]
#[reflect(opaque, Component, Clone)]
#[require(Effect, Adapters)]
pub struct ModelRequest {
    /// The request, shared with the effect from Start on; edit it with
    /// [`Self::request_mut`].
    pub request: Arc<CompletionRequest>,
    /// The model it goes to, when not its agent's, such as a fallback
    /// model; `Models::connect` makes one.
    pub connection: Option<Connection>,
    /// The system prompt of a request the kernel built, in parts, until
    /// Start puts it in front of the request's messages; `None`
    /// for a request a plugin brought whole, and from Start on.
    pub prompt: Option<completion::SystemPrompt>,
    /// The messages of a request the kernel built, until Start puts them
    /// in the request (after the system prompt), in place of what
    /// `request.chat_history` holds: a view of its agent's conversation
    /// without its asides (rig-core's `Conversation::request_view`), whose
    /// edits are never saved and never reach the agent. A Prepare system
    /// edits it as a conversation, so each edit names its author and the
    /// entries it kept keep their ids:
    /// `history.splice(0..0, examples, Author::plugin("few-shot"))`.
    /// Its [`unsaved`](rig_core::conversation::Conversation::unsaved)
    /// changes are who changed the request, and how, until Start; an id
    /// of an entry a plugin added belongs to this request alone. `None`
    /// for a request a plugin brought whole, and from Start on.
    pub history: Option<rig_core::conversation::Conversation>,
}

impl ModelRequest {
    /// A call of `request`, on its agent's model.
    pub fn new(request: impl Into<Arc<CompletionRequest>>) -> Self {
        Self {
            request: request.into(),
            connection: None,
            prompt: None,
            history: None,
        }
    }

    /// The request to edit, such as in Prepare: copied first only while
    /// something else shares it.
    pub fn request_mut(&mut self) -> &mut CompletionRequest {
        Arc::make_mut(&mut self.request)
    }

    /// This call, sent to `connection`.
    #[must_use]
    pub fn on(self, connection: Connection) -> Self {
        Self {
            connection: Some(connection),
            ..self
        }
    }
}

/// On a model call the kernel started: the model it went to, its adapters
/// applied, which its request was fitted to. Its spec says what that model
/// finally takes, such as tools another adapter made it read as text, and
/// its handler is the adapted one. From Start on, the call's
/// [`ModelRequest::request`] is the request as sent to that handler (a
/// layer of the handler may still change what the provider gets). Read it
/// in Review, or once the call settled, to learn what every adapter made
/// of the call rather than guess.
#[derive(Component, Clone)]
pub struct SentTo(pub Connection);

/// On a model call the kernel started, beside [`SentTo`]: the options its
/// request asked for that the model it went to does not take, left out
/// when it was fitted to the model's spec or named by a dry run of its
/// handler (rig-core's `Serve::check`). [`ModelStarts::sent`] gives the
/// same list before Start, such as to warn about an option in Prepare.
#[derive(Component, Clone, Debug, Default)]
pub struct DroppedOptions(pub Vec<UnsupportedOption>);

/// The adapters of a model call, applied to its final [`Connection`] at
/// Start: those placed [`outermost`](Adapter::outermost) last, each group
/// in the order pushed. Its own component beside the [`ModelRequest`], so
/// pushing an adapter and editing the request are different data: push in
/// Prepare, `Query<&mut Adapters, Added<Admitted>>`.
#[derive(Component, Clone, Default, Debug)]
pub struct Adapters(pub Vec<Adapter>);

impl Adapters {
    /// `connection`, as these adapters change it.
    pub fn apply(&self, connection: Connection) -> Connection {
        let placed = |outermost| self.0.iter().filter(move |it| it.outermost == outermost);
        placed(false)
            .chain(placed(true))
            .fold(connection, |connection, adapter| adapter.apply(connection))
    }
}

/// A change to the model a call goes to, in its [`Adapters`]: Start
/// applies it to the call's final [`Connection`], after every Prepare edit,
/// so routing a call (its `connection`) and adapting it commute. It may
/// wrap the handler, such as with a stream transform, and change the spec
/// to say what the wrapped model takes, such as tools it reads as text;
/// the request is fitted to the adapted spec.
#[derive(Clone)]
pub struct Adapter {
    /// What it does, for readers, such as `text-tools`.
    pub name: Cow<'static, str>,
    outermost: bool,
    apply: Arc<dyn Fn(Connection) -> Connection + Send + Sync>,
}

impl Adapter {
    /// The adapter `name`, which changes a connection with `apply`.
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        apply: impl Fn(Connection) -> Connection + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            outermost: false,
            apply: Arc::new(apply),
        }
    }

    /// This adapter, applied after every adapter not placed outermost: it
    /// sees what every other adapter made of the model, such as a local
    /// model's wrap or a recording that replaces the adapted handler.
    #[must_use]
    pub fn outermost(self) -> Self {
        Self {
            outermost: true,
            ..self
        }
    }

    /// `connection`, adapted.
    pub fn apply(&self, connection: Connection) -> Connection {
        (self.apply)(connection)
    }
}

impl fmt::Debug for Adapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Adapter").field(&self.name).finish()
    }
}

/// The turn's own model call, [`CallOf`] its turn: built from its agent
/// when admitted (unless spawned with a [`ModelRequest`]), after the
/// messages sent to the agent meanwhile are delivered, and its reply goes
/// into the agent's conversation and carries the turn on or ends it. Its
/// text streams into its [`Partial`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
#[require(Effect, Partial, Name = Name::new("model call"))]
pub struct TurnCall;

/// A model call whose request the kernel builds from the agent this names
/// when it is admitted, as a turn's request would be (system prompt,
/// conversation, tools, options), and that is not part of any turn, such
/// as a side question or a preview of the context. Prepare then edits it,
/// such as by adding a question; its [`Outcome`] is the plugin's to take.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
#[require(Effect)]
pub struct SideCall(pub Entity);

/// On a model call whose handler failed: what the [`RETRY_POLICY`] policy says,
/// put by the kernel in Arrive with the turn's [`Recovery`]. Review may read
/// it, edit it in place (`Query<&mut RetryVerdict>`), or take the failure over
/// by superseding the call; for a [`TurnCall`], Commit carries out what is
/// left: `Retry` waits a [`Backoff`] and sends a new call, `GaveUp`,
/// `Overflow` and `Final` end the turn and give the user's message back.
/// A call refused before it was sent gets none.
#[derive(Component, Clone, Debug)]
pub struct RetryVerdict(pub Verdict);

/// The feed of a model call's stream, read in [`Stage::Arrive`].
#[derive(Component)]
pub(crate) struct Feed(Receiver<Fed>);

/// What a model call's stream feeds its entity: an item of any kind came
/// at this time, the reply's end too, with its event, if it is one.
pub(crate) struct Fed(web_time::SystemTime, Option<StreamEvent>);

/// An event a model call streamed, any model call, written in
/// [`Stage::Arrive`] in the order it came, before the call's outcome is
/// applied: a part's start (a tool call's name), its fragments of text,
/// reasoning or argument JSON, and its end (a finished tool call, with its
/// id and arguments). The event's part tells parallel tool calls apart.
/// Bevy keeps a message at least two frames, so a reader in `PostUpdate`
/// sees the last events of a reply that settled this frame. A
/// [`TurnCall`]'s text and reasoning also add up in its [`Partial`].
#[derive(Message, Clone, Debug)]
pub struct StreamDelta {
    /// The model call.
    pub effect: Entity,
    /// The agent it is [`ForAgent`], if any.
    pub agent: Option<Entity>,
    /// What streamed.
    pub event: StreamEvent,
}

/// When a model call's stream sent its first and its latest item of any
/// kind, such as text, a tool call or the reply's end, and how many it
/// sent: the time to the first token beside its [`Started`], stalls, and
/// when the reply ended. On the call from its first item until it is
/// despawned.
#[derive(Component, Clone, Copy, Debug)]
pub struct StreamTiming {
    /// The wall-clock time of the first item.
    pub first: web_time::SystemTime,
    /// The wall-clock time of the latest item; the reply's end once it
    /// ended.
    pub last: web_time::SystemTime,
    /// How many items came, the end included.
    pub items: u64,
}

/// On a turn one of whose model calls streamed an event: when the first
/// did, the turn's time to its first token. A reply that only ended
/// streamed none.
#[derive(Component, Clone, Copy, Debug)]
pub struct FirstStreamed(pub web_time::SystemTime);

/// Model calls wait on the network: they run on the IO pool.
pub(crate) fn model_pool() -> &'static IoTaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Spawns the next model call of `turn`, for `agent`.
pub(crate) fn spawn_turn_call(commands: &mut Commands, turn: Entity, agent: Entity) {
    commands.spawn((TurnCall, CallOf(turn), ForAgent(agent)));
}

/// Fills in a new turn's [`Turn`] from a turn record it opens in its
/// agent's conversation. A [`Restored`] turn is the agent's last turn
/// carried on, with that turn's record while it is open. The turn is
/// [`ForAgent`] its agent from the start, so it is part of the agent's
/// work and its end inserts nothing but [`Settled`].
pub(crate) fn start_turn(mut world: DeferredWorld, context: HookContext) {
    let Some(&TurnOf(agent)) = world.get::<TurnOf>(context.entity) else {
        return;
    };
    world
        .commands()
        .entity(context.entity)
        .try_insert(ForAgent(agent));
    let restored = world.get::<Restored>(context.entity).is_some();
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
        return;
    };
    let last = conversation.turns().next_back().map(|(_, record)| record);
    let record = match last.filter(|record| restored && record.end.is_none()) {
        Some(record) => record,
        None => conversation.open_turn(),
    };
    if let Some(mut turn) = world.get_mut::<Turn>(context.entity) {
        turn.index = record.index;
        turn.start = record.start;
    }
}

/// How `outcome` ends a turn in its record, the answer being the last
/// entry of `conversation`.
fn turn_end(outcome: &TurnOutcome, conversation: &Conversation) -> TurnEnd {
    match outcome {
        TurnOutcome::Answered(_) => {
            let last = conversation.entries().next_back();
            let answer = last.filter(|entry| matches!(entry.message, Message::Assistant(_)));
            TurnEnd::Answered {
                answer: answer.map(|entry| entry.id),
            }
        }
        TurnOutcome::Failed(why) => TurnEnd::Failed { why: why.clone() },
        TurnOutcome::Stopped(why) => TurnEnd::Stopped { why: why.clone() },
    }
}

/// Fails `turn` for `why`, unless the turn already ends some other way,
/// also by a command queued before this one, such as a stop.
pub(crate) fn fail(commands: &mut Commands, turn: Entity, why: String) {
    commands
        .entity(turn)
        .queue_silenced(move |mut turn: EntityWorldMut| {
            if !turn.contains::<TurnOutcome>() {
                turn.insert(TurnOutcome::Failed(why));
            }
        });
}

/// The error a turn that failed for `why` shows: the reason as a sentence.
fn failure_notice(why: &str) -> String {
    let why = why.strip_suffix(['.', '?', '!']).unwrap_or(why);
    let mut chars = why.chars();
    let first = chars.next().map(char::to_uppercase);
    let why: String = first.into_iter().flatten().chain(chars).collect();
    format!("{why}.")
}

/// What a failed turn tells the user to do next when no text of theirs
/// goes back to the views, `taken` saying whether a message of theirs was
/// taken out ([`take_unanswered`]). A view that puts a text back says so.
fn way_on(taken: bool) -> &'static str {
    if taken {
        "Your message was taken out of the conversation: send it again, or switch models first."
    } else {
        "Send a message to go on, or switch models first."
    }
}

/// Takes the users' messages no model answered out of `conversation`, in
/// one splice, and returns the text of each of the local user's own
/// ([`EntryRef::is_users_own`](rig_core::conversation::EntryRef::is_users_own))
/// that has one, in the order they were sent; `None` when no message was
/// taken. They follow the last reply that counts, or the note that the
/// user stopped a request; failed replies among them go too, and the
/// results that reply asked for stay, wherever they are. A titled user's message, such as a chat bridge's or a skill's,
/// goes without coming back; an agent's or a plugin's keeps its place. A
/// text is what the conversation holds, after Prepare, its text blocks
/// joined; what is not text, such as an image or a file attached, does
/// not come back.
fn take_unanswered(conversation: &mut Conversation) -> Option<Vec<String>> {
    // A reply that failed, or that the token limit cut off before it
    // answered, which a restored session may hold.
    let failed = |message: &Message| {
        matches!(message, Message::Assistant(reply) if match &reply.stop {
            Some(StopReason::Length) => turn_delivered_no_answer(&reply.content),
            stop => stop.as_ref().is_some_and(StopReason::is_failure),
        })
    };
    let settled = conversation.entries().rev().find(|entry| {
        entry.is_stop_marker()
            || matches!(entry.message, Message::Assistant(_)) && !failed(entry.message)
    });
    let from = settled.as_ref().map_or(0, |settled| settled.index + 1);
    // The results that reply asked for stay with it, wherever they are.
    let owed: Vec<&CallId> = match settled.map(|settled| settled.message) {
        Some(Message::Assistant(reply)) => reply.tool_calls().map(|call| &call.id).collect(),
        _ => Vec::new(),
    };
    let (mut taken, mut texts, mut kept) = (false, Vec::new(), Vec::new());
    for entry in conversation.entries().skip(from) {
        let users = entry
            .author
            .is_none_or(|author| author.kind == AuthorKind::User);
        let (Message::User { content }, true) = (entry.message, users) else {
            if !failed(entry.message) {
                kept.extend(conversation.entry(entry.index));
            }
            continue;
        };
        let (results, asked): (Vec<&UserContent>, Vec<&UserContent>) = content
            .iter()
            .partition(|item| matches!(item, UserContent::ToolResult(_)));
        let results: Vec<UserContent> = results
            .into_iter()
            .filter(|item| matches!(item, UserContent::ToolResult(result) if owed.contains(&&result.call)))
            .cloned()
            .collect();
        if let Some(mut entry) = conversation
            .entry(entry.index)
            .filter(|_| !results.is_empty())
        {
            entry.message = Arc::new(Message::User { content: results });
            kept.push(entry);
        }
        if asked.is_empty() {
            continue;
        }
        taken = true;
        if entry.is_users_own() {
            let text: Vec<&str> = asked
                .iter()
                .filter_map(|item| match item {
                    UserContent::Text(text) if attached_file(&text.text).is_none() => {
                        Some(text.text.as_str())
                    }
                    _ => None,
                })
                .collect();
            texts.extend((!text.is_empty()).then(|| text.join("\n\n")));
        }
    }
    taken.then(|| {
        conversation.restore(from.., kept, Author::rig());
        texts
    })
}

/// Why a turn the user stopped stopped, as its calls are told.
const STOPPED_BY_USER: &str = "the user stopped the turn";

/// Stops the agent's running turn at once, and the running turns of every
/// agent below it in [`Spawned`], even when the agent itself is idle:
/// stopping follows ancestry, so work meant to outlive its parent is
/// spawned without [`SpawnedBy`](super::agent::SpawnedBy). What a streaming
/// reply had sent is kept as the model's message, aborted, so no model
/// reads it again and the user's request is halted as stopped, unless its
/// call was superseded; every tool call of the last reply gets a result,
/// real, its [`IfStopped`] or what [`not_finished`](feedback::not_finished)
/// says, and every call of the turn that has not settled is answered so
/// and settled, whatever held it gone. Each turn ends
/// [`TurnOutcome::Stopped`], in its record too, set before its calls
/// settle, so a call's owner sees it stopped them. The conversations are
/// saved and written to the log at once, in one flush, so a log that ends
/// mid-turn means a crash, a restart or a log that stopped (shown once, as
/// every log failure is); a stop never waits on the log.
pub(crate) fn on_stop_turn(
    interrupt: On<StopTurn>,
    spawned: Query<&Spawned>,
    mut agents: Query<(&AgentId, &mut Conversation, &ActiveTurn)>,
    mut turns: Query<(&mut Turn, Option<&Calls>)>,
    replies: Query<ReplyCall>,
    partials: Query<&Partial, (With<TurnCall>, Without<Settled>, Without<SupersededBy>)>,
    log: Res<SessionLog>,
    mut committed: MessageWriter<ConversationChanged>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let root = interrupt.entity;
    let reason = interrupt.reason.clone();
    let why = reason.clone().unwrap_or_else(|| STOPPED_BY_USER.to_owned());
    let below = spawned.iter_descendants::<Spawned>(root);
    let stopping: Vec<Entity> = std::iter::once(root).chain(below).collect();
    let mut stopped = false;
    for agent in stopping {
        let Ok((id, mut conversation, active)) = agents.get_mut(agent) else {
            continue;
        };
        let turn = active.turn();
        let (reply, calls) = match turns.get_mut(turn) {
            Ok((mut state, calls)) => (state.reply.take(), calls),
            Err(_) => (None, None),
        };
        let streamed = partials.iter_many(calls.into_iter().flat_map(Calls::iter));
        let (mut reasoning, mut text) = (String::new(), String::new());
        for partial in streamed.flatten() {
            reasoning.push_str(&partial.reasoning);
            text.push_str(&partial.text);
        }
        let reasoning =
            (!reasoning.trim().is_empty()).then(|| AssistantContent::reasoning(reasoning));
        let text = (!text.trim().is_empty()).then(|| AssistantContent::text(text));
        let content: Vec<_> = reasoning.into_iter().chain(text).collect();
        if !content.is_empty() {
            let aborted = AssistantMessage::aborted(None, content, format!("stopped: {why}"));
            conversation.push(Message::Assistant(aborted));
        }
        let reply = reply
            .zip(calls)
            .map(|(reply, calls)| reply_calls(reply, calls, &replies));
        let stopped_call =
            |call: &ReplyCallItem| Some(not_finished(&why, call.started, call.footprint));
        if let Some(results) = reply_results(reply.unwrap_or_default(), stopped_call) {
            commit_results(&mut conversation, results);
        }
        conversation.halt(Halt::Stopped);
        conversation.end_turn(TurnEnd::Stopped {
            why: reason.clone(),
        });
        let conversation = conversation.bypass_change_detection();
        committed.write_batch(save_conversation(&log, agent, id, conversation));
        let calls: Vec<Entity> = calls.into_iter().flat_map(Calls::iter).collect();
        let (why, reason) = (why.clone(), reason.clone());
        commands.queue(move |world: &mut World| {
            // The outcome goes on first, as `stop_calls` finds it, so what
            // observes a call settling sees that its turn stopped.
            if let Ok(mut turn) = world.get_entity_mut(turn) {
                turn.insert(TurnOutcome::Stopped(reason));
            }
            for call in calls {
                let said = world.get_entity(call).ok().and_then(|call| {
                    let footprint = call.get::<Footprint>();
                    call.contains::<ToolRequest>()
                        .then(|| not_finished_text(&why, call.contains::<Started>(), footprint))
                });
                cancel(world, call, said.as_deref().unwrap_or(&why));
            }
            if let Ok(mut turn) = world.get_entity_mut(turn) {
                turn.despawn_related::<Holds>();
            }
            if let Ok(mut turn) = world.get_entity_mut(turn) {
                turn.remove::<TurnOf>().insert(Settled);
            }
        });
        notices.write(Notice::info(agent, "Stopped."));
        stopped = true;
    }
    // A failed write stops the logging, which the frame's log write shows;
    // the turns stop all the same, and nothing more starts unsaved.
    if stopped {
        let _ = log.flush();
    }
}

/// What the model reads of a tool call that did not finish, for `why`
/// ([`feedback::not_finished`]); a call with no [`Footprint`] yet may
/// change anything.
fn not_finished_text(why: &str, started: bool, footprint: Option<&Footprint>) -> String {
    let footprint = footprint.cloned().unwrap_or_default();
    feedback::not_finished(why, started, &footprint.0)
}

/// The result of a tool call that did not finish, for `why`
/// ([`not_finished_text`]).
pub(crate) fn not_finished(why: &str, started: bool, footprint: Option<&Footprint>) -> ToolResult {
    let text = not_finished_text(why, started, footprint);
    ToolResult::failed(ToolExecutionError::other(text))
}

/// The results of a reply's tool calls, in call order: each answered
/// call's result, else what it says its result is if stopped
/// ([`IfStopped`]), else what `missing` gives it, else nothing for it.
/// `None` when that leaves no result.
fn reply_results(
    calls: Vec<ReplyCallItem>,
    missing: impl Fn(&ReplyCallItem) -> Option<ToolResult>,
) -> Option<Message> {
    let results: Vec<_> = calls
        .into_iter()
        .filter_map(|call| {
            let outcome = call.outcome.or(call.if_stopped.map(|stopped| &stopped.0));
            let result = outcome
                .map(Outcome::tool_result)
                .or_else(|| missing(&call))?;
            Some(call.request.call.answer(&result))
        })
        .collect();
    (!results.is_empty()).then(|| Message::tool_results(results))
}

/// How long exit waits for running calls to be cancelled. A cancellation
/// waits for a pool thread to drop the call's future, which records its
/// effect as cancelled before the last flush; a plugin tool that blocks in
/// `poll` must not stall the exit, so whatever misses this is dropped.
const EXIT_GRACE: Duration = Duration::from_secs(1);

type Cancelling = Vec<Pin<Box<dyn Future<Output = ()>>>>;

/// Inserted when the app exits: the turns it stops are left for the
/// restart to carry on, so they end without a [`TurnOutcome`] and are not
/// logged as halted.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct Exiting;

/// On exit (`/quit` or a signal; `/reload` and switching sessions wait for
/// idle agents) leaves the running turns for the restart to carry on, as
/// after a crash: every [`Running`] task, a plugin's too, is cancelled, the
/// tool results that came in are logged, and the restart answers the rest.
pub(crate) fn stop_turns_on_exit(world: &mut World) {
    world.insert_resource(Exiting);
    let mut cancelling = Cancelling::new();
    take_running(world, &mut cancelling);
    let deadline = Instant::now() + EXIT_GRACE;
    loop {
        cancelling.retain_mut(|cancel| check_ready(cancel).is_none());
        // On the web the calls run on this thread: waiting cannot help them.
        if cancelling.is_empty() || cfg!(target_family = "wasm") || Instant::now() >= deadline {
            break;
        }
        #[cfg(not(target_family = "wasm"))]
        std::thread::sleep(Duration::from_millis(2));
    }
    if let Err(error) = world.run_system_cached(log_answered_results) {
        warn!("could not log the results of stopped turns: {error}");
    }
    let turns: Vec<Entity> = world
        .query_filtered::<Entity, With<TurnOf>>()
        .iter(world)
        .collect();
    for turn in turns {
        world.despawn(turn);
    }
    world.flush();
}

/// Logs the results the running replies' calls got so far, in call order,
/// for the restart, which answers the rest.
fn log_answered_results(
    turns: Query<(&TurnOf, &Turn, &Calls)>,
    replies: Query<ReplyCall>,
    mut agents: Query<&mut Conversation>,
) {
    for (&TurnOf(agent), state, calls) in &turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let results = reply_results(reply_calls(reply, calls, &replies), |_| None);
        if let Ok(mut conversation) = agents.get_mut(agent)
            && let Some(results) = results
        {
            commit_results(&mut conversation, results);
        }
    }
}

/// Takes every [`Running`] task off its entity, a plugin's too, to be
/// cancelled.
fn take_running(world: &mut World, cancelling: &mut Cancelling) {
    let calls: Vec<Entity> = world
        .query_filtered::<Entity, With<Running>>()
        .iter(world)
        .collect();
    for call in calls {
        if let Ok(mut call) = world.get_entity_mut(call)
            && let Some(running) = call.take::<Running>()
        {
            cancelling.push(Box::pin(running.cancel()));
        }
    }
}

/// An agent as the kernel builds a model request from it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Builder {
    id: &'static AgentId,
    cache: Option<&'static CacheKey>,
    connection: Option<&'static Connection>,
    effort: &'static ReasoningSetting,
    prompt: &'static SystemPrompt,
    access: &'static ToolAccess,
    conversation: &'static mut Conversation,
    parent: Option<&'static EffectParent>,
}

/// A model call whose request the kernel builds, as Admit finds it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Unbuilt {
    entity: Entity,
    effect: &'static mut Effect,
    from: Option<&'static SideCall>,
    turn_call: Has<TurnCall>,
    of: Option<&'static CallOf>,
    agent: Option<&'static ForAgent>,
}

/// What a request is built from besides its agent: the tools and prompt
/// sections.
#[derive(SystemParam)]
pub(crate) struct Parts<'w, 's> {
    tools: Query<'w, 's, &'static ToolDef>,
    sections: Query<'w, 's, (&'static PromptSection, Option<&'static SectionOf>)>,
}

/// Builds the [`ModelRequest`] of each [`TurnCall`] and [`SideCall`]
/// admitted this pass without one, in [`Stage::Admit`]: the agent's system
/// prompt with its prompt sections, its conversation, the tools,
/// sorted by name so the prompt cache keeps them, and its model's options
/// with its reasoning setting. A turn's call first delivers the steering
/// and Inform messages its agent accepted meanwhile. A call the kernel
/// cannot build is answered with why.
pub(crate) fn build_requests(
    mut unbuilt: Query<
        Unbuilt,
        (
            Or<(With<TurnCall>, With<SideCall>)>,
            Added<Admitted>,
            Without<ModelRequest>,
            Without<Outcome>,
            Without<Started>,
        ),
    >,
    turns: Query<&TurnOf>,
    mut agents: Query<Builder>,
    inboxes: Inboxes,
    parts: Parts,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for mut call in &mut unbuilt {
        let turn = call.of.map(|of| of.0);
        let agent = call
            .from
            .map(|from| from.0)
            .or(call.agent.map(|agent| agent.0))
            .or_else(|| turns.get(turn?).ok().map(|of| of.0));
        let mut built = commands.entity(call.entity);
        let Some((agent, mut builder)) =
            agent.and_then(|agent| Some((agent, agents.get_mut(agent).ok()?)))
        else {
            let why = ErrorReport::new(ErrorKind::Internal, "a model call needs an agent");
            built.try_insert_if_new(Outcome(Err(why)));
            continue;
        };
        if call.agent.is_none() {
            built.try_insert(ForAgent(agent));
        }
        let effect = &mut *call.effect;
        if effect.parent.is_none() {
            effect.parent = builder.parent.map(|parent| parent.0);
        }
        let spec = builder.connection.map(|connection| &*connection.spec);
        if call.turn_call {
            effect.purpose.get_or_insert("turn".into());
            let waiting = inboxes.of(agent).into_iter();
            let waiting = waiting.filter(|(_, message)| message.mode != DeliveryMode::Queue);
            let conversation = &mut *builder.conversation;
            let into = (turn, &mut commands);
            deliver(agent, spec, waiting, conversation, &mut notices, into);
        }
        if call.turn_call && builder.conversation.messages().is_empty() {
            let why = ErrorReport::new(ErrorKind::Internal, "The conversation is empty.");
            commands
                .entity(call.entity)
                .try_insert_if_new(Outcome(Err(why)));
            continue;
        }
        let history = builder.conversation.request_view();
        let model = build((agent, &builder), spec, &parts, history);
        commands.entity(call.entity).try_insert(model);
    }
}

/// The request `agent`, as `builder` reads it, sends with `history` to the
/// model `spec`, if it has one.
fn build(
    (agent, builder): (Entity, &BuilderItem),
    spec: Option<&rig_core::catalog::ModelSpec>,
    parts: &Parts,
    history: rig_core::conversation::Conversation,
) -> ModelRequest {
    // Offered by access alone: Start takes them away from a final model
    // that calls none.
    let offered = offered(parts.tools.iter(), builder.access);
    // The role, then the sections by order, then tag.
    let mut sections: Vec<&PromptSection> = parts
        .sections
        .iter()
        .filter(|(_, of)| of.is_none_or(|of| of.0 == agent))
        .map(|(section, _)| section)
        .collect();
    sections.sort_by(|a, b| (a.order, &a.tag).cmp(&(b.order, &b.tag)));
    let sections = sections
        .into_iter()
        .map(|section| Section::new(&section.tag, &section.text));
    let prompt = completion::SystemPrompt {
        role: builder.prompt.0.clone(),
        sections: sections.collect(),
    };
    let effort = builder.effort.0;
    let options = spec.map_or_else(
        || {
            let options = GenerationOptions::default();
            match effort {
                Some(reasoning) => options.reasoning(reasoning),
                None => options,
            }
        },
        |spec| spec.default_options(effort),
    );
    let request = CompletionRequest::from(Vec::new())
        .tools(offered.iter().map(|def| def.0.clone()))
        .options(options.cache_key(CacheKey::of(builder.cache, builder.id)));
    ModelRequest {
        request: request.into(),
        connection: builder.connection.cloned(),
        prompt: Some(prompt),
        history: Some(history),
    }
}

/// What Start sends a model call: read it in Prepare's Fit phase to
/// measure a request as its model will get it, or in Gate to answer a call
/// with what its model would have been sent.
#[derive(SystemParam)]
pub struct ModelStarts<'w, 's> {
    agents: Query<'w, 's, &'static Connection, With<Agent>>,
}

impl ModelStarts<'_, '_> {
    /// The connection `model`, a call for `agent`, goes to, its `adapters`
    /// applied, and its request as Start sends it: its
    /// [`history`](ModelRequest::history)'s messages, tools taken off a
    /// model that calls none, options left unset taking that model's
    /// defaults (its caching), options it does not take left out, and the
    /// [`prompt`](ModelRequest::prompt) in front. Refused without a model.
    pub fn sent(
        &self,
        model: &ModelRequest,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<(Connection, CompletionRequest), ErrorReport> {
        let (connection, request, _) = self.sending(model, adapters, agent)?;
        Ok((connection, request))
    }

    /// The options the request Start sends for `model`, a call for `agent`,
    /// leaves out, as [`DroppedOptions`] will say: those its model does not
    /// take and those its handler's check refuses. The check builds the
    /// provider's request, so a caller that needs only [`sent`](Self::sent)
    /// does not pay for it. Refused without a model.
    pub fn dropped(
        &self,
        model: &ModelRequest,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<Vec<UnsupportedOption>, ErrorReport> {
        let (connection, request, mut dropped) = self.sending(model, adapters, agent)?;
        let kind = EffectKind::Completion {
            request: request.into(),
            stream: true,
        };
        dropped.extend(connection.handler.0.check(&kind));
        Ok(dropped)
    }

    /// The connection `model`, a call for `agent`, goes to, as
    /// [`sent`](Self::sent) says, and the system prompt Start sends in
    /// front of its messages (`None` without a prompt in parts): what
    /// [`sent`](Self::sent) puts before the history, without copying the
    /// request. Refused without a model.
    pub fn prompt(
        &self,
        model: &ModelRequest,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<(Connection, Option<completion::SystemPrompt>), ErrorReport> {
        let connection = self.connection(model.connection.as_ref(), adapters, agent)?;
        Ok((connection, model.prompt.clone()))
    }

    /// [`Self::sent`], with the options fitting left out: Start on a copy
    /// of `model`, whose messages are shared.
    fn sending(
        &self,
        model: &ModelRequest,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<(Connection, CompletionRequest, Vec<UnsupportedOption>), ErrorReport> {
        let mut model = model.clone();
        let (connection, dropped) = self.start(&mut model, adapters, agent)?;
        Ok((connection, Arc::unwrap_or_clone(model.request), dropped))
    }

    /// Makes `model`'s request the one [`Self::sent`] says, in place, on the
    /// connection it returns with the options fitting left out.
    pub(crate) fn start(
        &self,
        model: &mut ModelRequest,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<(Connection, Vec<UnsupportedOption>), ErrorReport> {
        let request = Arc::make_mut(&mut model.request);
        let to = model.connection.as_ref();
        let started = self.fit(request, to, adapters, agent)?;
        let history = match model.history.take() {
            Some(history) => history.into_messages(),
            None => mem::take(&mut request.chat_history),
        };
        // The system prompt, if the request has one in parts, goes first.
        let prompt = model.prompt.take();
        let prompt = prompt.map(|prompt| Arc::new(Message::system(prompt.text())));
        request.chat_history = prompt.into_iter().chain(history).collect();
        Ok(started)
    }

    /// The connection a call goes to: `to` or else `agent`'s, with
    /// `adapters` applied. Refused without a model.
    fn connection(
        &self,
        to: Option<&Connection>,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<Connection, ErrorReport> {
        let own = agent.and_then(|agent| self.agents.get(agent).ok());
        let Some(connection) = to.or(own).cloned() else {
            return Err(ErrorReport::new(ErrorKind::HandlerUnavailable, NO_MODEL));
        };
        Ok(match adapters {
            Some(adapters) => adapters.apply(connection),
            None => connection,
        })
    }

    /// Fits `request` to the [connection](Self::connection) it goes to:
    /// tools taken off a model that calls none, options left unset taking
    /// that model's defaults (its caching), options it does not take left
    /// out.
    fn fit(
        &self,
        request: &mut CompletionRequest,
        to: Option<&Connection>,
        adapters: Option<&Adapters>,
        agent: Option<Entity>,
    ) -> Result<(Connection, Vec<UnsupportedOption>), ErrorReport> {
        let connection = self.connection(to, adapters, agent)?;
        let spec = &connection.spec;
        if !spec.tools {
            request.tools.clear();
        }
        // An option the request leaves unset is the final model's default,
        // whoever built the request for whichever model; `cache` is the
        // only default a model has, and a new one is filled the same way.
        if request.options.cache.is_none() {
            request.options.cache = spec.default_options(request.options.reasoning).cache;
        }
        let dropped = spec.fit(request);
        Ok((connection, dropped))
    }
}

/// Streams the reply, feeding `feed` when each item comes and with its
/// events, waking the loop for each, and
/// returns the response the provider ended it with.
pub(crate) async fn stream_reply(
    reply: Reply,
    feed: Sender<Fed>,
    wake: Wake,
) -> Result<effect::Answered, ErrorReport> {
    let mut events = reply.into_stream();
    while let Some(item) = events.next().await {
        let event = match item {
            Err(report) => return Err(report),
            // Where the reply is from, before its first item.
            Ok(Relayed::Origin(_)) => continue,
            Ok(Relayed::Done(response)) => {
                // The call may be gone; the reply still finishes.
                feed.send(Fed(web_time::SystemTime::now(), None)).ok();
                return Ok(effect::Answered::Completion(response));
            }
            Ok(Relayed::Item(Item::Event(event))) => Some(event),
            Ok(Relayed::Item(_)) => None,
        };
        feed.send(Fed(web_time::SystemTime::now(), event)).ok();
        wake.wake();
    }
    Err(stream_truncated())
}

/// The channel a model call streams into: the sender for its task, and
/// the [`Feed`] for its entity.
pub(crate) fn feed() -> (Sender<Fed>, Feed) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    (sender, Feed(receiver))
}

/// Reads what each model call streamed, in [`Stage::Arrive`] after the
/// finished calls' outcomes, so a reply that finished has fed it all: its
/// [`StreamTiming`], a [`StreamDelta`] per event, in order, and the text
/// and reasoning into its [`Partial`], if it has one.
pub(crate) fn stream_partials(
    mut calls: Query<(
        Entity,
        &Feed,
        Option<&mut Partial>,
        Option<&mut StreamTiming>,
        Option<&ForAgent>,
        Option<&CallOf>,
        Has<TurnCall>,
    )>,
    mut deltas: MessageWriter<StreamDelta>,
    mut commands: Commands,
) {
    for (effect, feed, mut partial, mut streamed, agent, of, turn_call) in &mut calls {
        // The stamp of a call whose first item came now.
        let mut first: Option<StreamTiming> = None;
        for Fed(at, event) in feed.0.try_iter() {
            match streamed.as_deref_mut().or(first.as_mut()) {
                Some(stamp) => {
                    stamp.last = at;
                    stamp.items += 1;
                }
                None => {
                    first = Some(StreamTiming {
                        first: at,
                        last: at,
                        items: 1,
                    });
                }
            }
            let Some(event) = event else {
                continue;
            };
            if let Some(&CallOf(turn)) = of.filter(|_| turn_call) {
                commands.entity(turn).try_insert_if_new(FirstStreamed(at));
            }
            if let Some(partial) = partial.as_mut() {
                match &event {
                    StreamEvent::Text { text, .. } => partial.text.push_str(text),
                    StreamEvent::Reasoning { text, .. } => partial.reasoning.push_str(text),
                    _ => {}
                }
            }
            let agent = agent.map(|agent| agent.0);
            deltas.write(StreamDelta {
                effect,
                agent,
                event,
            });
        }
        if let Some(first) = first {
            commands.entity(effect).try_insert(first);
        }
    }
}

/// Puts the [`RETRY_POLICY`] policy's [`RetryVerdict`] on each model call whose
/// handler just failed, in [`Stage::Arrive`], counting its turn's
/// [`Recovery`].
pub(crate) fn judge_failures(
    failed: Query<
        (Entity, &Effect, &Outcome, Option<&CallOf>),
        (With<ModelRequest>, Added<Outcome>, Without<Settled>),
    >,
    turns: Query<&Recovery>,
    mut commands: Commands,
) {
    for (call, effect, Outcome(outcome), of) in &failed {
        let Err(report) = outcome else {
            continue;
        };
        if !effect.ran() {
            continue;
        }
        let retries = of
            .and_then(|of| turns.get(of.0).ok())
            .map_or(0, |recovery| recovery.retries);
        // std's clock, which a `Retry-After` date is read against, read
        // through `web_time` so it works on the web too.
        let now = UNIX_EPOCH + Duration::from_millis(now_ms());
        let verdict = RETRY_POLICY.verdict(report, retries, now);
        commands.entity(call).try_insert(RetryVerdict(verdict));
    }
}

/// Holds each failed [`TurnCall`] whose verdict, after Review, is still
/// `Retry` with a [`Backoff`], in [`Stage::Commit`] before the kernel
/// settles, and sends a new call that supersedes it once the wait passed,
/// if the hold is still there then. Not when a Review system failed, nor
/// for a turn that is failed or stopped.
pub(crate) fn back_off(
    failed: Query<
        (Entity, &Effect, &Outcome, &RetryVerdict, &CallOf),
        (
            With<TurnCall>,
            Without<Holds>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
    mut turns: Query<(&TurnOf, &mut Recovery, Option<&TurnOutcome>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, effect, Outcome(outcome), RetryVerdict(verdict), &CallOf(turn)) in &failed {
        let (Err(report), Verdict::Retry(delay)) = (outcome, verdict) else {
            continue;
        };
        let Ok((&TurnOf(agent), mut recovery, ending)) = turns.get_mut(turn) else {
            continue;
        };
        if ending.and_then(TurnOutcome::stops).is_some() {
            continue;
        }
        recovery.retries += 1;
        let backoff = Backoff::after(recovery.retries, *delay);
        notices.write(Notice::info(
            agent,
            format!(
                "The model call failed: {report}. Retrying in {}s ({}/{}).",
                backoff.seconds_left(),
                backoff.attempt,
                RETRY_POLICY.max_retries
            ),
        ));
        let hold = commands
            .spawn((Name::new("retry wait"), backoff, HoldOf(call)))
            .id();
        let parent = effect.parent;
        commands
            .delayed()
            .duration(*delay)
            .queue(move |world: &mut World| {
                // Interrupting the turn despawned the call and its hold.
                if world.get_entity(hold).is_err() {
                    return;
                }
                world.despawn(hold);
                world
                    .spawn((
                        TurnCall,
                        CallOf(turn),
                        ForAgent(agent),
                        Effect::with_parent_id(parent),
                    ))
                    .add_related::<SupersededBy>(&[call]);
            });
    }
}

/// Answers every admitted call of a failed or stopped turn that has not
/// settled, in [`Stage::Commit`] before the kernel settles: one not answered
/// yet with its [`IfStopped`], else as stopped, which cancels it if it runs,
/// and whatever holds them is despawned, so they settle in this pass and the
/// turn can end. A call spawned since the last pass is left for Admit, where a
/// new [`TurnCall`] carries the turn on.
pub(crate) fn stop_calls(
    turns: Query<(&TurnOutcome, &Calls), With<TurnOf>>,
    calls: Query<
        (
            Has<Outcome>,
            Option<&Holds>,
            Option<&IfStopped>,
            (Has<ToolRequest>, Has<Started>, Option<&Footprint>),
        ),
        (With<Admitted>, Without<Settled>),
    >,
    mut commands: Commands,
) {
    for (ending, of) in &turns {
        let Some(why) = ending.stops() else {
            continue;
        };
        for (call, (answered, holds, if_stopped, (tool, started, footprint))) in of
            .iter()
            .filter_map(|call| Some((call, calls.get(call).ok()?)))
        {
            release(&mut commands, holds);
            if !answered {
                let said = match tool {
                    true => not_finished_text(&why, started, footprint),
                    false => why.clone(),
                };
                let stopped = match if_stopped {
                    Some(IfStopped(outcome)) => outcome.clone(),
                    None => Outcome(Err(ErrorReport::new(ErrorKind::Cancelled, said))),
                };
                commands.entity(call).try_insert_if_new(stopped);
            }
        }
    }
}

/// Takes each [`TurnCall`] settled this pass and not superseded, in
/// [`Stage::Commit`], its streamed [`Partial`] gone: rig-core's
/// turn-failure rule decides first, and a failed reply is not appended
/// and runs none of its tool calls; a reply without tool calls is the
/// turn's answer, unless
/// messages sent meanwhile carry it on; otherwise its tool calls are
/// spawned, all at once, and become the turn's [`Turn::reply`]. A failure
/// fails the turn as its verdict says and gives the user's message back.
/// Each end is a [`TurnOutcome`] proposed on the turn, unless it has one: a
/// failed or stopped turn takes in a reply that still came, its tool calls
/// not run, and goes no further.
pub(crate) fn commit_turn_calls(
    calls: Query<
        (
            Entity,
            &Effect,
            Option<&Outcome>,
            Option<&RetryVerdict>,
            &CallOf,
            Has<SupersededBy>,
        ),
        (With<TurnCall>, Added<Settled>),
    >,
    mut turns: Query<(&TurnOf, &mut Recovery, &mut Turn, Option<&TurnOutcome>)>,
    mut agents: Query<Answered>,
    inboxes: Inboxes,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, effect, outcome, verdict, &CallOf(turn), superseded) in &calls {
        // What it streamed is in the conversation now, or never goes there.
        commands.entity(call).try_remove::<(Partial, Feed)>();
        let Some(Outcome(outcome)) = outcome.filter(|_| !superseded) else {
            continue;
        };
        let Ok((&TurnOf(agent), mut recovery, mut state, ending)) = turns.get_mut(turn) else {
            continue;
        };
        let Ok(mut answered) = agents.get_mut(agent) else {
            continue;
        };
        let stopped = ending.and_then(TurnOutcome::stops);
        let report = match outcome {
            Ok(effect::Answered::Completion(response)) => {
                recovery.retries = 0;
                let inbox = (&inboxes, &mut commands);
                let next =
                    commit_reply((agent, turn), response, &mut answered, inbox, &mut notices);
                match (next, stopped) {
                    (Next::Answered(answer), _) => {
                        if ending.is_none() {
                            let answer = TurnOutcome::Answered(answer);
                            commands.entity(turn).try_insert(answer);
                        }
                    }
                    (Next::Failed(why), _) => fail(&mut commands, turn, why),
                    (Next::Call, None) => spawn_turn_call(&mut commands, turn, agent),
                    (Next::Run(calls), None) => {
                        state.reply = Some(effect.id());
                        let cut_off = cut_off(response);
                        spawn_reply(&mut commands, (agent, turn), calls, effect.id(), &cut_off);
                    }
                    (Next::Run(calls), Some(why)) => {
                        let results = close_pending_with(&calls, &format!("not run: {why}"));
                        commit_results(&mut answered.conversation, results);
                    }
                    (Next::Call, Some(_)) => {}
                }
                continue;
            }
            _ if stopped.is_some() => continue,
            Ok(other) => ErrorReport::new(
                ErrorKind::Internal,
                format!("the model answered with a {} outcome", other.family()),
            ),
            Err(report) => report.clone(),
        };
        // The verdict says what failed; a call refused before it was sent
        // has none, so its report does.
        let verdict = verdict.map(|verdict| &verdict.0);
        let why = match verdict {
            None => report.message.clone(),
            Some(Verdict::GaveUp(why)) => {
                format!("The model call failed: {report}. Not retrying: {why}")
            }
            Some(Verdict::Overflow) => {
                format!("The conversation does not fit the model's context window: {report}")
            }
            Some(Verdict::Retry(_) | Verdict::Final) => format!("The model call failed: {report}"),
        };
        fail(&mut commands, turn, why);
    }
}

/// What a turn does after a reply.
enum Next {
    /// It would end with its answer, as proposed.
    Answered(Message),
    /// The kernel would fail it, for its reason.
    Failed(String),
    /// It calls the model again, with the messages sent meanwhile.
    Call,
    /// It runs the reply's tool calls.
    Run(Vec<ToolCall>),
}

/// Appends `response`, the reply to `agent`, and says what its turn does
/// next.
fn commit_reply(
    (agent, turn): (Entity, Entity),
    response: &CompletionResponse,
    answered: &mut AnsweredItem,
    (inboxes, commands): (&Inboxes, &mut Commands),
    notices: &mut MessageWriter<Notice>,
) -> Next {
    let tool_calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    let failure = turn_failure(
        &response.choice,
        Some(&response.stop()),
        response.finish_reason().as_ref(),
    );
    // A failed reply stays out of the conversation, as the message it
    // failed to answer goes out of it.
    if let Some(failure) = failure {
        return Next::Failed(failure);
    }
    if let Some(message) = response.message() {
        answered.conversation.push(message);
        let through = answered.conversation.messages().len().checked_sub(1);
        let through = through.and_then(|last| answered.conversation.id(last));
        if let Some(through) = through.filter(|_| response.usage.context_tokens().is_some()) {
            let usage = response.usage;
            *answered.last = LastUsage(Some(Anchor { usage, through }));
        }
    }
    if !tool_calls.is_empty() {
        return Next::Run(tool_calls);
    }
    // What was sent meanwhile carries the turn on. Inform messages go along
    // but carry nothing on.
    if let Some(next) = carry_on(inboxes.of(agent)) {
        let spec = answered.connection.map(|connection| &*connection.spec);
        let conversation = &mut *answered.conversation;
        let into = (Some(turn), commands);
        deliver(agent, spec, next, conversation, notices, into);
        return Next::Call;
    }
    match answered
        .conversation
        .messages()
        .last()
        .map(|message| &**message)
    {
        Some(answer @ Message::Assistant(_)) => Next::Answered(answer.clone()),
        _ => Next::Failed("the model ended the turn without a message".to_owned()),
    }
}

/// An agent as its model's reply finds it.
#[derive(QueryData)]
#[query_data(mutable)]
pub(crate) struct Answered {
    conversation: &'static mut Conversation,
    connection: Option<&'static Connection>,
    last: &'static mut LastUsage,
}

/// Spawns a call of `turn`, for `agent`, for each of `calls`, in order,
/// asked for by the model call `reply`. Every call exists before any is
/// gated.
pub(crate) fn spawn_reply(
    commands: &mut Commands,
    (agent, turn): (Entity, Entity),
    calls: Vec<ToolCall>,
    reply: EffectId,
    cut_off: &[ToolCall],
) {
    for call in calls {
        let refused = cut_off.iter().find(|cut| cut.id == call.id).map(|cut| {
            let name = cut.function.name.as_str();
            let why = feedback::output_limit_feedback(name, &cut.function.raw_arguments());
            Outcome(Err(ErrorReport::new(
                ErrorKind::Tool(ToolErrorKind::InvalidArgs),
                why,
            )))
        });
        let mut call = commands.spawn((
            ToolRequest { call },
            Effect::with_parent_id(Some(reply)),
            CallOf(turn),
            ForAgent(agent),
        ));
        if let Some(refused) = refused {
            call.insert(refused);
        }
    }
}

/// The tool calls `response` ended in the middle of at the output limit:
/// each is answered as cut off, never run, so the model sends less rather
/// than the same call again.
fn cut_off(response: &CompletionResponse) -> Vec<ToolCall> {
    let truncated = response
        .finish_reason()
        .is_some_and(|finish| finish.truncated_output());
    let unfinished = response.unfinished_calls().filter(|_| truncated);
    unfinished.cloned().collect()
}

/// A call of a model's reply, as the stages find it.
#[derive(QueryData)]
pub(crate) struct ReplyCall {
    effect: &'static Effect,
    request: &'static ToolRequest,
    outcome: Option<&'static Outcome>,
    if_stopped: Option<&'static IfStopped>,
    successor: Option<&'static SupersededBy>,
    footprint: Option<&'static Footprint>,
    settled: Has<Settled>,
    started: Has<Started>,
    waiting: Has<Waiting>,
    entity: Entity,
}

/// The calls of the reply `reply` among `calls`, in the reply's order:
/// for each call the model asked for, the last effect that superseded it.
fn reply_calls<'a>(
    reply: EffectId,
    calls: &Calls,
    replies: &'a Query<ReplyCall>,
) -> Vec<ReplyCallItem<'a, 'a>> {
    let mut asked: Vec<ReplyCallItem<'a, 'a>> = Vec::new();
    for call in calls.iter() {
        let Ok(mut call) = replies.get(call) else {
            continue;
        };
        let first = |at: &ReplyCallItem| at.request.call.id == call.request.call.id;
        if call.effect.parent != Some(reply) || asked.iter().any(first) {
            continue;
        }
        for _ in 0..calls.len() {
            match call
                .successor
                .and_then(|by| replies.get(by.successor()).ok())
            {
                Some(successor) => call = successor,
                None => break,
            }
        }
        asked.push(call);
    }
    asked
}

/// Marks each call of a running reply [`Waiting`] while an earlier call
/// whose [`Footprint`] conflicts with its own is not settled, and takes
/// the mark off once none is, when [`Stage::Gate`](super::stage::Stage::Gate)
/// opens. A held call holds back the calls after it that conflict with it;
/// a refused one, once it settled, does not. A call not yet admitted may
/// change anything.
pub(crate) fn wait_in_order(
    turns: Query<(&Turn, &Calls)>,
    replies: Query<ReplyCall>,
    mut commands: Commands,
) {
    let anything = Footprint::default();
    for (state, calls) in &turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let mut earlier: Vec<&Footprint> = Vec::new();
        for call in reply_calls(reply, calls, &replies) {
            let footprint = call.footprint.unwrap_or(&anything);
            let waits = earlier
                .iter()
                .any(|before| footprint.0.conflicts_with(&before.0));
            let open = !call.started && call.outcome.is_none() && !call.settled;
            if open && waits != call.waiting {
                let mut entity = commands.entity(call.entity);
                match waits {
                    true => entity.try_insert(Waiting),
                    false => entity.try_remove::<Waiting>(),
                };
            }
            if !call.settled {
                earlier.push(footprint);
            }
        }
    }
}

/// Once every call of a turn's running reply settled, appends their
/// results in call order and spawns the turn's next [`TurnCall`], unless
/// the turn is failed or stopped, in [`Stage::Commit`], after the kernel
/// settled that pass's effects. Until then, the mark of each call that
/// settled says its result, for a restart.
pub(crate) fn commit_replies(
    mut turns: Query<(Entity, &TurnOf, &mut Turn, &Calls, Option<&TurnOutcome>)>,
    replies: Query<ReplyCall>,
    mut agents: Query<&mut Conversation>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), mut state, calls, ending) in &mut turns {
        let Some(reply) = state.reply else {
            continue;
        };
        let calls = reply_calls(reply, calls, &replies);
        let Ok(mut conversation) = agents.get_mut(agent) else {
            continue;
        };
        if calls.iter().any(|call| !call.settled) {
            for call in calls.iter().filter(|call| call.settled) {
                if let Some(outcome) = call.outcome {
                    let result = outcome.tool_result();
                    mark_settled(&mut conversation, &call.request.call.id, &result);
                }
            }
            continue;
        }
        state.reply = None;
        let unanswered = ToolResult::failed(ToolExecutionError::other("the call had no result"));
        if let Some(results) = reply_results(calls, |_| Some(unanswered.clone())) {
            commit_results(&mut conversation, results);
        }
        if ending.and_then(TurnOutcome::stops).is_none() {
            spawn_turn_call(&mut commands, turn, agent);
        }
    }
}

/// Ends each turn whose [`TurnOutcome`] nothing holds and whose calls all
/// settled, in [`Stage::Commit`], before this pass's proposals are made, so
/// Review sees each proposal first: a failed turn, whoever failed it,
/// gives the user's unanswered messages back ([`take_unanswered`]) and
/// shows its error, a message left unanswered is halted, the turn is no
/// longer [`TurnOf`] its agent, and it is [`Settled`].
/// An `Answered` turn waits while a message that would carry it on is on
/// its way to its agent, and any turn waits a pass for a model call that
/// settled in this one, so [`commit_turn_calls`] takes in a reply that came
/// as a plugin failed the turn. Not when a Review system failed.
pub(crate) fn end_turns(
    turns: Query<(Entity, &TurnOf, &TurnOutcome, Option<&Calls>), Without<Holds>>,
    open: Query<(), (With<Effect>, Without<Settled>)>,
    arrived: Query<(), (With<TurnCall>, Added<Settled>)>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    inboxes: Inboxes,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (turn, &TurnOf(agent), outcome, calls) in &turns {
        let Ok((id, mut conversation)) = agents.get_mut(agent) else {
            continue;
        };
        if let TurnOutcome::Answered(_) = outcome
            && inboxes.coming(agent)
        {
            continue;
        }
        let waiting = |call| open.contains(call) || arrived.contains(call);
        if calls.is_some_and(|calls| calls.iter().any(waiting)) {
            continue;
        }
        if let TurnOutcome::Failed(why) = outcome {
            let taken = take_unanswered(&mut conversation);
            notices.write(Notice::error(agent, failure_notice(why)));
            match taken {
                Some(texts) if !texts.is_empty() => {
                    commands.entity(agent).try_insert(Unanswered(texts));
                }
                taken => {
                    notices.write(Notice::info(agent, way_on(taken.is_some())));
                }
            }
        }
        if !matches!(outcome, TurnOutcome::Answered(_)) {
            conversation.halt(Halt::Kept);
        }
        let end = turn_end(outcome, &conversation);
        conversation.end_turn(end);
        let how = match outcome {
            TurnOutcome::Answered(_) => "answered",
            TurnOutcome::Failed(_) => "failed",
            TurnOutcome::Stopped(_) => "stopped",
        };
        info!(agent = %id.0, "turn ended: {how}");
        commands
            .entity(turn)
            .try_remove::<TurnOf>()
            .try_insert(Settled);
    }
}

/// Takes the [`TurnOutcome`] off each turn whose [`TurnCall`] was admitted
/// this pass, in [`Stage::Admit`]: calling the model again carries the turn
/// on, whatever was proposed or decided.
pub(crate) fn carry_on_called(
    called: Query<&CallOf, (With<TurnCall>, Added<Admitted>)>,
    ending: Query<Option<&Holds>, With<TurnOutcome>>,
    mut commands: Commands,
) {
    for &CallOf(turn) in &called {
        if let Ok(holds) = ending.get(turn) {
            take_outcome(&mut commands, turn, holds);
        }
    }
}

/// Takes the outcome off `turn`, which carries on, and with it the `holds`
/// on that outcome: they were about an end that no longer comes.
pub(crate) fn take_outcome(commands: &mut Commands, turn: Entity, holds: Option<&Holds>) {
    commands.entity(turn).try_remove::<TurnOutcome>();
    release(commands, holds);
}

/// Whether the settled effect `entity` stays past its frame: a call of a
/// reply whose other calls still run, which keeps its result for them.
pub(crate) fn keeps_settled(world: &World, entity: Entity) -> bool {
    let reply = world
        .get::<CallOf>(entity)
        .and_then(|of| world.get::<Turn>(of.0))
        .and_then(|turn| turn.reply);
    let parent = world.get::<Effect>(entity).and_then(|effect| effect.parent);
    reply.is_some() && parent == reply && world.get::<ToolRequest>(entity).is_some()
}
