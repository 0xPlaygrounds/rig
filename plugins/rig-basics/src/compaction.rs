//! Compaction: keeps each agent's conversation inside its model's input limit by summarizing its
//! older messages, as the pi coding agent does. When enabled, it checks the context against
//! `[compaction] reserve_tokens` of the input limit before a prompt is sent (the context as it was
//! before the prompt), before each request after tool results (the context sent), and after a
//! completed turn, when it compacts while the agent is idle. After a request refused as too long,
//! or a reply the window cut short, it compacts and sends the request again, once.
//! `/compact [instructions]` compacts at once.
//!
//! A compaction keeps about `keep_recent_tokens` of the newest messages, never cutting between a
//! tool call and its result, and summarizes the older ones with a separate call to the agent's
//! model ([`Summarizing`]): the previous summary is updated with the new messages, and a turn the
//! cut splits gets a second summary of its earlier part. rig-memory's [`prepare_compaction`]
//! holds the cut, the prompts and the requests. A summary that failed or was cut short replaces
//! nothing and says why. A summary call's text is in [`Summarizing::summary`] in
//! [`ReviewSystems::Act`], where a reviewer may edit it or set it to `None` to cancel, and
//! [`Uncompact`] puts the newest compaction back.

use std::sync::{Arc, LazyLock};

use regex::Regex;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, Reasoning, UnsupportedOption, tokens_label};
use rig_core::conversation::Change;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{StopReason, UserContent};
use rig_ecs::prelude::*;
use rig_ecs::retry::RetryPolicy;
use rig_ecs::session_log::{ConversationChanged, SessionStore};
use rig_harness::AppKeepsExt;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::settings::AppSettingsExt;
use rig_harness::successor::resend;
use rig_harness::{PluginEntity, ProvidedBy};
pub use rig_memory::CutPoint;
use rig_memory::{
    CompactionPreparation, CompactionSettings, SummaryState, apply_summary, clamp_max_tokens,
    estimate_tokens, merge_split_turn, no_history, prepare_compaction, prepare_compaction_at,
    summary_request, summary_text, turn_prefix_request, undo_summary,
};
pub use rig_memory::{summarized_prefix, summary_author, summary_in, summary_state};
use serde::Deserialize;
#[cfg(feature = "tui")]
pub use tui::SummaryLooks;

/// The purpose of a summary's model call.
pub const PURPOSE: &str = "compaction";

/// Compacts agents' conversations near their model's input limit, after an overflow, and on
/// `/compact`, by [`CompactionDefault`].
#[derive(Default)]
pub struct CompactionPlugin;

impl Plugin for CompactionPlugin {
    fn build(&self, app: &mut App) {
        app.keeps_saved::<ContextStale>()
            .add_setting::<CompactionDefault>("compaction")
            .init_resource::<RetryPolicy>()
            .add_command(
                "compact",
                "Summarize the older messages to free context; /compact <instructions> says what to focus on",
                |In(args): In<CommandArgs>, mut commands: Commands| {
                    let (entity, focus) = (args.agent, args.args);
                    commands.trigger(Compact { entity, focus });
                },
            )
            .add_observer(on_compact)
            .add_observer(on_uncompact)
            .add_observer(check_after_turn)
            .add_systems(
                AgentPass,
                (
                    (compact_near_the_window, nothing_to_compact).in_set(PrepareSystems::Fit),
                    read_replies.in_set(ReviewSystems::Edit),
                    (retry_compaction_calls, recover_from_overflow).in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(Update, (stale_after_edits, compact_when_idle))
            .add_observer(take_summary)
            .add_observer(follow_successor::<Summarizing>)
            .add_observer(carry_recovery)
            .add_observer(provided_by_compaction);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// `[compaction]` from `settings.toml`: `enabled`, `reserve_tokens` and `keep_recent_tokens`
/// over rig-memory's defaults; an unknown key is refused.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct CompactionDefault(pub CompactionSettings);

/// Why a conversation is compacted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactReason {
    /// The user asked, with what to focus on, if anything.
    Asked {
        /// What the summary should focus on; may be empty.
        focus: String,
    },
    /// The next request leaves less than the reserve of the model's input limit free.
    Threshold,
    /// The model refused the request as too long.
    Overflow,
}

impl CompactReason {
    /// What the user asked the summary to focus on, trimmed; empty when they named nothing or
    /// did not ask.
    pub fn focus(&self) -> &str {
        match self {
            Self::Asked { focus } => focus.trim(),
            Self::Threshold | Self::Overflow => "",
        }
    }
}

/// Compact the agent's conversation now, focused on `focus` when it is not empty. Refused while
/// a turn runs.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Compact {
    /// The agent.
    pub entity: Entity,
    /// What the summary should focus on; may be empty.
    pub focus: String,
}

/// Put back the messages the agent's newest compaction replaced, with their ids, in place of its
/// summary, as the session's log holds them ([`undo_summary`]). Refused while a turn runs or in
/// a session that is not saved.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Uncompact {
    /// The agent.
    pub entity: Entity,
}

/// On a turn's model call compacted for, or whose compaction failed or was cancelled: it is not
/// compacted for again near the window. A call that replaces it, such as a retry, carries it.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Component, Clone, Debug)]
pub enum Recovering {
    /// Its compaction failed or was cancelled.
    NotSummarized,
    /// The conversation was compacted for it.
    Compacted,
}

/// On a turn's model call whose overflow or cut-short reply was recovered from, or tried to be: a
/// second one fails the turn. A call that replaces it carries it.
#[derive(Component, Clone, Copy, Debug)]
pub struct OverflowTried;

/// On an agent whose last turn ended over the reserve: it is compacted once idle.
#[derive(Component)]
struct CompactWhenIdle;

/// A summary call of a compaction: a [`ModelRequest`] of the turn, holding ([`HoldOf`]) the turn's
/// call it compacts for, if any, and [`ProvidedBy`] this plugin when loaded as one. A turn the cut
/// splits takes two calls, one after the other: the history, then the turn's earlier part.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The first message the compaction keeps, by id, as messages may move while the summary is
    /// written; `None` summarizes every message. A summary that arrives after it was taken out
    /// replaces nothing.
    pub keep_from: Option<MessageId>,
    /// The state the summary goes into: its files; the summary is written in once it is taken.
    pub state: SummaryState,
    /// The newest compaction's summary and file lists, which the update prompt sends verbatim;
    /// `None` before the first compaction.
    pub previous: Option<String>,
    /// The messages this call summarizes.
    pub messages: Arc<[Arc<completion::Message>]>,
    /// The earlier part of a turn the cut splits, summarized by the next call.
    pub turn_prefix: Option<Arc<[Arc<completion::Message>]>>,
    /// On the call for a split turn's earlier part, the history summary it follows.
    pub history: Option<String>,
    /// This call's summary, read from the model's reply in [`ReviewSystems::Edit`]: `None` before,
    /// and when the reply cannot be used. A reviewer in [`ReviewSystems::Act`] may replace it, or
    /// set it to `None` to cancel the compaction.
    pub summary: Option<String>,
    /// How many times this call was sent again after a passing failure.
    retries: u32,
}

impl Summarizing {
    /// The first summary call of a compaction of `conversation` for `reason`, keeping about
    /// `keep_recent_tokens`; `None` when nothing would be summarized.
    pub fn plan(
        conversation: &rig_core::conversation::Conversation,
        keep_recent_tokens: u64,
        reason: CompactReason,
    ) -> Option<Self> {
        let overflowed = reason == CompactReason::Overflow;
        let prepared = prepare_compaction(conversation, keep_recent_tokens, overflowed)?;
        Some(Self::of(conversation, prepared, reason))
    }

    /// The first summary call of a compaction of `conversation` for `reason` cut at `cut`, as a
    /// plugin that cuts its own way plans it; `None` when nothing would be summarized.
    pub fn plan_at(
        conversation: &rig_core::conversation::Conversation,
        cut: CutPoint,
        reason: CompactReason,
    ) -> Option<Self> {
        let prepared = prepare_compaction_at(conversation, cut)?;
        Some(Self::of(conversation, prepared, reason))
    }

    /// The first summary call of the compaction `prepared` of `conversation` for `reason`.
    fn of(
        conversation: &rig_core::conversation::Conversation,
        prepared: CompactionPreparation,
        reason: CompactReason,
    ) -> Self {
        let previous = prepared.previous;
        let state = SummaryState {
            summary: String::new(),
            read_files: prepared.files.read_files,
            modified_files: prepared.files.modified_files,
        };
        let (messages, turn_prefix, history) = match prepared.to_summarize.is_empty() {
            true => (
                prepared.turn_prefix,
                None,
                Some(no_history(previous.as_deref())),
            ),
            false => {
                let prefix =
                    (!prepared.turn_prefix.is_empty()).then(|| prepared.turn_prefix.into());
                (prepared.to_summarize, prefix, None)
            }
        };
        Self {
            reason,
            keep_from: conversation.id(prepared.first_kept),
            state,
            previous,
            messages: messages.into(),
            turn_prefix,
            history,
            summary: None,
            retries: 0,
        }
    }

    /// A `/compact` this plugin finds nothing to summarize in, which a plugin that cuts its own
    /// way may plan anew ([`plan_at`](Self::plan_at)) before Fit, where it is refused.
    fn nothing(reason: CompactReason) -> Self {
        Self {
            reason,
            keep_from: None,
            state: SummaryState::default(),
            previous: None,
            messages: Arc::new([]),
            turn_prefix: None,
            history: None,
            summary: None,
            retries: 0,
        }
    }

    /// What this call's reply is called in a failure: the history summary or the turn prefix's.
    fn label(&self) -> &'static str {
        match self.history {
            Some(_) => "Turn prefix summarization",
            None => "Summarization",
        }
    }

    /// The request of this call to `spec`, with the agent's `reasoning` effort.
    ///
    /// # Errors
    ///
    /// When `spec` does not take the request's options.
    pub fn request(
        &self,
        spec: &ModelSpec,
        reasoning: Option<Reasoning>,
    ) -> Result<CompletionRequest, UnsupportedOption> {
        let focus = self.reason.focus();
        match self.history {
            Some(_) => turn_prefix_request(&self.messages, focus, spec, reasoning),
            None => {
                let previous = (self.previous.as_deref(), focus);
                summary_request(&self.messages, previous, spec, reasoning)
            }
        }
    }

    /// The call for the earlier part of the split turn this call's `summary` is the history of;
    /// `None` when the cut split no turn.
    fn next(&self, summary: String) -> Option<Self> {
        let messages = self.turn_prefix.clone()?;
        Some(Self {
            messages,
            turn_prefix: None,
            history: Some(summary),
            summary: None,
            retries: 0,
            ..self.clone()
        })
    }
}

/// An agent as compacting reads it.
type Compacting = (
    &'static Conversation,
    Option<&'static Connection>,
    &'static LastUsage,
    Option<&'static ContextStale>,
    Option<&'static ReasoningSetting>,
);

/// Spawns the summary call `summarizing` of `agent` on `turn` to `spec`, by `route` when that is
/// not the agent's own model, holding `hold` when it is a turn's call; says whether it started.
fn start(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    (agent, turn, hold): (Entity, Entity, Option<Entity>),
    (route, spec, reasoning): (Option<Connection>, &ModelSpec, Option<Reasoning>),
    summarizing: Summarizing,
) -> bool {
    let request = match summarizing.request(spec, reasoning) {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
            return false;
        }
    };
    let model = ModelRequest {
        connection: route,
        ..ModelRequest::new(request)
    };
    let mut call = commands.spawn((
        Name::new(SUMMARIZING),
        summarizing,
        model,
        Effect::default().with_purpose(PURPOSE),
        ForAgent(agent),
        CallOf(turn),
    ));
    if let Some(hold) = hold {
        call.insert(HoldOf(hold));
    }
    true
}

/// The name of a summary call, which the agent's status shows.
const SUMMARIZING: &str = "summarizing older messages";

/// How a running summary call is cancelled, as a notice says it.
const CANCEL: &str = "stopping the turn (Esc) cancels it";

/// The context tokens of `history`, a request with the system prompt `prompt` and `tools`, as pi
/// counts them. Before a prompt (`before_prompt`), the context the last reply left: the usage the
/// provider reported for it. After tool results, that usage plus an estimate of the messages after
/// the reply. Without a usage that still describes the conversation (none reported, or a
/// compaction or an edit since, [`ContextStale`]), an estimate of the whole, before a prompt
/// without the messages after the last reply.
fn context_tokens(
    history: &rig_core::conversation::Conversation,
    (last, stale): (&LastUsage, Option<&ContextStale>),
    (prompt, tools): (Option<&str>, &[Arc<completion::ToolDefinition>]),
    before_prompt: bool,
) -> u64 {
    let messages = history.messages();
    let anchored = last
        .0
        .as_ref()
        .filter(|_| stale.is_none())
        .and_then(|anchor| {
            let tokens = anchor.usage.context_tokens().filter(|&tokens| tokens > 0)?;
            Some((tokens, history.index_of(anchor.through)?))
        });
    if let Some((tokens, at)) = anchored {
        if before_prompt {
            return tokens;
        }
        let after = messages.get(at.saturating_add(1)..).unwrap_or_default();
        return tokens.saturating_add(estimate_tokens(after));
    }
    let end = match before_prompt {
        true => last_reply(messages).map_or(0, |at| at.saturating_add(1)),
        false => messages.len(),
    };
    let prompt = prompt.map_or(0, |prompt| {
        estimate_tokens(&[completion::Message::system(prompt)])
    });
    let tools = tools
        .iter()
        .map(|tool| {
            serde_json::to_string(&**tool)
                .map_or(0, |json| (json.encode_utf16().count() as u64).div_ceil(4))
        })
        .fold(0, u64::saturating_add);
    let messages = messages.get(..end).unwrap_or_default();
    prompt
        .saturating_add(tools)
        .saturating_add(estimate_tokens(messages))
}

/// The index of the last reply among `messages`.
fn last_reply(messages: &[Arc<completion::Message>]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| matches!(&**message, completion::Message::Assistant(_)))
}

/// Whether `messages` end with a prompt rather than tool results: nothing after the last reply
/// answers a tool call.
fn before_prompt(messages: &[Arc<completion::Message>]) -> bool {
    let after = last_reply(messages).map_or(0, |at| at.saturating_add(1));
    let after = messages.get(after..).unwrap_or_default();
    !after.iter().any(|message| match &**message {
        completion::Message::User { content } => content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        _ => false,
    })
}

/// Before each turn's request, compacts when the context leaves less than the reserve of the
/// input limit of the model it goes to free: the request is held, and built again once the summary
/// is in. A request compacted for already, or whose compaction failed, goes as it is. A request
/// that goes asks for at most the room its input leaves in the window, as pi asks.
fn compact_near_the_window(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &Adapted,
            &CallOf,
            &ForAgent,
            Has<Recovering>,
        ),
        (
            Added<Admitted>,
            With<TurnCall>,
            Without<Started>,
            Without<Outcome>,
            Without<SupersededBy>,
        ),
    >,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let settings = default.0;
    for (call, mut model, Adapted(connection), &CallOf(turn), &ForAgent(agent), recovering) in
        &mut calls
    {
        let Ok((conversation, _, last, stale, reasoning)) = agents.get(agent) else {
            continue;
        };
        let spec = &*connection.spec;
        let prompt = model.prompt.as_ref().map(|prompt| prompt.text());
        let tools = if spec.tools {
            model.request.tools.clone()
        } else {
            Vec::new()
        };
        let history = model.history.as_ref().unwrap_or(conversation);
        let measure = (prompt.as_deref(), tools.as_slice());
        let sent = context_tokens(history, (last, stale), measure, false);
        let window = spec.input_limit().map(u64::from);
        if let Some(window) = window.filter(|_| !recovering) {
            let before = before_prompt(history.messages());
            let tokens = match before {
                true => context_tokens(history, (last, stale), measure, true),
                false => sent,
            };
            let reason = CompactReason::Threshold;
            let planned = settings
                .should_compact(tokens, window)
                .then(|| Summarizing::plan(conversation, settings.keep_recent_tokens, reason))
                .flatten();
            if let Some(summarizing) = planned {
                let route = (
                    model.connection.clone(),
                    spec,
                    reasoning.and_then(|it| it.0),
                );
                let at = (agent, turn, Some(call));
                if start(&mut commands, &mut notices, at, route, summarizing) {
                    continue;
                }
            }
        }
        let asked = model
            .request
            .max_tokens
            .or(spec.output_limit().map(u64::from));
        if let Some(asked) = asked {
            let fitted = clamp_max_tokens(spec, sent, asked);
            if model.request.max_tokens != Some(fitted) {
                model.request_mut().max_tokens = Some(fitted);
            }
        }
    }
}

/// Refuses each `/compact` summary call that nothing was planned for, nor a plugin planned anew:
/// "Nothing to compact yet."
fn nothing_to_compact(
    calls: Query<(Entity, &Summarizing, &ForAgent), (Added<Admitted>, ModelCalls)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, summarizing, &ForAgent(agent)) in &calls {
        if summarizing.messages.is_empty() {
            notices.write(Notice::info(agent, NOTHING));
            let why = ErrorReport::new(ErrorKind::Cancelled, NOTHING);
            commands.entity(call).try_insert_if_new(Outcome(Err(why)));
        }
    }
}

/// What `/compact` says when there is nothing to summarize.
const NOTHING: &str = "Nothing to compact yet.";

/// Compacts an idle agent's conversation on the user's request, in a turn of its own that ends
/// with the summary. With nothing to summarize, the call still goes to Prepare, where a plugin
/// that cuts its own way may plan it; otherwise it is refused there.
fn on_compact(
    compact: On<Compact>,
    agents: Query<(Compacting, Has<ActiveTurn>), With<Agent>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok(((conversation, connection, _, _, reasoning), busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::no_model(agent));
        return;
    };
    let reason = CompactReason::Asked {
        focus: compact.focus.clone(),
    };
    let keep = default.0.keep_recent_tokens;
    let summarizing = Summarizing::plan(conversation, keep, reason.clone());
    let summarizing = summarizing.unwrap_or_else(|| Summarizing::nothing(reason));
    let reasoning = reasoning.and_then(|it| it.0);
    compaction_turn(
        &mut commands,
        &mut notices,
        (agent, connection, reasoning),
        summarizing,
    );
}

/// Starts `summarizing` for `agent` on its `connection` in a turn of its own, which ends with the
/// summary.
fn compaction_turn(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    (agent, connection, reasoning): (Entity, &Connection, Option<Reasoning>),
    summarizing: Summarizing,
) {
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent)))
        .id();
    let request = if summarizing.messages.is_empty() {
        Ok(CompletionRequest::new(completion::Message::user(NOTHING)))
    } else {
        summarizing.request(&connection.spec, reasoning)
    };
    match request {
        Ok(request) => {
            commands.spawn((
                Name::new(SUMMARIZING),
                summarizing,
                ModelRequest::new(request),
                Effect::default().with_purpose(PURPOSE),
                ForAgent(agent),
                CallOf(turn),
            ));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
            commands.trigger(StopTurn::plugin_alone(agent, "compaction", "not compacted"));
        }
    }
}

/// After a turn answered, marks its agent to be compacted once idle when the usage its last reply
/// reported leaves less than the reserve free, as pi checks after each run.
fn check_after_turn(
    ended: On<Add<TurnOutcome>>,
    turns: Query<(&TurnOf, &TurnOutcome)>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), TurnOutcome::Answered(_))) = turns.get(ended.entity) else {
        return;
    };
    let Ok((_, Some(connection), last, None, _)) = agents.get(agent) else {
        return;
    };
    let window = connection.spec.input_limit().map(u64::from);
    let tokens = last
        .0
        .as_ref()
        .and_then(|anchor| anchor.usage.context_tokens());
    if let (Some(window), Some(tokens)) = (window, tokens)
        && default.0.should_compact(tokens, window)
    {
        commands.entity(agent).try_insert(CompactWhenIdle);
    }
}

/// Compacts each agent marked after its turn ([`CompactWhenIdle`]) once it is idle, in a turn of
/// its own. A turn that started first checks before its prompt instead.
fn compact_when_idle(
    agents: Query<(Entity, Compacting, Has<ActiveTurn>), With<CompactWhenIdle>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, (conversation, connection, _, _, reasoning), busy) in &agents {
        if busy {
            continue;
        }
        commands.entity(agent).try_remove::<CompactWhenIdle>();
        let keep = default.0.keep_recent_tokens;
        let planned = Summarizing::plan(conversation, keep, CompactReason::Threshold);
        if let (Some(summarizing), Some(connection)) = (planned, connection) {
            let reasoning = reasoning.and_then(|it| it.0);
            compaction_turn(
                &mut commands,
                &mut notices,
                (agent, connection, reasoning),
                summarizing,
            );
        }
    }
}

/// Marks an agent's context measure stale ([`ContextStale`]) once its conversation is edited
/// rather than added to, as pi recounts after an edit.
fn stale_after_edits(mut changed: MessageReader<ConversationChanged>, mut commands: Commands) {
    for ConversationChanged { agent, change } in changed.read() {
        let edit = matches!(change.change, Change::Splice { .. } | Change::Reset { .. });
        if edit && change.by != Some(summary_author()) {
            commands
                .entity(*agent)
                .try_insert_if_new(ContextStale::default());
        }
    }
}

/// After the model refused a turn's request as too long, or the window cut its reply short (it
/// stopped at its output limit below the model's own), compacts and sends the request again,
/// once: a second one fails the turn. Not when compaction is disabled.
fn recover_from_overflow(
    failed: Query<
        (
            Entity,
            &SentTo,
            &Outcome,
            Option<&RetryVerdict>,
            &CallOf,
            &ForAgent,
        ),
        (
            ReadyToSettle,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<OverflowTried>,
        ),
    >,
    started: Query<&HoldOf, With<Summarizing>>,
    ended: Query<(), With<TurnOutcome>>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !default.0.enabled {
        return;
    }
    for (call, SentTo(connection), outcome, verdict, &CallOf(turn), &ForAgent(agent)) in &failed {
        let overflow = verdict.is_some_and(|verdict| verdict.0 == Verdict::Overflow);
        let cut_short = outcome.completion().is_some_and(|reply| {
            let most = connection.spec.output_limit().map(u64::from).unwrap_or(0);
            let written = reply.usage.output_tokens.unwrap_or(0);
            reply.stop() == StopReason::Length && most > 0 && written < most
        });
        if !(overflow || cut_short)
            || ended.contains(turn)
            || started.iter().any(|&HoldOf(held)| held == call)
        {
            continue;
        }
        let Ok((conversation, _, _, _, reasoning)) = agents.get(agent) else {
            continue;
        };
        commands.entity(call).try_insert(OverflowTried);
        let keep = default.0.keep_recent_tokens;
        let Some(summarizing) = Summarizing::plan(conversation, keep, CompactReason::Overflow)
        else {
            continue;
        };
        let route = (
            Some(connection.clone()),
            &*connection.spec,
            reasoning.and_then(|it| it.0),
        );
        if start(
            &mut commands,
            &mut notices,
            (agent, turn, Some(call)),
            route,
            summarizing,
        ) {
            let situation = match overflow {
                true => OUTGREW,
                false => "The context window cut the reply short",
            };
            notices.write(Notice::warn(
                agent,
                format!("{situation}: summarizing older messages and sending it again; {CANCEL}."),
            ));
        }
    }
}

/// What the agent is told a compaction after an overflow is for.
const OUTGREW: &str = "The conversation outgrew the model's context window";

/// When a turn's call that carries a [`Recovering`] or [`OverflowTried`] is replaced, such as by a
/// retry, gives them to its successor.
fn carry_recovery(
    superseded: On<Add<SupersededBy>>,
    calls: Query<(&SupersededBy, Option<&Recovering>, Has<OverflowTried>)>,
    mut commands: Commands,
) {
    let Ok((by, step, tried)) = calls.get(superseded.entity) else {
        return;
    };
    let mut successor = commands.entity(by.successor());
    if let Some(&step) = step {
        successor.try_insert_if_new(step);
    }
    if tried {
        successor.try_insert_if_new(OverflowTried);
    }
}

/// Whether `report` reads as a passing provider or connection failure, by pi's patterns.
fn passing(report: &str) -> bool {
    static LIMIT: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r#"(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing|subscription_sharing_usage_limit_exceeded|subscription_sharing_workplace_individual_usage_limit_exceeded|"code"\s*:\s*"?1308(?:[^0-9]|$)"#,
        )
        .ok()
    });
    static PASSING: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"(?i)overloaded|server_busy|servers are currently busy|currently experiencing high demand|model is at capacity|rate.?limit|too many requests|(?:^|[^a-z0-9])(?:429|500|502|503|504|520|524)(?:[^a-z0-9]|$)|service.?(?:is.?)?(?:temporarily.?)?unavailable|server.?error|internal.?error|provider.?returned.?(?:an.?)?error|exceeded request buffer limit while retrying upstream|unable to forward upstream|upstream.?request.?failed|upstream_stream_read_error|stream was interrupted|http/?2 stream failed|network.?error|connection.?error|connection.?refused|connection.?lost|connection.?closed|other side closed|fetch failed|request to \S+ failed, reason:|getaddrinfo|(?:^|[^a-z0-9_])(?:ECONNRESET|ECONNABORTED|ECONNREFUSED|ETIMEDOUT|EPIPE)(?:[^a-z0-9_]|$)|connection reset by peer|ENOTFOUND|EAI_AGAIN|upstream.?connect|reset before headers|socket hang up|socket connection was closed|timed? out|timeout|terminated|websocket.?closed|websocket.?error|websocket.?connection.?limit.?reached|ended without|stream ended before message_stop|stream ended before a terminal response event|http2 request did not get a response|stream disconnected before completion|cut short after the reply had begun|pending stream has been canceled|retry delay|you can retry your request|try your request again|please retry your request|ResourceExhausted|subscription_sharing_usage_unavailable|subscription_sharing_user_unavailable",
        )
        .ok()
    });
    let limited = LIMIT.as_ref().is_some_and(|it| it.is_match(report));
    !limited && PASSING.as_ref().is_some_and(|it| it.is_match(report))
}

/// Whether a summary call that failed with `report` is sent again: a passing failure, by the
/// kernel's verdict or pi's patterns, never an overflow or a cancel.
fn retryable(report: &ErrorReport) -> bool {
    !report.is_context_overflow()
        && report.kind != ErrorKind::Cancelled
        && (report.is_retryable() || passing(&report.to_string()))
}

/// Sends a failed summary call again once the wait passed, when its failure passes and its turn
/// goes on, up to `[retry] max_retries` times for each summary call, as pi retries each: a hold
/// with the [`Backoff`] keeps it from settling until then, and the call sent again ([`resend`])
/// takes its marker and its hold on the call it is for.
fn retry_compaction_calls(
    mut failed: Query<
        (
            Entity,
            &Outcome,
            Option<&RetryVerdict>,
            &CallOf,
            &ForAgent,
            &mut Summarizing,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            Without<TurnCall>,
            Without<SupersededBy>,
        ),
    >,
    ended: Query<(), With<TurnOutcome>>,
    policy: Res<RetryPolicy>,
    mut commands: Commands,
) {
    for (call, Outcome(outcome), verdict, &CallOf(turn), &ForAgent(agent), mut summarizing) in
        &mut failed
    {
        let Err(report) = outcome else {
            continue;
        };
        if ended.contains(turn) || !retryable(report) || summarizing.retries >= policy.max_retries {
            continue;
        }
        let delay = match verdict.map(|verdict| &verdict.0) {
            Some(Verdict::Retry(delay)) => *delay,
            _ => policy.backoff(summarizing.retries),
        };
        summarizing.retries = summarizing.retries.saturating_add(1);
        let backoff = Backoff::after(summarizing.retries, delay);
        let tries = format!("{}/{}", backoff.attempt, policy.max_retries);
        let report = report.clone();
        commands.queue(move |world: &mut World| {
            world.write_message(Notice::warn(
                agent,
                format!(
                    "The compaction call failed: {report}. Retrying in {}s (retry {tries}); {CANCEL}.",
                    backoff.seconds_left(),
                ),
            ));
            let hold = world
                .spawn((Name::new("retry wait"), backoff, HoldOf(call)))
                .id();
            world
                .commands()
                .delayed()
                .duration(delay)
                .queue(move |world: &mut World| {
                    // Stopping the turn despawned the hold.
                    if world.get_entity(hold).is_err() {
                        return;
                    }
                    world.despawn(hold);
                    resend(call).apply(world);
                });
        });
    }
}

/// Spawns the call that supersedes `call`, a turn's model call held while the conversation was
/// compacted for it, built anew from the conversation with `step`, and [`OverflowTried`] when
/// `tried`, under the effect `parent`.
fn supersede(
    commands: &mut Commands,
    (turn, agent, call): (Entity, Entity, Entity),
    parent: Option<rig_core::effect::EffectId>,
    (step, tried): (Recovering, bool),
) {
    let mut successor = commands.spawn((
        TurnCall,
        CallOf(turn),
        ForAgent(agent),
        Effect::with_parent_id(parent),
        step,
    ));
    if tried {
        successor.insert(OverflowTried);
    }
    successor.add_related::<SupersededBy>(&[call]);
}

/// Marks each summary call as this plugin's.
fn provided_by_compaction(
    added: On<Add<Summarizing>>,
    me: Option<Res<PluginEntity<CompactionPlugin>>>,
    mut commands: Commands,
) {
    if let Some(me) = me {
        commands
            .entity(added.entity)
            .try_insert_if_new(ProvidedBy(me.entity()));
    }
}

/// On a summary call whose text was read from its reply, so a reviewer's edit stays.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct SummaryRead;

/// Reads each answered summary call's text from its reply, once.
fn read_replies(
    mut calls: Query<
        (Entity, &mut Summarizing, &Outcome),
        (
            ReadyToSettle,
            ModelCalls,
            Without<TurnCall>,
            Without<SummaryRead>,
        ),
    >,
    mut commands: Commands,
) {
    for (call, mut summarizing, outcome) in &mut calls {
        let label = summarizing.label();
        let reply = outcome.completion();
        summarizing.summary = reply.and_then(|reply| summary_text(reply, label).ok());
        commands.entity(call).try_insert(SummaryRead);
    }
}

/// Why a summary call's text went in nowhere.
enum NotTaken {
    /// A reviewer cancelled it, or its turn ended first.
    Cancelled,
    /// It failed, for this reason.
    Failed(String),
}

/// The text a summary call settled with, read from its reply or left by a reviewer, or why there
/// is none; a passing failure that used up its retries under `policy` says so.
fn taken(
    outcome: &Outcome,
    summarizing: &Summarizing,
    policy: &RetryPolicy,
) -> Result<String, NotTaken> {
    let label = summarizing.label();
    let response = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => {
            let most = policy.max_retries;
            let why = match retryable(report) && summarizing.retries >= most {
                true => format!(
                    "{report}. Not retrying: it failed {} times in a row (`[retry] max_retries` is \
                     {most})",
                    summarizing.retries.saturating_add(1)
                ),
                false => report.to_string(),
            };
            return Err(NotTaken::Failed(format!("{label} failed: {why}")));
        }
        (Ok(_), None) => {
            let why = "the model did not answer with a completion";
            return Err(NotTaken::Failed(format!("{label} failed: {why}")));
        }
        (Ok(_), Some(response)) => response,
    };
    match &summarizing.summary {
        Some(text) => Ok(text.clone()),
        // The reply had one, so a reviewer took it out.
        None => match summary_text(response, label) {
            Ok(_) => Err(NotTaken::Cancelled),
            Err(why) => Err(NotTaken::Failed(why.to_string())),
        },
    }
}

/// Takes a settled summary call. The history of a split turn starts the call for its earlier
/// part. A whole summary replaces the summarized messages in the conversation as an entry of its
/// own, a splice its log records, and the call it held is sent again, built anew. A failed
/// summary, one a reviewer cancelled, one whose turn ended first, or one whose kept message is
/// gone replaces nothing and says why; the call it held is sent as it is, or, after an overflow,
/// fails. The turn ends when the user asked for the compaction.
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(
        (&CallOf, &ForAgent),
        (&Summarizing, &ModelRequest, &Outcome),
        Option<&HoldOf>,
    )>,
    held: Query<&Effect>,
    ended: Query<(), With<TurnOutcome>>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Connection>,
        Option<&ReasoningSetting>,
    )>,
    policy: Res<RetryPolicy>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(((&CallOf(turn), &ForAgent(agent)), (summarizing, model, outcome), hold)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let Ok((mut conversation, own, reasoning)) = agents.get_mut(agent) else {
        return;
    };
    let ended = ended.contains(turn);
    if summarizing.messages.is_empty() {
        // A `/compact` with nothing to summarize, refused in Fit.
        if !ended {
            commands.trigger(StopTurn::plugin_alone(
                agent,
                "compaction",
                "nothing to compact",
            ));
        }
        return;
    }
    let text = match ended {
        true => Err(NotTaken::Cancelled),
        false => taken(outcome, summarizing, &policy),
    };
    let hold = hold.map(|&HoldOf(call)| call);
    let text = match text.map(|text| (summarizing.next(text.clone()), text)) {
        Ok((Some(next), _)) => {
            let route = model.connection.clone();
            let spec = route
                .as_ref()
                .or(own)
                .map(|connection| connection.spec.clone());
            let reasoning = reasoning.and_then(|it| it.0);
            if let Some(spec) = spec
                && start(
                    &mut commands,
                    &mut notices,
                    (agent, turn, hold),
                    (route, &spec, reasoning),
                    next,
                )
            {
                return;
            }
            Err(NotTaken::Failed(
                "the turn's earlier part could not be summarized".to_owned(),
            ))
        }
        Ok((None, text)) => Ok(text),
        Err(why) => Err(why),
    };
    let revision = conversation.revision();
    let end = match summarizing.keep_from {
        None => Some(conversation.messages().len()),
        Some(kept) => conversation.index_of(kept),
    };
    let summary = text.and_then(|text| {
        let gone = || NotTaken::Failed("the messages it kept were taken out".to_owned());
        let end = end.ok_or_else(gone)?;
        let messages = conversation.messages();
        let older = messages.get(..end).unwrap_or_default();
        let older = (older.len(), estimate_tokens(older));
        let kept = messages.get(end..).unwrap_or_default();
        let kept = (kept.len(), estimate_tokens(kept));
        let summary = match &summarizing.history {
            Some(history) => merge_split_turn(history, &text),
            None => text,
        };
        let state = SummaryState {
            summary,
            ..summarizing.state.clone()
        };
        #[cfg(feature = "tui")]
        let before = tui::Before::of(&conversation, end);
        match apply_summary(&mut conversation, summarizing.keep_from, &state) {
            #[cfg(feature = "tui")]
            Some(spliced) => {
                before.keep(&mut commands, agent, &spliced);
                Ok((older, kept))
            }
            #[cfg(not(feature = "tui"))]
            Some(_) => Ok((older, kept)),
            None => Err(gone()),
        }
    });
    let mut estimate = None;
    let step = match summary {
        Ok((older, kept)) => {
            let left = estimate_tokens(conversation.messages());
            notices.write(Notice::info(agent, compacted_text(older, kept, left)));
            let through = conversation.entries().last().map(|entry| entry.id);
            estimate = through.map(|through| ContextEstimate {
                tokens: left,
                through,
            });
            Recovering::Compacted
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
            Recovering::NotSummarized
        }
        Err(NotTaken::Failed(why)) => {
            notices.write(Notice::error(
                agent,
                format!("{why}; the conversation was not compacted."),
            ));
            Recovering::NotSummarized
        }
    };
    if conversation.revision() != revision {
        commands.entity(agent).try_insert(ContextStale { estimate });
    }
    let overflowed = summarizing.reason == CompactReason::Overflow;
    match hold {
        // Built again unchanged, the refused call would be refused again: it settles as it is.
        Some(_) if step == Recovering::NotSummarized && overflowed => {}
        Some(call) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            supersede(
                &mut commands,
                (turn, agent, call),
                parent,
                (step, overflowed),
            );
        }
        // The user's own compaction turn ends with it, unless it ended.
        None if !ended => {
            commands.trigger(StopTurn::plugin_alone(agent, "compaction", "compacted"))
        }
        None => {}
    }
}

/// Says that `older` messages of so many tokens were summarized, `kept` were kept, and the
/// conversation now holds about `left` tokens.
fn compacted_text(
    (older, older_tokens): (usize, u64),
    (kept, kept_tokens): (usize, u64),
    left: u64,
) -> String {
    let plural = |count: usize| if count == 1 { "message" } else { "messages" };
    format!(
        "Compacted {older} {} (about {} tokens) into a summary and kept the newest {kept} {} \
         (about {} tokens); the conversation now holds about {} tokens.",
        plural(older),
        tokens_label(older_tokens),
        plural(kept),
        tokens_label(kept_tokens),
        tokens_label(left)
    )
}

/// Undoes the agent's newest compaction on [`Uncompact`].
fn on_uncompact(
    uncompact: On<Uncompact>,
    mut agents: Query<(&AgentId, &mut Conversation, Has<ActiveTurn>)>,
    store: Option<Res<SessionStore>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = uncompact.entity;
    let Ok((id, mut conversation, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(store) = store else {
        let why = "Cannot undo the compaction: the session is not saved.";
        notices.write(Notice::error(agent, why));
        return;
    };
    match undo_summary(&mut conversation, &*store.0, &id.0, summary_author()) {
        Ok(undone) => {
            commands.entity(agent).try_insert(ContextStale::default());
            let left = estimate_tokens(conversation.messages());
            let count = undone.restored.len();
            let text = format!(
                "Put back the {count} {} the newest compaction summarized; the model now gets \
                 about {} tokens of conversation.",
                if count == 1 { "entry" } else { "entries" },
                tokens_label(left)
            );
            notices.write(Notice::info(agent, text));
        }
        Err(why) => {
            let text = format!("Cannot undo the compaction: {why}.");
            notices.write(Notice::error(agent, text));
        }
    }
}

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;
