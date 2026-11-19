//! Compaction: keeps each agent's conversation inside its model's input limit by summarizing its
//! older messages, as the pi coding agent does. Before each turn's request, the first and each
//! one after tool results, it checks whether the context leaves less than `[compaction]
//! reserve_tokens` of the input limit free: the usage the provider reported for the last reply,
//! plus an estimate of the messages after it. After a request refused as too long it compacts
//! and sends the request again, once. `/compact [instructions]` compacts at once.
//!
//! A compaction keeps about `keep_recent_tokens` of the newest messages, never cutting between a
//! tool call and its result, and summarizes the older ones with a separate call to the agent's
//! model ([`Summarizing`]): the previous summary is updated with the new messages, and a turn the
//! cut splits gets a second summary of its earlier part. rig-memory's [`prepare_compaction`]
//! holds the cut, the prompts and the requests. A summary that failed or was cut short replaces
//! nothing and says why. Every compaction keeps what it replaced, and [`Uncompact`] puts the
//! newest one back.
//!
//! A summary call's text is in [`Summarizing::summary`] in [`ReviewSystems::Act`], where a
//! reviewer may edit it or set it to `None` to cancel. Another compaction is a plugin that
//! replaces this one.

use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, UnsupportedOption, tokens_label};
use rig_ecs::prelude::*;
use rig_ecs::retry::RetryPolicy;
use rig_ecs::session_log::SessionStore;
use rig_harness::AppKeepsExt;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::settings::AppSettingsExt;
use rig_harness::successor::resend;
use rig_harness::{PluginEntity, ProvidedBy};
pub use rig_memory::CutPoint;
use rig_memory::{
    CompactionPreparation, CompactionSettings, SummaryState, apply_summary, estimate_tokens,
    merge_split_turn, no_history, prepare_compaction, prepare_compaction_at, summary_request,
    summary_text, turn_prefix_request, undo_summary,
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
            .add_systems(
                AgentPass,
                (
                    compact_near_the_window.in_set(PrepareSystems::Fit),
                    read_replies.in_set(ReviewSystems::Edit),
                    (retry_compaction_calls, recover_from_overflow).in_set(ReviewSystems::Act),
                ),
            )
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
/// compacted for again, and an overflow fails the turn. A call that replaces it, such as a retry,
/// carries it.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Component, Clone, Debug)]
pub enum Recovering {
    /// Its compaction failed or was cancelled.
    NotSummarized,
    /// The conversation was compacted for it.
    Compacted,
}

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
    /// The state the summary goes into: the previous summary until it is taken, and the files
    /// listed after it.
    pub state: SummaryState,
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
}

impl Summarizing {
    /// The first summary call of a compaction of `conversation` for `reason`, keeping about
    /// `keep_recent_tokens`; `None` when nothing would be summarized.
    pub fn plan(
        conversation: &rig_core::conversation::Conversation,
        keep_recent_tokens: u64,
        reason: CompactReason,
    ) -> Option<Self> {
        let prepared = prepare_compaction(conversation, keep_recent_tokens)?;
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
        let mut state = prepared.previous;
        let previous = state.previous_summary();
        state.read_files = prepared.files.read_files;
        state.modified_files = prepared.files.modified_files;
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
            messages: messages.into(),
            turn_prefix,
            history,
            summary: None,
        }
    }

    /// What this call's reply is called in a failure: the history summary or the turn prefix's.
    fn label(&self) -> &'static str {
        match self.history {
            Some(_) => "Turn prefix summarization",
            None => "Summarization",
        }
    }

    /// The request of this call to `spec`.
    ///
    /// # Errors
    ///
    /// When `spec` does not take the request's options.
    pub fn request(&self, spec: &ModelSpec) -> Result<CompletionRequest, UnsupportedOption> {
        let focus = self.reason.focus();
        match self.history {
            Some(_) => turn_prefix_request(&self.messages, focus, spec),
            None => {
                let previous = self.state.previous_summary();
                summary_request(&self.messages, previous.as_deref(), focus, spec)
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
);

/// Spawns the summary call `summarizing` of `agent` on `turn` to `spec`, by `route` when that is
/// not the agent's own model, holding `hold` when it is a turn's call; says whether it started.
fn start(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    (agent, turn, hold): (Entity, Entity, Option<Entity>),
    (route, spec): (Option<Connection>, &ModelSpec),
    summarizing: Summarizing,
) -> bool {
    let request = match summarizing.request(spec) {
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

/// The context tokens of `conversation` with the system prompt `prompt` and `tools`: the usage
/// the provider reported for the last reply, or the estimate a compaction left, plus an estimate
/// of the messages after it; with neither, or once the conversation changed under them, an
/// estimate of the whole.
fn context_tokens(
    conversation: &Conversation,
    (last, stale): (&LastUsage, Option<&ContextStale>),
    (prompt, tools): (Option<&str>, &[Arc<completion::ToolDefinition>]),
) -> u64 {
    let anchored = match stale {
        Some(stale) => stale
            .estimate
            .map(|estimate| (estimate.tokens, estimate.through)),
        None => last.0.as_ref().and_then(|anchor| {
            let tokens = anchor.usage.context_tokens().filter(|&tokens| tokens > 0)?;
            Some((tokens, anchor.through))
        }),
    };
    let messages = conversation.messages();
    if let Some((tokens, at)) =
        anchored.and_then(|(tokens, through)| Some((tokens, conversation.index_of(through)?)))
    {
        let after = messages.get(at.saturating_add(1)..).unwrap_or_default();
        return tokens.saturating_add(estimate_tokens(after));
    }
    let prompt = prompt.map_or(0, |prompt| {
        estimate_tokens(&[completion::Message::system(prompt)])
    });
    let tools = tools
        .iter()
        .map(|tool| serde_json::to_string(&**tool).map_or(0, |json| json.len() as u64 / 4))
        .fold(0, u64::saturating_add);
    prompt
        .saturating_add(tools)
        .saturating_add(estimate_tokens(messages))
}

/// Before each turn's request, compacts when the context leaves less than the reserve of the
/// input limit of the model it goes to free: the request is held, and built again once the summary
/// is in. A request compacted for already, or whose compaction failed, goes as it is.
fn compact_near_the_window(
    calls: Query<
        (Entity, &ModelRequest, &Adapted, &CallOf, &ForAgent),
        (
            Added<Admitted>,
            With<TurnCall>,
            Without<Recovering>,
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
    for (call, model, Adapted(connection), &CallOf(turn), &ForAgent(agent)) in &calls {
        let Ok((conversation, _, last, stale)) = agents.get(agent) else {
            continue;
        };
        let spec = &*connection.spec;
        let Some(window) = spec.input_limit() else {
            continue;
        };
        let prompt = model.prompt.as_ref().map(|prompt| prompt.text());
        let tools = if spec.tools {
            model.request.tools.as_slice()
        } else {
            &[]
        };
        let tokens = context_tokens(conversation, (last, stale), (prompt.as_deref(), tools));
        if !settings.should_compact(tokens, u64::from(window)) {
            continue;
        }
        let reason = CompactReason::Threshold;
        let Some(summarizing) =
            Summarizing::plan(conversation, settings.keep_recent_tokens, reason)
        else {
            continue;
        };
        let route = (model.connection.clone(), spec);
        start(
            &mut commands,
            &mut notices,
            (agent, turn, Some(call)),
            route,
            summarizing,
        );
    }
}

/// Compacts an idle agent's conversation on the user's request, in a turn of its own that ends
/// with the summary.
fn on_compact(
    compact: On<Compact>,
    agents: Query<(Compacting, Has<ActiveTurn>), With<Agent>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok(((conversation, connection, ..), busy)) = agents.get(agent) else {
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
    let Some(summarizing) = Summarizing::plan(conversation, keep, reason) else {
        notices.write(Notice::info(agent, "Nothing to compact yet."));
        return;
    };
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent)))
        .id();
    let route = (None, &*connection.spec);
    if !start(
        &mut commands,
        &mut notices,
        (agent, turn, None),
        route,
        summarizing,
    ) {
        commands.trigger(StopTurn::plugin_alone(agent, "compaction", "not compacted"));
    }
}

/// After the model refused a turn's request as too long, compacts and sends it again, once: a
/// call compacted for already, or whose compaction failed, fails the turn.
fn recover_from_overflow(
    failed: Query<
        (Entity, &SentTo, &RetryVerdict, &CallOf, &ForAgent),
        (
            ReadyToSettle,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<Recovering>,
        ),
    >,
    started: Query<&HoldOf, With<Summarizing>>,
    ended: Query<(), With<TurnOutcome>>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, SentTo(connection), verdict, &CallOf(turn), &ForAgent(agent)) in &failed {
        if verdict.0 != Verdict::Overflow
            || ended.contains(turn)
            || started.iter().any(|&HoldOf(held)| held == call)
        {
            continue;
        }
        let Ok((conversation, ..)) = agents.get(agent) else {
            continue;
        };
        let keep = default.0.keep_recent_tokens;
        let Some(summarizing) = Summarizing::plan(conversation, keep, CompactReason::Overflow)
        else {
            continue;
        };
        let route = (Some(connection.clone()), &*connection.spec);
        if start(
            &mut commands,
            &mut notices,
            (agent, turn, Some(call)),
            route,
            summarizing,
        ) {
            notices.write(Notice::warn(
                agent,
                format!(
                    "{OUTGREW}: summarizing its older messages and sending it again; {CANCEL}."
                ),
            ));
        }
    }
}

/// What the agent is told a compaction after an overflow is for.
const OUTGREW: &str = "The conversation outgrew the model's context window";

/// When a turn's call that carries a [`Recovering`] is replaced, such as by a retry, gives it to
/// its successor, unless that has one of its own.
fn carry_recovery(
    superseded: On<Add<SupersededBy>>,
    calls: Query<(&SupersededBy, &Recovering)>,
    mut commands: Commands,
) {
    if let Ok((by, &step)) = calls.get(superseded.entity) {
        commands.entity(by.successor()).try_insert_if_new(step);
    }
}

/// Sends a failed summary call again once the wait passed, when its verdict is a retry and its
/// turn goes on: a hold with the [`Backoff`] keeps it from settling until then, and the call sent
/// again ([`resend`]) takes its marker and its hold on the call it is for. The retry counts toward
/// its turn's [`Recovery`], so it waits and gives up as the turn's calls do.
fn retry_compaction_calls(
    failed: Query<
        (Entity, &Outcome, &RetryVerdict, &CallOf, &ForAgent),
        (
            ReadyToSettle,
            ModelCalls,
            Without<TurnCall>,
            Without<SupersededBy>,
            With<Summarizing>,
        ),
    >,
    ended: Query<(), With<TurnOutcome>>,
    policy: Res<RetryPolicy>,
    mut commands: Commands,
) {
    for (call, Outcome(outcome), RetryVerdict(verdict), &CallOf(turn), &ForAgent(agent)) in &failed
    {
        let (Err(report), Verdict::Retry(delay)) = (outcome, verdict) else {
            continue;
        };
        if ended.contains(turn) {
            continue;
        }
        let policy = *policy;
        let (report, delay) = (report.clone(), *delay);
        // Counted on the world, as other plugins move the turn's count too.
        commands.queue(move |world: &mut World| {
            let retries = world
                .get::<Recovery>(turn)
                .map_or(0, |recovery| recovery.retries)
                .saturating_add(1);
            let Ok(mut on) = world.get_entity_mut(turn) else {
                return;
            };
            on.insert(Recovery { retries });
            let backoff = Backoff::after(retries, delay);
            let tries = format!("{}/{}", backoff.attempt, policy.max_retries);
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
/// compacted for it, built anew from the conversation with `step`, under the effect `parent`.
fn supersede(
    commands: &mut Commands,
    (turn, agent, call): (Entity, Entity, Entity),
    parent: Option<rig_core::effect::EffectId>,
    step: Recovering,
) {
    let mut successor = commands.spawn((
        TurnCall,
        CallOf(turn),
        ForAgent(agent),
        Effect::with_parent_id(parent),
        step,
    ));
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
/// is none.
fn taken(
    (outcome, verdict): (&Outcome, Option<&RetryVerdict>),
    summarizing: &Summarizing,
) -> Result<String, NotTaken> {
    let label = summarizing.label();
    let response = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => {
            let why = match verdict.map(|verdict| &verdict.0) {
                Some(Verdict::GaveUp(why)) => format!("{report}. Not retrying: {why}"),
                _ => report.to_string(),
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
        Some(text) if text.trim().is_empty() => Err(NotTaken::Failed(format!(
            "{label} failed: the summary was empty"
        ))),
        Some(text) => Ok(text.clone()),
        // The reply had one, so a reviewer took it out.
        None => match summary_text(response, label) {
            Ok(_) => Err(NotTaken::Cancelled),
            Err(why) => Err(NotTaken::Failed(why.to_string())),
        },
    }
}

/// What the figure counting `conversation` holds besides its messages, such as the system prompt
/// and the tools: a stale one's estimate, else the last reply's, less the estimate of the entries
/// it counts. `None` when no figure counts it.
fn counted_around(
    conversation: &Conversation,
    last: Option<&LastUsage>,
    stale: Option<&ContextStale>,
) -> Option<u64> {
    let (tokens, through) = match stale {
        Some(stale) => stale
            .estimate
            .map(|estimate| (estimate.tokens, estimate.through))?,
        None => last
            .and_then(|last| last.0.as_ref())
            .and_then(|anchor| Some((anchor.usage.context_tokens()?, anchor.through)))?,
    };
    let at = conversation.index_of(through)?;
    let counted = conversation.messages().get(..=at).unwrap_or_default();
    Some(tokens.saturating_sub(estimate_tokens(counted)))
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
        (&Summarizing, &ModelRequest, &Outcome, Option<&RetryVerdict>),
        Option<&HoldOf>,
    )>,
    held: Query<&Effect>,
    ended: Query<(), With<TurnOutcome>>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Connection>,
        Option<&LastUsage>,
        Option<&ContextStale>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(((&CallOf(turn), &ForAgent(agent)), (summarizing, model, outcome, verdict), hold)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let Ok((mut conversation, own, last, stale)) = agents.get_mut(agent) else {
        return;
    };
    let ended = ended.contains(turn);
    let text = match ended {
        true => Err(NotTaken::Cancelled),
        false => taken((outcome, verdict), summarizing),
    };
    let hold = hold.map(|&HoldOf(call)| call);
    let text = match text.map(|text| (summarizing.next(text.clone()), text)) {
        Ok((Some(next), _)) => {
            let route = model.connection.clone();
            let spec = route
                .as_ref()
                .or(own)
                .map(|connection| connection.spec.clone());
            if let Some(spec) = spec
                && start(
                    &mut commands,
                    &mut notices,
                    (agent, turn, hold),
                    (route, &spec),
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
    let around = counted_around(&conversation, last, stale);
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
            let left = around
                .unwrap_or_default()
                .saturating_add(estimate_tokens(conversation.messages()));
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
    match hold {
        // Built again unchanged, the refused call would be refused again: it fails as it is.
        Some(call)
            if step == Recovering::NotSummarized
                && summarizing.reason == CompactReason::Overflow =>
        {
            commands.entity(call).try_insert(step);
        }
        Some(call) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            supersede(&mut commands, (turn, agent, call), parent, step);
        }
        // The user's own compaction turn ends with it, unless it ended.
        None if !ended => {
            commands.trigger(StopTurn::plugin_alone(agent, "compaction", "compacted"))
        }
        None => {}
    }
}

/// Says that `older` messages of so many tokens were summarized, `kept` were kept, and the model
/// now gets about `left` tokens of context.
fn compacted_text(
    (older, older_tokens): (usize, u64),
    (kept, kept_tokens): (usize, u64),
    left: u64,
) -> String {
    let plural = |count: usize| if count == 1 { "message" } else { "messages" };
    format!(
        "Compacted {older} {} (about {} tokens) into a summary and kept the newest {kept} {} \
         (about {} tokens); the model now gets about {} tokens of context.",
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
