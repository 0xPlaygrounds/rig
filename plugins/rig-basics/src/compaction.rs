//! Compaction: keeps each agent's conversation inside its model's input limit by summarizing its
//! older messages, as the pi coding agent does. When enabled, it checks the context against
//! `[compaction] reserve_tokens` of the context window at the end of each run, before a prompt is
//! sent (the context before it) and before each request after tool results. After a request
//! refused as too long, or a reply the window cut short, it compacts and sends the request again,
//! once.
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
use rig_core::message::StopReason;
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
    Anchor, CompactionPreparation, CompactionSettings, SummaryState, apply_summary,
    clamp_max_tokens, estimate_tokens, find_cut_point, merge_split_turn, no_history,
    prepare_compaction, prepare_compaction_at, request_tokens, summary_request, summary_text,
    turn_prefix_request, undo_summary,
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
                    (compact_near_the_window, nothing_to_compact).in_set(PrepareSystems::Fit),
                    check_after_run.in_set(ReviewSystems::Act),
                    read_replies.in_set(ReviewSystems::Edit),
                    (retry_compaction_calls, recover_from_overflow).in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(
                Update,
                (
                    stale_after_edits,
                    record_usage,
                    compact_after_error,
                ),
            )
            .add_observer(on_delivered)
            .add_observer(check_after_error)
            .add_observer(take_summary)
            .add_observer(follow_successor::<Summarizing>)
            .add_observer(follow_successor::<SummaryRetries>)
            .add_observer(follow_successor::<AfterRun>)
            .add_observer(repoint_waiters)
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
    /// The context leaves less than the reserve of the model's context window free.
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
}

/// How many times a summary call was sent again after a passing failure; each summary call
/// counts its own. A call sent again takes it.
#[derive(Component, Clone, Copy, Debug, Default)]
struct SummaryRetries(u32);

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
/// not the agent's own model, holding `hold` when it is a turn's call; the call, when it started.
fn start(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    (agent, turn, hold): (Entity, Entity, Option<Entity>),
    (route, spec, reasoning): (Option<Connection>, &ModelSpec, Option<Reasoning>),
    summarizing: Summarizing,
) -> Option<Entity> {
    let request = match summarizing.request(spec, reasoning) {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
            return None;
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
    Some(call.id())
}

/// The name of a summary call, which the agent's status shows.
const SUMMARIZING: &str = "summarizing older messages";

/// How a running summary call is cancelled, as a notice says it.
const CANCEL: &str = "stopping the turn (Esc) cancels it";

/// The anchor of `history`'s context: the tokens the provider reported for its last valid reply and
/// that reply's index, while it still describes it (no compaction or edit since, [`ContextStale`]).
fn anchor(
    history: &rig_core::conversation::Conversation,
    (last, stale): (&LastUsage, Option<&ContextStale>),
) -> Option<(u64, usize)> {
    let anchor = last.0.as_ref().filter(|_| stale.is_none())?;
    let tokens = anchor.usage.context_tokens().filter(|&tokens| tokens > 0)?;
    Some((tokens, history.index_of(anchor.through)?))
}

/// The context tokens of `history`'s first `end` messages as pi's compaction counts them: the
/// [`anchor`] plus an estimate of the messages after it, else an estimate of the messages alone.
fn context_tokens(
    history: &rig_core::conversation::Conversation,
    measure: (&LastUsage, Option<&ContextStale>),
    end: usize,
) -> u64 {
    let messages = history.messages().get(..end).unwrap_or_default();
    match anchor(history, measure).filter(|&(_, at)| at < end) {
        Some((tokens, at)) => {
            let after = messages.get(at.saturating_add(1)..).unwrap_or_default();
            tokens.saturating_add(estimate_tokens(after))
        }
        None => estimate_tokens(messages),
    }
}

/// The window the threshold compares against: the model's context window, as pi's, else its input
/// limit.
fn window(spec: &ModelSpec) -> Option<u64> {
    spec.context_window.or(spec.input_limit()).map(u64::from)
}

/// The threshold compaction of `conversation`, whose messages from `end` on are a pending prompt
/// that is neither counted nor cut, when its context leaves less than the reserve of `spec`'s
/// window free; `None` otherwise or with nothing to summarize.
fn over_threshold(
    conversation: &Conversation,
    measure: (&LastUsage, Option<&ContextStale>),
    (settings, spec): (&CompactionSettings, &ModelSpec),
    end: usize,
) -> Option<Summarizing> {
    let window = window(spec)?;
    let tokens = context_tokens(conversation, measure, end);
    if !settings.should_compact(tokens, window) {
        return None;
    }
    let messages = conversation.messages();
    let start = summarized_prefix(conversation);
    let cut = find_cut_point(messages, (start, end), settings.keep_recent_tokens, false);
    Summarizing::plan_at(conversation, cut, CompactReason::Threshold)
}

/// Before each turn's request, compacts when the context leaves less than the reserve of the
/// window of the model it goes to free, as pi checks before a prompt (the context before it, the
/// prompt neither counted nor cut) and before each request after tool results: the request is held,
/// and built again once the summary is in. A request compacted for already, or whose compaction
/// failed, goes as it is. A request that goes asks for at most the room its input leaves in the
/// window, by pi-ai's estimate and clamp.
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
    prompts: Query<&Prompt>,
    usages: Query<&Usages>,
    after_runs: Query<(Entity, &CallOf), (With<AfterRun>, Without<Settled>)>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let settings = default.0;
    for (call, mut model, Adapted(connection), &CallOf(turn), &ForAgent(agent), recovering) in
        &mut calls
    {
        if let Some((summary, _)) = after_runs.iter().find(|(_, of)| of.0 == turn) {
            commands.spawn((Name::new(AFTER_RUN), WaitsFor(summary), HoldOf(call)));
            continue;
        }
        let Ok((conversation, _, last, stale, reasoning)) = agents.get(agent) else {
            continue;
        };
        // The route as the request names it, before adapters wrapped it, and the spec it went to.
        commands.entity(turn).try_insert(RanOn {
            route: model.connection.clone(),
            spec: connection.spec.clone(),
        });
        // A request of a new turn is checked here, so a check left from a failed one is moot.
        commands.entity(agent).try_remove::<CheckAfterError>();
        let spec = &*connection.spec;
        let messages = conversation.messages();
        // Before the turn's first reply, the message its prompt delivered is pending; whatever
        // came before it is context.
        let end = prompts
            .get(turn)
            .ok()
            .and_then(|&Prompt(id)| conversation.index_of(id))
            .filter(|&at| {
                !messages
                    .get(at..)
                    .unwrap_or_default()
                    .iter()
                    .any(|message| matches!(&**message, completion::Message::Assistant(_)))
            })
            .unwrap_or(messages.len());
        let measure = (last, stale);
        if !recovering
            && let Some(summarizing) = over_threshold(conversation, measure, (&settings, spec), end)
        {
            let route = (
                model.connection.clone(),
                spec,
                reasoning.and_then(|it| it.0),
            );
            if start(
                &mut commands,
                &mut notices,
                (agent, turn, Some(call)),
                route,
                summarizing,
            )
            .is_some()
            {
                continue;
            }
        }
        let asked = model
            .request
            .max_tokens
            .or(spec.output_limit().map(u64::from));
        if let Some(asked) = asked {
            let prompt = model.prompt.as_ref().map(|prompt| prompt.text());
            let tools = spec
                .tools
                .then(|| serde_json::to_string(&model.request.tools).ok())
                .flatten()
                .filter(|_| !model.request.tools.is_empty());
            let history = model.history.as_ref().unwrap_or(conversation);
            let reported = usages.get(agent).ok().map(|it| it.0.as_slice());
            let anchor = newest_valid(
                history,
                last.0
                    .iter()
                    .chain(reported.unwrap_or_default().iter().rev()),
            );
            let input = request_tokens(
                (prompt.as_deref(), tools.as_deref()),
                history.messages(),
                anchor,
            );
            let fitted = clamp_max_tokens(spec, input, asked);
            if model.request.max_tokens != Some(fitted) {
                model.request_mut().max_tokens = Some(fitted);
            }
        }
    }
}

/// On a turn: the id of the message its prompt delivered, the first message delivered into it.
#[derive(Component)]
struct Prompt(rig_core::conversation::MessageId);

/// Marks each turn with its [`Prompt`], and, as pi resets its recovery on each user message, with
/// the newest message of the user's delivered into it ([`LastPrompt`]), which lets it be recovered
/// and checked again ([`Recovered`]).
fn on_delivered(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&DeliveredIn, &DeliveredAs, &Incoming)>,
    mut commands: Commands,
) {
    let Ok((&DeliveredIn(turn), &DeliveredAs(id), message)) = messages.get(delivered.entity) else {
        return;
    };
    let mut turn = commands.entity(turn);
    turn.try_insert_if_new(Prompt(id));
    if message.author == Author::user() {
        turn.try_insert(LastPrompt(id));
    }
}

/// The usage each reply of an agent reported, newest last, as pi-ai reads every reply's usage.
#[derive(Component, Default)]
struct Usages(Vec<Anchor>);

impl Usages {
    /// How many reports are kept.
    const KEPT: usize = 64;
}

/// The first of `anchors`, newest first, still describing the start of `history`, as pi-ai's
/// estimate takes the newest valid usage: its reply is there, no newer message precedes it and it
/// counted tokens. Its tokens and the reply's index.
fn newest_valid<'a>(
    history: &rig_core::conversation::Conversation,
    mut anchors: impl Iterator<Item = &'a Anchor>,
) -> Option<(u64, usize)> {
    anchors.find_map(|anchor| {
        let tokens = anchor.usage.context_tokens().filter(|&tokens| tokens > 0)?;
        let at = history.index_of(anchor.through)?;
        let through = history.id(at);
        (0..at)
            .all(|before| history.id(before) <= through)
            .then_some((tokens, at))
    })
}

/// Keeps each agent's [`Usages`] as its replies report them.
fn record_usage(
    agents: Query<(Entity, &LastUsage, Option<&mut Usages>), Changed<LastUsage>>,
    mut commands: Commands,
) {
    for (agent, last, usages) in agents {
        let Some(anchor) = last.0 else {
            continue;
        };
        match usages {
            Some(mut usages) => {
                if usages
                    .0
                    .last()
                    .is_some_and(|it| it.through == anchor.through)
                {
                    usages.0.pop();
                }
                usages.0.push(anchor);
                let over = usages.0.len().saturating_sub(Usages::KEPT);
                usages.0.drain(..over);
            }
            None => {
                commands.entity(agent).try_insert(Usages(vec![anchor]));
            }
        }
    }
}

/// On a summary call made after a run, as pi checks at the end of each run: its turn waits for it.
#[derive(Component)]
struct AfterRun;

/// A hold on a turn's proposed answer, or on a request the turn makes meanwhile, until the summary
/// call made after the run (the entity it names) is in; a request held so is built again then.
#[derive(Component)]
struct WaitsFor(Entity);

/// When a summary call made after a run is sent again, such as by a retry, what waited for it
/// waits for its successor.
fn repoint_waiters(
    superseded: On<Add<SupersededBy>>,
    calls: Query<&SupersededBy, With<AfterRun>>,
    waiters: Query<(Entity, &WaitsFor)>,
    mut commands: Commands,
) {
    let Ok(by) = calls.get(superseded.entity) else {
        return;
    };
    let next = by.successor();
    for (waiter, &WaitsFor(summary)) in &waiters {
        if summary == superseded.entity {
            commands.entity(waiter).try_insert(WaitsFor(next));
        }
    }
}

/// Where a turn's last request went, which the check at the end of its run measures against and
/// sends its summary to, as pi uses the model that wrote the reply: the route the request named
/// (`None` for the agent's own connection), before adapters wrapped it, and the model's spec.
#[derive(Component, Clone)]
struct RanOn {
    route: Option<Connection>,
    spec: Arc<ModelSpec>,
}

/// On a turn whose request was recovered from an overflow or a reply cut short, or tried to be,
/// with the turn's [`LastPrompt`] then: its run's end is not checked until another message is
/// delivered into it, as pi's check returns after that recovery and resets on each user message.
#[derive(Component)]
struct Recovered(Option<rig_core::conversation::MessageId>);

/// On a turn: the id of the newest message of the user's delivered into it.
#[derive(Component)]
struct LastPrompt(rig_core::conversation::MessageId);

/// Marks `turn` [`Recovered`] as of its newest prompt.
fn mark_recovered(commands: &mut Commands, turn: Entity) {
    commands.queue(move |world: &mut World| {
        let last = world.get::<LastPrompt>(turn).map(|it| it.0);
        if let Ok(mut turn) = world.get_entity_mut(turn) {
            turn.insert(Recovered(last));
        }
    });
}

/// The name of the hold on a turn while its summary after the run is written.
const AFTER_RUN: &str = "compacting after the run";

/// After a turn's answer is proposed, compacts when the context (the usage its last reply
/// reported, plus an estimate of what follows) leaves less than the reserve of the window of the
/// model its last request went to free, as pi checks at the end of each run: the answer is held
/// until the summary is in. Not after a recovery from an overflow or a reply cut
/// short ([`Recovered`]). A message accepted meanwhile carries the turn on, as
/// pi's queue drains after its compaction: the request it makes waits for the summary and is built
/// again then ([`WaitsFor`]).
fn check_after_run(
    turns: Query<
        (
            (Entity, &TurnOf, &TurnOutcome, Option<&RanOn>),
            (Option<&Recovered>, Option<&LastPrompt>),
        ),
        Changed<TurnOutcome>,
    >,
    running: Query<&CallOf, (With<Summarizing>, Without<Settled>)>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for ((turn, &TurnOf(agent), outcome, ran_on), (recovered, last)) in &turns {
        // Once per answer: not while a summary of the turn is being written.
        if recovered.is_some_and(|recovered| recovered.0 == last.map(|it| it.0))
            || running.iter().any(|&CallOf(of)| of == turn)
        {
            continue;
        }
        // A failed turn gives its prompt back, so its context is the one checked before its
        // request, and pi's check after an error has nothing new to count.
        let TurnOutcome::Answered(_) = outcome else {
            continue;
        };
        let Ok((conversation, Some(connection), last, stale, reasoning)) = agents.get(agent) else {
            continue;
        };
        let route = ran_on.and_then(|ran| ran.route.clone());
        let spec = ran_on.map_or(&*connection.spec, |ran| &*ran.spec);
        let end = conversation.messages().len();
        let Some(summarizing) =
            over_threshold(conversation, (last, stale), (&default.0, spec), end)
        else {
            continue;
        };
        let request = match summarizing.request(spec, reasoning.and_then(|it| it.0)) {
            Ok(request) => request,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
                continue;
            }
        };
        let model = ModelRequest {
            connection: route,
            ..ModelRequest::new(request)
        };
        let summary = commands
            .spawn((
                Name::new(SUMMARIZING),
                summarizing,
                model,
                Effect::default().with_purpose(PURPOSE),
                ForAgent(agent),
                CallOf(turn),
                AfterRun,
            ))
            .id();
        commands.spawn((Name::new(AFTER_RUN), WaitsFor(summary), HoldOf(turn)));
    }
}

/// On an agent whose turn ended in a final error: where its last request went, which
/// [`compact_after_error`] checks against once the turn is gone, unless a new turn's request is
/// checked first.
#[derive(Component)]
struct CheckAfterError(Option<RanOn>);

/// Marks the agent of each turn that ended in a final error for [`compact_after_error`]. Not a
/// turn that compaction itself ran, nor one whose overflow recovery was tried for its newest
/// prompt, as pi's check returns after that recovery.
fn check_after_error(
    settled: On<Add<Settled>>,
    turns: Query<
        (
            (&ForAgent, &TurnOutcome, Option<&RanOn>),
            (Option<&Recovered>, Option<&LastPrompt>),
        ),
        Without<OwnTurn>,
    >,
    mut commands: Commands,
) {
    let Ok(((&ForAgent(agent), TurnOutcome::Failed(_), ran_on), (recovered, last))) =
        turns.get(settled.entity)
    else {
        return;
    };
    if recovered.is_some_and(|recovered| recovered.0 == last.map(|it| it.0)) {
        return;
    }
    commands
        .entity(agent)
        .try_insert(CheckAfterError(ran_on.cloned()));
}

/// Once the turn that ended in a final error is gone, compacts in a turn of its own when the
/// context it left (the last reported usage plus what follows, such as a message another author
/// sent that the turn left) leaves less than the reserve free, as pi checks after a run that ended
/// in an error.
fn compact_after_error(
    agents: Query<(Entity, Compacting, &CheckAfterError), Without<ActiveTurn>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, (conversation, connection, last, stale, reasoning), CheckAfterError(ran)) in &agents
    {
        commands.entity(agent).try_remove::<CheckAfterError>();
        let Some(spec) = ran
            .as_ref()
            .map(|ran| ran.spec.clone())
            .or(connection.map(|it| it.spec.clone()))
        else {
            continue;
        };
        let end = conversation.messages().len();
        if !default.0.enabled {
            continue;
        }
        let measure = (last, stale);
        let Some(summarizing) = over_threshold(conversation, measure, (&default.0, &spec), end)
        else {
            continue;
        };
        let route = ran.as_ref().and_then(|ran| ran.route.clone());
        let route = (route, &*spec, reasoning.and_then(|it| it.0));
        compact_in_own_turn(&mut commands, &mut notices, agent, route, summarizing);
    }
}

/// On a turn compaction started for itself, which ends with its summary.
#[derive(Component)]
struct OwnTurn;

/// Starts `summarizing` for `agent` on `route` (the agent's own connection when `None`), sized for
/// `spec`, in a turn of its own that ends with the summary.
fn compact_in_own_turn(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    agent: Entity,
    (route, spec, reasoning): (Option<Connection>, &ModelSpec, Option<Reasoning>),
    summarizing: Summarizing,
) {
    let request = if summarizing.messages.is_empty() {
        Ok(CompletionRequest::new(completion::Message::user(NOTHING)))
    } else {
        summarizing.request(spec, reasoning)
    };
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent), OwnTurn))
        .id();
    match request {
        Ok(request) => {
            let model = ModelRequest {
                connection: route,
                ..ModelRequest::new(request)
            };
            commands.spawn((
                Name::new(SUMMARIZING),
                summarizing,
                model,
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

/// Refuses each `/compact` summary call that nothing was planned for, nor a plugin planned anew,
/// with pi's reason: "Already compacted" when nothing was added since the last summary, else
/// "Nothing to compact (session too small)".
fn nothing_to_compact(
    calls: Query<(Entity, &Summarizing, &ForAgent), (Added<Admitted>, ModelCalls)>,
    conversations: Query<&Conversation>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, summarizing, &ForAgent(agent)) in &calls {
        if summarizing.messages.is_empty() {
            let already = conversations.get(agent).is_ok_and(already_compacted);
            let why = match already {
                true => "Already compacted",
                false => "Nothing to compact (session too small)",
            };
            notices.write(Notice::error(agent, format!("Compaction failed: {why}")));
            let why = ErrorReport::new(ErrorKind::Cancelled, why);
            commands.entity(call).try_insert_if_new(Outcome(Err(why)));
        }
    }
}

/// Whether `conversation` ends in its last compaction, as pi's session does when nothing was
/// added after it: a summary is in, and every other message is older than it.
fn already_compacted(conversation: &Conversation) -> bool {
    let summary = conversation
        .entries()
        .find(|entry| entry.author == Some(&summary_author()))
        .map(|entry| entry.id);
    summary.is_some_and(|summary| conversation.entries().all(|entry| entry.id <= summary))
}

/// What a `/compact` with nothing to summarize sends, refused before it goes.
const NOTHING: &str = "Nothing to compact.";

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
    let route = (None, &*connection.spec, reasoning.and_then(|it| it.0));
    compact_in_own_turn(&mut commands, &mut notices, agent, route, summarizing);
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

/// After the model refused a turn's request as too long, or the window cut a reply with no tool
/// calls short (it stopped at its output limit below the model's own), compacts and sends the
/// request again, once, as pi does; the reply cut short is dropped. A second one after that is
/// left as it is, with a notice that recovery failed. Not when compaction is disabled.
fn recover_from_overflow(
    failed: Query<
        (
            Entity,
            &SentTo,
            &Outcome,
            Option<&RetryVerdict>,
            &CallOf,
            &ForAgent,
            Has<OverflowTried>,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
        ),
    >,
    started: Query<&HoldOf, With<Summarizing>>,
    ended: Query<&TurnOutcome>,
    ran: Query<&RanOn>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !default.0.enabled {
        return;
    }
    for (call, SentTo(connection), outcome, verdict, &CallOf(turn), &ForAgent(agent), tried) in
        &failed
    {
        let overflow = verdict.is_some_and(|verdict| verdict.0 == Verdict::Overflow);
        let cut_short = outcome.completion().is_some_and(|reply| {
            let most = connection.spec.output_limit().map(u64::from).unwrap_or(0);
            let written = reply.usage.output_tokens.unwrap_or(0);
            reply.stop() == StopReason::Length
                && reply.tool_calls().next().is_none()
                && most > 0
                && written < most
        });
        if !(overflow || cut_short)
            || ended.get(turn).is_ok_and(stops)
            || started.iter().any(|&HoldOf(held)| held == call)
        {
            continue;
        }
        if tried {
            mark_recovered(&mut commands, turn);
            if cut_short {
                let why = "Truncated response recovery failed after one compact-and-retry attempt.";
                notices.write(Notice::error(agent, why));
            }
            continue;
        }
        let Ok((conversation, ..)) = agents.get(agent) else {
            continue;
        };
        let reasoning = agents
            .get(agent)
            .ok()
            .and_then(|it| it.4)
            .and_then(|it| it.0);
        commands.entity(call).try_insert(OverflowTried);
        let keep = default.0.keep_recent_tokens;
        let Some(summarizing) = Summarizing::plan(conversation, keep, CompactReason::Overflow)
        else {
            // pi omits the attempt before it compacts, so with nothing to summarize it is gone
            // all the same, and the run ends there.
            mark_recovered(&mut commands, turn);
            commands.queue(move |world: &mut World| fail_recovered(world, (call, agent), None));
            continue;
        };
        // As pi omits the attempt before it compacts: what streamed is not saved by a stop now.
        commands.entity(call).try_remove::<Partial>();
        // The turn's own route, before adapters wrapped it, sized for the model it went to.
        let route = ran.get(turn).ok().and_then(|ran| ran.route.clone());
        let route = (route, &*connection.spec, reasoning);
        if start(
            &mut commands,
            &mut notices,
            (agent, turn, Some(call)),
            route,
            summarizing,
        )
        .is_some()
        {
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

/// Whether `outcome` ends its turn for good: failed or stopped, not an answer proposed, which a
/// summary after the run holds.
fn stops(outcome: &TurnOutcome) -> bool {
    matches!(outcome, TurnOutcome::Failed(_) | TurnOutcome::Stopped(_))
}

/// How summary calls are retried: pi's coding-agent defaults (`settings.retry`: 3 retries, waits
/// from 2 s, at most 60 s).
pub const SUMMARY_RETRY: RetryPolicy = RetryPolicy {
    max_retries: 3,
    first_backoff: std::time::Duration::from_secs(2),
    max_backoff: std::time::Duration::from_secs(60),
    ..RetryPolicy::DEFAULT
};

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

/// pi's patterns of a spent quota or budget, which is never retried.
static SPENT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
            r#"(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing|subscription_sharing_usage_limit_exceeded|subscription_sharing_workplace_individual_usage_limit_exceeded|"code"\s*:\s*"?1308(?:[^0-9]|$)"#,
        )
        .ok()
});

/// pi's patterns of a passing provider or connection failure.
static PASSING: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
            r"(?i)overloaded|server_busy|servers are currently busy|currently experiencing high demand|model is at capacity|rate.?limit|too many requests|(?:^|[^a-z0-9])(?:429|500|502|503|504|520|524)(?:[^a-z0-9]|$)|service.?(?:is.?)?(?:temporarily.?)?unavailable|server.?error|internal.?error|provider.?returned.?(?:an.?)?error|exceeded request buffer limit while retrying upstream|unable to forward upstream|upstream.?request.?failed|upstream_stream_read_error|stream was interrupted|http/?2 stream failed|network.?error|connection.?error|connection.?refused|connection.?lost|connection.?closed|other side closed|fetch failed|request to \S+ failed, reason:|getaddrinfo|(?:^|[^a-z0-9_])(?:ECONNRESET|ECONNABORTED|ECONNREFUSED|ETIMEDOUT|EPIPE)(?:[^a-z0-9_]|$)|connection reset by peer|ENOTFOUND|EAI_AGAIN|upstream.?connect|reset before headers|socket hang up|socket connection was closed|timed? out|timeout|terminated|websocket.?closed|websocket.?error|websocket.?connection.?limit.?reached|ended without|stream ended before message_stop|stream ended before a terminal response event|http2 request did not get a response|stream disconnected before completion|cut short after the reply had begun|pending stream has been canceled|retry delay|you can retry your request|try your request again|please retry your request|ResourceExhausted|subscription_sharing_usage_unavailable|subscription_sharing_user_unavailable",
        )
        .ok()
});

/// Whether a summary call that failed with `report` is sent again, as pi's
/// `isRetryableAssistantError` decides by its message: a passing failure by pi's patterns, never
/// a spent quota or budget; never an overflow or a cancel.
fn retryable(report: &ErrorReport) -> bool {
    let terminal = report.is_context_overflow() || report.kind == ErrorKind::Cancelled;
    !terminal && passing(&report.to_string())
}

/// Whether an error message reads as a passing failure by pi's patterns, never a spent quota or
/// budget.
fn passing(text: &str) -> bool {
    let matches =
        |pattern: &LazyLock<Option<Regex>>| pattern.as_ref().is_some_and(|it| it.is_match(text));
    !matches(&SPENT) && matches(&PASSING)
}

/// The passing failure a summary call settled with, as pi reads its assistant's error: a failed
/// call's report, or a completed reply whose stop is an error; `None` for anything else.
fn passing_failure(outcome: &Outcome) -> Option<String> {
    match (&outcome.0, outcome.completion()) {
        (Err(report), _) => retryable(report).then(|| report.to_string()),
        (Ok(_), Some(reply)) => match reply.stop() {
            StopReason::Error(why) if passing(&why) => Some(why),
            _ => None,
        },
        (Ok(_), None) => None,
    }
}

/// pi's `retryDelayMs`: the first wait doubled for each retry already made, capped, with no jitter.
fn retry_delay(policy: &RetryPolicy, retries: u32) -> std::time::Duration {
    let doubled = policy.first_backoff.saturating_mul(1u32 << retries.min(16));
    doubled.min(policy.max_backoff)
}

/// Sends a failed summary call again once the wait passed, when its failure passes and its turn
/// goes on, up to [`SUMMARY_RETRY`]'s three times for each summary call, as pi retries each: a hold
/// with the [`Backoff`] keeps it from settling until then, and the call sent again ([`resend`])
/// takes its marker and its hold on the call it is for.
fn retry_compaction_calls(
    failed: Query<
        (
            Entity,
            &Outcome,
            &CallOf,
            &ForAgent,
            Option<&SummaryRetries>,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            With<Summarizing>,
            Without<TurnCall>,
            Without<SupersededBy>,
        ),
    >,
    ended: Query<&TurnOutcome>,
    mut commands: Commands,
) {
    let policy = SUMMARY_RETRY;
    for (call, outcome, &CallOf(turn), &ForAgent(agent), retries) in &failed {
        let Some(report) = passing_failure(outcome) else {
            continue;
        };
        let retries = retries.map_or(0, |it| it.0);
        if ended.get(turn).is_ok_and(stops) || retries >= policy.max_retries {
            continue;
        }
        let delay = retry_delay(&policy, retries);
        let retries = retries.saturating_add(1);
        commands.entity(call).try_insert(SummaryRetries(retries));
        let backoff = Backoff::after(retries, delay);
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
/// is none; a passing failure that used up its retries ([`SUMMARY_RETRY`]) says so.
fn taken(
    (outcome, retries): (&Outcome, u32),
    summarizing: &Summarizing,
) -> Result<String, NotTaken> {
    let label = summarizing.label();
    let response = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => {
            let most = SUMMARY_RETRY.max_retries;
            let why = match retryable(report) && retries >= most {
                true => format!(
                    "{report}. Not retrying: it failed {} times in a row",
                    retries.saturating_add(1)
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
        (
            &Summarizing,
            &ModelRequest,
            &Outcome,
            Option<&SummaryRetries>,
        ),
        (Option<&HoldOf>, Has<AfterRun>),
    )>,
    held: Query<&Effect>,
    waiting: Query<(Entity, &WaitsFor, &HoldOf)>,
    turn_calls: Query<(), With<TurnCall>>,
    ended: Query<&TurnOutcome>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Connection>,
        Option<&ReasoningSetting>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((
        (&CallOf(turn), &ForAgent(agent)),
        (summarizing, model, outcome, retries),
        (hold, after_run),
    )) = calls.get(settled.entity)
    else {
        return;
    };
    let Ok((mut conversation, own, reasoning)) = agents.get_mut(agent) else {
        return;
    };
    let ended = ended.get(turn).is_ok_and(stops);
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
        false => taken((outcome, retries.map_or(0, |it| it.0)), summarizing),
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
            let started = spec.and_then(|spec| {
                start(
                    &mut commands,
                    &mut notices,
                    (agent, turn, hold),
                    (route, &spec, reasoning),
                    next,
                )
            });
            if let Some(next) = started {
                if after_run {
                    // What waited for this call waits for the next.
                    commands.entity(next).insert(AfterRun);
                    for (waiter, &WaitsFor(summary), _) in &waiting {
                        if summary == settled.entity {
                            commands.entity(waiter).insert(WaitsFor(next));
                        }
                    }
                }
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
    let mut failure = None;
    let step = match summary {
        Ok((older, kept)) => {
            notices.write(Notice::info(agent, compacted_text(older, kept)));
            Recovering::Compacted
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
            Recovering::NotSummarized
        }
        Err(NotTaken::Failed(why)) => {
            // pi's words for an automatic compaction that failed.
            let text = match summarizing.reason {
                CompactReason::Threshold => format!("Auto-compaction failed: {why}"),
                CompactReason::Overflow => format!("Context overflow recovery failed: {why}"),
                CompactReason::Asked { .. } => {
                    format!("{why}; the conversation was not compacted.")
                }
            };
            notices.write(Notice::error(agent, text));
            failure = Some(why);
            Recovering::NotSummarized
        }
    };
    // As in pi, the context is unknown until a reply reports it.
    if conversation.revision() != revision {
        commands.entity(agent).try_insert(ContextStale::default());
    }
    let overflowed = summarizing.reason == CompactReason::Overflow;
    if after_run {
        // The turn's answer goes on to end, and each request made meanwhile is built again.
        for (waiter, &WaitsFor(summary), &HoldOf(held_call)) in &waiting {
            if summary != settled.entity {
                continue;
            }
            commands.entity(waiter).try_despawn();
            if !ended && turn_calls.contains(held_call) {
                let parent = held.get(held_call).ok().and_then(|effect| effect.parent);
                supersede(
                    &mut commands,
                    (turn, agent, held_call),
                    parent,
                    (step, false),
                );
            }
        }
        return;
    }
    match hold {
        // Built again unchanged, the refused call would be refused again: it settles as it is,
        // its failure naming why the summary that was to make room failed; a reply cut short is
        // dropped, as pi omits it before it compacts, and the turn stops.
        Some(call) if step == Recovering::NotSummarized && overflowed => {
            mark_recovered(&mut commands, turn);
            let why = failure.or_else(|| Some("the compaction was cancelled".to_owned()));
            commands.queue(move |world: &mut World| {
                fail_recovered(world, (call, agent), why.as_deref());
            });
        }
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

/// Ends the recovery of `call`, `agent`'s turn's call refused as too long or whose reply the window
/// cut short, whose compaction did not go in (`why`, when it failed or was cancelled, a notice
/// already said). As pi omits the attempt and ends the run, the reply is dropped and the turn
/// stops there, the user's prompt and its attachments kept; a refusal nothing could be compacted
/// for is shown.
fn fail_recovered(world: &mut World, (call, agent): (Entity, Entity), why: Option<&str>) {
    let Some(mut outcome) = world.get_mut::<Outcome>(call) else {
        return;
    };
    let (refused, cut) = match &outcome.0 {
        Err(report) => (
            Some(report.to_string()),
            "the model refused the request as too long",
        ),
        Ok(_) => (None, "the context window cut the reply short"),
    };
    outcome.0 = Err(ErrorReport::new(ErrorKind::Cancelled, cut));
    world.entity_mut(call).remove::<Partial>();
    if why.is_none() {
        let text = match refused {
            Some(refused) => {
                format!("The model call failed: {refused}. There is nothing to compact.")
            }
            None => {
                "The context window cut the reply short. There is nothing to compact.".to_owned()
            }
        };
        world.write_message(Notice::error(agent, text));
    }
    world.trigger(StopTurn::plugin_alone(agent, "compaction", cut));
}

/// Says that `older` messages of so many tokens were summarized and `kept` were kept.
fn compacted_text(
    (older, older_tokens): (usize, u64),
    (kept, kept_tokens): (usize, u64),
) -> String {
    let plural = |count: usize| if count == 1 { "message" } else { "messages" };
    format!(
        "Compacted {older} {} (about {} tokens) into a summary and kept the newest {kept} {} \
         (about {} tokens).",
        plural(older),
        tokens_label(older_tokens),
        plural(kept),
        tokens_label(kept_tokens),
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
