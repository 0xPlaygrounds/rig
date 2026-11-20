//! Compaction: keeps each agent's conversation inside its model's input limit by summarizing its
//! older messages, as the pi coding agent does. When enabled, it checks the context against
//! `[compaction] reserve_tokens` of the model's input limit before each request of a turn, the
//! request as it goes: before a prompt is sent (the prompt counted, never summarized) and after
//! tool results. After a request refused as too long, or a reply the window cut short, it
//! compacts and sends the request again, once.
//! `/compact [instructions]` compacts at once.
//!
//! A compaction keeps about `keep_recent_tokens` of the newest messages, never cutting between a
//! tool call and its result, and summarizes the older ones with a separate call to the model the
//! request goes to ([`Summarizing`]): the previous summary is updated with the new messages, and
//! a turn the cut splits gets a second summary of its earlier part. rig-memory's
//! [`prepare_compaction`] holds the cut, the prompts and the requests. A summary that failed, was
//! cut short or is blank, or whose messages changed while it was written, replaces nothing and
//! says why. A summary call's text is in [`Summarizing::summary`] in [`ReviewSystems::Act`],
//! where a reviewer may edit it or set it to `None` to cancel.

use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, Reasoning, UnsupportedOption, tokens_label};
use rig_core::conversation::Change;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::StopReason;
use rig_ecs::prelude::*;
use rig_ecs::retry::Verdict;
use rig_ecs::session_log::ConversationChanged;
use rig_harness::AppKeepsExt;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::settings::AppSettingsExt;
use rig_harness::{PluginEntity, ProvidedBy};
use rig_memory::{
    CompactionPreparation, CompactionSettings, SummaryFailure, SummaryState, already_compacted,
    apply_summary, clamp_max_tokens, estimate_tokens, find_cut_point, merge_split_turn, no_history,
    prepare_compaction, prepare_compaction_at, request_tokens, summary_request, summary_text,
    turn_prefix_request,
};
pub use rig_memory::{summarized_prefix, summary_author, summary_in, summary_state};
use serde::Deserialize;
#[cfg(feature = "tui")]
pub use tui::SummaryLooks;

/// The purpose of a summary's model call.
pub const PURPOSE: &str = "compaction";

/// The kind of the saved aside that keeps a reply the window cut short and recovery left out of
/// the conversation, written when the reply is dropped: its value is the reply's message.
pub const OMITTED: &str = "compaction.omitted";

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
            .add_systems(
                AgentPass,
                (
                    (compact_near_the_window, nothing_to_compact).in_set(PrepareSystems::Fit),
                    read_replies.in_set(ReviewSystems::Edit),
                    recover_from_overflow.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(Update, stale_after_edits)
            .add_observer(on_delivered)
            .add_observer(take_summary)
            .add_observer(follow_successor::<Summarizing>)
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
    /// The context leaves less than the reserve of the model's input limit free.
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

/// On a turn: what compaction did for the request it is about to send, as pi's one recovery flag.
/// It goes, so both may happen again, once a message of the user's is delivered into the turn or
/// a reply of the turn comes that neither failed nor stopped at a limit. A summary going in
/// changes neither.
#[derive(Component, Clone, Copy, Default)]
struct Tried {
    /// It was compacted for near the window, or that was tried: it goes as it is.
    threshold: bool,
    /// Its overflow or cut-short reply was recovered from, or that was tried: a second one stands.
    overflow: bool,
}

/// On a turn's call: the output tokens its request asked for before they were fitted to the
/// window, which says whether the window cut its reply short.
#[derive(Component, Clone, Copy)]
struct Asked(u64);

/// A summary call of a compaction: a [`ModelRequest`] of the turn, holding ([`HoldOf`]) the turn's
/// call it compacts for, if any, and [`ProvidedBy`] this plugin when loaded as one. A turn the cut
/// splits takes two calls, one after the other: the history, then the turn's earlier part. The
/// kernel retries a failed one by `[retry]`, each call on a count of its own.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The messages the summary replaces: the conversation before the first one kept, earlier
    /// summaries included, as it was when planned. A summary that arrives after one of them
    /// changed replaces nothing.
    pub replaces: Arc<[Arc<completion::Message>]>,
    /// The state the summary goes into: its files; the summary is written in once it is taken.
    pub state: SummaryState,
    /// The newest compaction's state, which the update prompt sends with its file lists; `None`
    /// before the first compaction.
    pub previous: Option<SummaryState>,
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
        let overflowed = reason == CompactReason::Overflow;
        let prepared = prepare_compaction(conversation, keep_recent_tokens, overflowed)?;
        Some(Self::of(prepared, reason))
    }

    /// The first summary call of a compaction of `conversation` for `reason` that keeps from
    /// message `first_kept`, as a plugin that cuts its own way plans it; `None` when nothing
    /// would be summarized or a tool result would be kept without its call
    /// ([`prepare_compaction_at`]).
    pub fn plan_at(
        conversation: &rig_core::conversation::Conversation,
        first_kept: usize,
        reason: CompactReason,
    ) -> Option<Self> {
        let prepared = prepare_compaction_at(conversation, first_kept)?;
        Some(Self::of(prepared, reason))
    }

    /// The first summary call of the compaction `prepared` for `reason`.
    fn of(prepared: CompactionPreparation, reason: CompactReason) -> Self {
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
                Some(no_history(previous.as_ref())),
            ),
            false => {
                let prefix =
                    (!prepared.turn_prefix.is_empty()).then(|| prepared.turn_prefix.into());
                (prepared.to_summarize, prefix, None)
            }
        };
        Self {
            reason,
            replaces: prepared.replaces.into(),
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
            replaces: Arc::new([]),
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
                let previous = (self.previous.as_ref(), focus);
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
/// that reply's index, while it still describes it: no compaction or edit since
/// ([`ContextStale`]), and no message before the reply newer than it, such as one a plugin put in
/// front of a request, which the figure never counted.
fn anchor(
    history: &rig_core::conversation::Conversation,
    (last, stale): (&LastUsage, Option<&ContextStale>),
) -> Option<(u64, usize)> {
    let anchor = last.0.as_ref().filter(|_| stale.is_none())?;
    let tokens = anchor.usage.context_tokens().filter(|&tokens| tokens > 0)?;
    let at = history.index_of(anchor.through)?;
    let through = history.id(at);
    (0..at)
        .all(|before| history.id(before) <= through)
        .then_some((tokens, at))
}

/// The threshold compaction of `conversation` for a request of `tokens` input tokens, when that
/// leaves less than the reserve of `spec`'s input limit free; `None` otherwise or with nothing to
/// summarize. Its messages from `end` on are a pending prompt, which is never cut.
fn over_threshold(
    conversation: &Conversation,
    tokens: u64,
    (settings, spec): (&CompactionSettings, &ModelSpec),
    end: usize,
) -> Option<Summarizing> {
    let window = u64::from(spec.input_limit()?);
    if !settings.should_compact(tokens, window) {
        return None;
    }
    let range = (summarized_prefix(conversation), end);
    let keep = settings.keep_recent_tokens;
    let first_kept = find_cut_point(conversation.messages(), range, keep, false);
    Summarizing::plan_at(conversation, first_kept, CompactReason::Threshold)
}

/// Before each turn's request, compacts when the request leaves less than the reserve of the
/// input limit of the model it goes to free, before a prompt (which is counted and never cut) and
/// before each request after tool results: the request is held, and built again once the summary
/// is in. A request compacted for already, or whose compaction failed, goes as it is ([`Tried`]).
/// A request that goes asks for at most the room its input leaves in the window, by pi-ai's
/// clamp. Both go by one count of the request as it is sent, by pi-ai's estimate: its system
/// prompt, tools and the history it sends, or the tokens last reported for that history plus an
/// estimate of the messages since ([`anchor`]). pi's own threshold leaves a pending prompt out. A
/// request a plugin brought whole, which sends no history of the agent's, is only clamped.
fn compact_near_the_window(
    mut calls: Query<
        (Entity, &mut ModelRequest, &Adapted, &CallOf, &ForAgent),
        (
            Added<Admitted>,
            With<TurnCall>,
            Without<Started>,
            Without<Outcome>,
            Without<SupersededBy>,
        ),
    >,
    turns: Query<(Option<&Prompt>, Option<&Tried>)>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let settings = default.0;
    for (call, mut model, Adapted(connection), &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok((conversation, _, last, stale, reasoning)) = agents.get(agent) else {
            continue;
        };
        let (prompt, tried) = turns.get(turn).unwrap_or_default();
        let tried = tried.copied().unwrap_or_default();
        let spec = &*connection.spec;
        let messages = conversation.messages();
        // Before the turn's first reply, the message its prompt delivered is pending; whatever
        // came before it is context.
        let end = prompt
            .and_then(|&Prompt(id)| conversation.index_of(id))
            .filter(|&at| {
                !messages
                    .get(at..)
                    .unwrap_or_default()
                    .iter()
                    .any(|message| matches!(&**message, completion::Message::Assistant(_)))
            })
            .unwrap_or(messages.len());
        let prompt = model.prompt.as_ref().map(|prompt| prompt.text());
        let tools = spec
            .tools
            .then(|| serde_json::to_string(&model.request.tools).ok())
            .flatten()
            .filter(|_| !model.request.tools.is_empty());
        let parts = (prompt.as_deref(), tools.as_deref());
        // What Start sends: the call's own history, else the request's, which has no ids to
        // anchor a reported count to.
        let input = match &model.history {
            Some(history) => {
                request_tokens(parts, history.messages(), anchor(history, (last, stale)))
            }
            None => request_tokens(parts, &model.request.chat_history, None),
        };
        // A summary of the conversation shortens only a request that sends it.
        if !tried.threshold
            && model.history.is_some()
            && let Some(summarizing) = over_threshold(conversation, input, (&settings, spec), end)
        {
            let route = (
                model.connection.clone(),
                spec,
                reasoning.and_then(|it| it.0),
            );
            let of = (agent, turn, Some(call));
            if start(&mut commands, &mut notices, of, route, summarizing).is_some() {
                let threshold = true;
                commands
                    .entity(turn)
                    .try_insert(Tried { threshold, ..tried });
                continue;
            }
        }
        let asked = model
            .request
            .max_tokens
            .or(spec.output_limit().map(u64::from));
        if let Some(asked) = asked {
            let fitted = clamp_max_tokens(spec, input, asked);
            commands.entity(call).try_insert(Asked(asked));
            if model.request.max_tokens != Some(fitted) {
                model.request_mut().max_tokens = Some(fitted);
            }
        }
    }
}

/// On a turn: the id of the message its prompt delivered, the first message delivered into it.
#[derive(Component)]
struct Prompt(rig_core::conversation::MessageId);

/// Marks each turn with its [`Prompt`], and, as pi resets its recovery on each user message, lets
/// a turn a message of the user's is delivered into be compacted for and recovered again
/// ([`Tried`]).
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
        turn.try_remove::<Tried>();
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
            let already = conversations
                .get(agent)
                .is_ok_and(|it| already_compacted(it));
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

/// What a `/compact` with nothing to summarize sends, refused before it goes.
const NOTHING: &str = "Nothing to compact.";

/// Compacts an idle agent's conversation on the user's request, in a turn of its own that ends
/// with the summary; what the agent accepts meanwhile carries that turn on
/// ([`StopTurn::plugin_done`]). With nothing to summarize, the call still goes to Prepare, where
/// a plugin that cuts its own way may plan it; otherwise it is refused there.
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
    let request = match summarizing.messages.is_empty() {
        true => Ok(CompletionRequest::new(completion::Message::user(NOTHING))),
        false => summarizing.request(&connection.spec, reasoning.and_then(|it| it.0)),
    };
    match request {
        Ok(request) => {
            let turn = commands.spawn(TurnOf(agent)).id();
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

/// After the model refused a turn's request as too long, or the window cut a reply short (it
/// stopped at a limit having written less than its request asked for, [`Asked`]), compacts and
/// sends the request again, once ([`Tried`]), as pi does; the reply cut short is dropped, its
/// tool calls never run, and it is kept as an [`OMITTED`] aside. A second one after that is left
/// as it is, with a notice that recovery failed. Not when compaction is disabled.
fn recover_from_overflow(
    replied: Query<
        (
            (Entity, &SentTo, &ModelRequest, &Outcome),
            (Option<&RetryVerdict>, Option<&Asked>),
            (&CallOf, &ForAgent),
        ),
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
        ),
    >,
    started: Query<&HoldOf, With<Summarizing>>,
    turns: Query<(Option<&TurnOutcome>, Option<&Tried>)>,
    agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !default.0.enabled {
        return;
    }
    for ((call, SentTo(connection), model, outcome), (verdict, asked), of) in &replied {
        let (&CallOf(turn), &ForAgent(agent)) = of;
        let Ok((ended, tried)) = turns.get(turn) else {
            continue;
        };
        let reply = outcome.completion();
        let overflow = verdict.is_some_and(|verdict| verdict.0 == Verdict::Overflow);
        let cut_short = reply.is_some_and(|reply| {
            let written = reply.usage.output_tokens;
            reply.stop() == StopReason::Length
                && asked.is_some_and(|&Asked(asked)| written.is_some_and(|it| it < asked))
        });
        // As pi's flag resets on a reply that is neither an error nor cut at a limit.
        let whole = |reply: &completion::CompletionResponse| {
            let stop = reply.stop();
            !stop.is_failure() && stop != StopReason::Length
        };
        if tried.is_some() && reply.is_some_and(whole) {
            commands.entity(turn).try_remove::<Tried>();
        }
        if !(overflow || cut_short)
            || ended.is_some_and(stops)
            || started.iter().any(|&HoldOf(held)| held == call)
        {
            continue;
        }
        if tried.is_some_and(|tried| tried.overflow) {
            if cut_short {
                let why = "Truncated response recovery failed after one compact-and-retry attempt.";
                notices.write(Notice::error(agent, why));
            }
            continue;
        }
        let Ok((conversation, .., reasoning)) = agents.get(agent) else {
            continue;
        };
        commands.entity(turn).try_insert(Tried {
            threshold: true,
            overflow: true,
        });
        let omitted = reply.filter(|_| cut_short).and_then(|it| it.message());
        let keep = default.0.keep_recent_tokens;
        let Some(summarizing) = Summarizing::plan(conversation, keep, CompactReason::Overflow)
        else {
            // pi omits the attempt before it compacts, so with nothing to summarize it is gone
            // all the same, and the run ends there.
            commands.queue(move |world: &mut World| {
                omit(world, agent, omitted);
                fail_recovered(world, (call, agent), None);
            });
            continue;
        };
        // The route the request named, before adapters wrapped it, sized for where it went.
        let route = (
            model.connection.clone(),
            &*connection.spec,
            reasoning.and_then(|it| it.0),
        );
        let of = (agent, turn, Some(call));
        if start(&mut commands, &mut notices, of, route, summarizing).is_none() {
            // A reply cut short is dropped all the same, so its tool calls never run.
            if cut_short {
                commands.queue(move |world: &mut World| {
                    omit(world, agent, omitted);
                    fail_recovered(world, (call, agent), Some("no summary request"));
                });
            }
        } else {
            // As pi omits the attempt before it compacts: what streamed is not saved by a stop.
            commands.entity(call).try_remove::<Partial>();
            commands.queue(move |world: &mut World| omit(world, agent, omitted));
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

/// Keeps `omitted`, a reply of `agent`'s recovery drops, in its saved conversation as an
/// [`OMITTED`] aside, which no request sends, as pi keeps the omitted attempt in its session file.
fn omit(world: &mut World, agent: Entity, omitted: Option<completion::Message>) {
    if let Some(omitted) = omitted
        && let Ok(value) = serde_json::to_value(&omitted)
        && let Some(mut conversation) = world.get_mut::<Conversation>(agent)
    {
        let after = conversation.entries().next_back().map(|entry| entry.id);
        conversation.add_aside(Aside::new(OMITTED, summary_author(), value).after(after));
    }
}

/// Whether `outcome` ends its turn for good: failed or stopped, not an answer proposed.
fn stops(outcome: &TurnOutcome) -> bool {
    matches!(outcome, TurnOutcome::Failed(_) | TurnOutcome::Stopped(_))
}

/// What the agent is told a compaction after an overflow is for.
const OUTGREW: &str = "The conversation outgrew the model's context window";

/// Spawns the call that supersedes `call`, a turn's model call held while the conversation was
/// compacted for it, built anew from the conversation on the same route ([`Rebuilt`]), under the
/// effect `parent`.
fn supersede(
    commands: &mut Commands,
    (turn, agent, call): (Entity, Entity, Entity),
    parent: Option<rig_core::effect::EffectId>,
) {
    let successor = (
        TurnCall,
        Rebuilt,
        CallOf(turn),
        ForAgent(agent),
        Effect::with_parent_id(parent),
    );
    commands
        .spawn(successor)
        .add_related::<SupersededBy>(&[call]);
}

/// Marks each summary call as this plugin's, and as one the kernel retries on a count of its own
/// ([`Recovery`]): a retry carries its count, and a call sent again any other way starts afresh.
fn provided_by_compaction(
    added: On<Add<Summarizing>>,
    me: Option<Res<PluginEntity<CompactionPlugin>>>,
    mut commands: Commands,
) {
    commands
        .entity(added.entity)
        .try_insert_if_new(Recovery::default());
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
/// is none; a failure the kernel gave up retrying (`verdict`) says so.
fn taken(
    (outcome, verdict): (&Outcome, Option<&RetryVerdict>),
    summarizing: &Summarizing,
) -> Result<String, NotTaken> {
    let label = summarizing.label();
    let response = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => {
            let why = match verdict {
                Some(RetryVerdict(Verdict::GaveUp(why))) => {
                    format!("{report}. Not retrying: {why}")
                }
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
        Some(text) if text.trim().is_empty() => {
            Err(NotTaken::Failed(SummaryFailure::Empty(label).to_string()))
        }
        Some(text) => Ok(text.clone()),
        // The reply had one, so a reviewer took it out.
        None => match summary_text(response, label) {
            Ok(_) => Err(NotTaken::Cancelled),
            Err(why) => Err(NotTaken::Failed(why.to_string())),
        },
    }
}

/// Takes a settled summary call. The history of a split turn starts the call for its earlier
/// part. A whole summary replaces the messages it read in the conversation as an entry of its
/// own, a splice its log records, and the call it held is sent again, built anew. A failed
/// summary, one a reviewer cancelled, one whose turn ended first, or one whose messages changed
/// meanwhile replaces nothing and says why; the call it held is sent as it is, or, after an
/// overflow, fails. A turn started for the summary is done then ([`StopTurn::plugin_done`]).
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(
        (&CallOf, &ForAgent),
        (&Summarizing, &ModelRequest, &Outcome, Option<&RetryVerdict>),
        Option<&HoldOf>,
    )>,
    held: Query<&Effect>,
    ended: Query<&TurnOutcome>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Connection>,
        Option<&ReasoningSetting>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(((&CallOf(turn), &ForAgent(agent)), (summarizing, model, outcome, verdict), hold)) =
        calls.get(settled.entity)
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
            let why = "nothing to compact";
            commands.trigger(StopTurn::plugin_done(agent, "compaction", why));
        }
        return;
    }
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
            let reasoning = reasoning.and_then(|it| it.0);
            let of = (agent, turn, hold);
            let started = spec.and_then(|spec| {
                let route = (route, &*spec, reasoning);
                start(&mut commands, &mut notices, of, route, next)
            });
            if started.is_some() {
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
    let summary = text.and_then(|text| {
        let end = summarizing.replaces.len();
        let older = (end, estimate_tokens(&summarizing.replaces));
        let kept = conversation.messages().get(end..).unwrap_or_default();
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
        let taken = (0..end).filter_map(|at| conversation.entry(at)).collect();
        let spliced = apply_summary(&mut conversation, &summarizing.replaces, &state);
        let spliced = spliced.ok_or_else(|| {
            NotTaken::Failed("the messages it summarized changed meanwhile".to_owned())
        })?;
        #[cfg(feature = "tui")]
        tui::keep(&mut commands, agent, &spliced, taken);
        let _ = spliced;
        Ok((older, kept))
    });
    let mut failure = None;
    let summarized = match summary {
        Ok((older, kept)) => {
            notices.write(Notice::info(agent, compacted_text(older, kept)));
            true
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
            false
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
            false
        }
    };
    // As in pi, the context is unknown until a reply reports it.
    if conversation.revision() != revision {
        commands.entity(agent).try_insert(ContextStale::default());
    }
    match hold {
        // Built again unchanged, the refused call would be refused again: it settles as it is,
        // its failure naming why the summary that was to make room failed; a reply cut short is
        // dropped, as pi omits it before it compacts, and the turn stops.
        Some(call) if !summarized && summarizing.reason == CompactReason::Overflow => {
            let why = failure.or_else(|| Some("the compaction was cancelled".to_owned()));
            commands.queue(move |world: &mut World| {
                fail_recovered(world, (call, agent), why.as_deref());
            });
        }
        Some(call) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            supersede(&mut commands, (turn, agent, call), parent);
        }
        // The turn started for it is done, unless it ended.
        None if !ended => commands.trigger(StopTurn::plugin_done(agent, "compaction", COMPACTED)),
        None => {}
    }
}

/// Why the turn started for a `/compact` is done once its summary settled.
const COMPACTED: &str = "compacted";

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

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;
