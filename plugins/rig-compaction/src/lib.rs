//! Compaction: keeps each agent's requests inside its model's context
//! window.
//!
//! - **When:**
//!   - before a turn's request that nears the window, in
//!     [`PrepareSystems::Fit`], after every plugin's edits, by an estimate
//!     anchored to the usage the provider last reported (rig-memory's
//!     [`next_request_tokens`]); there too the request's output limit is
//!     clamped to the room the window leaves;
//!   - after a request refused as too long, in [`ReviewSystems::Act`], read
//!     from its [`RetryVerdict`];
//!   - on `/compact <focus>`.
//! - **Bounded by progress:** a request that does not fit goes through one
//!   recovery path ([`Recovering`]): as it is, then with old tool outputs
//!   cleared, then compacted, then the turn fails. Each step happens once,
//!   carried through retries, until a model reply moves the conversation
//!   on, so a turn compacts as often as it needs but never twice without a
//!   reply in between.
//! - **How:** older tool outputs are cleared first, near the window down to
//!   the newest within a tenth of the room below the reserve, so a
//!   clearing, which rewrites what the prompt cache holds, comes seldom
//!   and leaves little to read again. If that is not enough,
//!   the model summarizes the older messages into a summary, built on the
//!   [`summary_state`].
//!   Both are splices by [`summary_author`], so they are logged, restored
//!   and announced as [`ConversationChanged`]. The summary call is a
//!   [`ModelRequest`] of the turn ([`TurnCall`]) holding the call it
//!   compacts for ([`HoldOf`]), so stopping the turn cancels it; a call
//!   built from the compacted conversation then supersedes the held one.
//!   Near the window it is the held call's own request with one
//!   instruction appended, so the provider reads the conversation from its
//!   prompt cache, and it is asked only when it would summarize at least a
//!   quarter of the conversation: every summary rewrites what the cache
//!   holds after the system prompt and tools.
//! - **Review:** once answered, [`Summarizing::summary`] holds the text in
//!   [`ReviewSystems::Act`]. A reviewer may hold the call, edit the text,
//!   or set it to `None` to cancel.
//! - **Undo:** every compaction keeps what it replaced
//!   ([`Compacted`](rig_memory::Compacted)); [`Uncompact`] restores the
//!   newest one with its message ids. The next compaction builds on
//!   [`summary_state`].
//! - **Policy:** [`CompactionDefault`], `[compaction]` in settings. Other
//!   compaction, per agent or by other rules, is a plugin that replaces this
//!   one.

use bevy_ecs::query::QueryData;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, tokens_label};
use rig_core::conversation::Entry;
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::settings::AppSettingsExt;
use rig_harness::{PluginEntity, ProvidedBy};
use rig_memory::{
    ClearToolOutputs, Cleared, CompactReason, CompactionPolicy, Prepared, RequestStart, Summarizer,
    SummaryCut, SummaryPlan, SummaryState, TrackArgument, apply_summary, estimate_tokens,
    next_request_tokens, output_room, undo_summary,
};
pub use rig_memory::{summarized_prefix, summary_author, summary_in, summary_state};
use serde::{Deserialize, Deserializer};

/// Compacts agents' conversations near their model's window, after an
/// overflow, and on `/compact`, by [`CompactionDefault`].
#[derive(Default)]
pub struct CompactionPlugin;

impl Plugin for CompactionPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<CompactionDefault>("compaction")
            .add_command(
                "compact",
                "Summarize all but the newest reply to free context; /compact <focus> says what to keep",
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
                read_summary.in_set(ReviewSystems::Edit),
                recover_from_overflow.in_set(ReviewSystems::Act),
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

/// The compaction policy of every agent: rig-memory's default, tracking
/// the files the built-in file tools read and changed, with `[compaction]`
/// from `settings.toml` over it (each key a field of the
/// [`CompactionPolicy`] table, so `keep_recent = 1000` keeps the rest). A
/// plugin whose tool reads or changes files tracks its argument by adding
/// a [`TrackArgument`] to its `tracked` in `build` or `finish`, after this
/// plugin's.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct CompactionDefault(pub CompactionPolicy);

impl Default for CompactionDefault {
    /// The built-in default.
    fn default() -> Self {
        Self(CompactionPolicy {
            tracked: TRACKED.to_vec(),
            ..CompactionPolicy::default()
        })
    }
}

impl<'de> Deserialize<'de> for CompactionDefault {
    /// `[compaction]` over the built-in default: a table without `tracked`
    /// tracks the built-in file tools.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let table = serde_json::Map::deserialize(deserializer)?;
        let tracks = table.contains_key("tracked");
        let policy = CompactionPolicy::deserialize(serde_json::Value::Object(table));
        let mut policy = policy.map_err(serde::de::Error::custom)?;
        if !tracks {
            policy.tracked = TRACKED.to_vec();
        }
        Ok(Self(policy))
    }
}

/// The files the built-in file tools (rig-coding-tools') were called with:
/// `read`'s are read, `edit`'s and `write`'s changed (a later set wins in
/// the summary).
const TRACKED: &[TrackArgument] = &[
    TrackArgument::new("read", "path", "read-files"),
    TrackArgument::new("edit", "path", "modified-files"),
    TrackArgument::new("write", "path", "modified-files"),
];

/// Compact the agent's conversation now: its messages up to the small tail
/// its policy keeps (by default the newest reply alone) are
/// replaced in requests by a summary the model writes, focused on `focus`
/// when it is not empty. Refused while a turn runs.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Compact {
    /// The agent.
    pub entity: Entity,
    /// What the summary should keep above all; may be empty.
    pub focus: String,
}

/// Put back the messages the agent's newest compaction replaced, with
/// their ids, in place of its summary, as the session's log holds them
/// ([`undo_summary`]); the summary state ([`summary_state`]) goes back to
/// the compaction's before it. Refused while a turn runs or in a session that is not
/// saved.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Uncompact {
    /// The agent.
    pub entity: Entity,
}

/// What the agent's last turn request started with, as sent: the usage
/// the provider reports for its reply anchors the next estimate while the
/// next request starts the same ([`next_request_tokens`]).
#[derive(Component, Clone, Copy, Debug)]
struct Sent(RequestStart);

/// The step a turn's model call is on along its recovery path, when it is
/// past the first: the request as it was, then with old tool outputs
/// cleared, then compacted, then the turn fails. A call that replaces one
/// (a retry, or the call rebuilt after a summary) carries its step, so it
/// is never cleared or compacted again; a call after a model reply starts
/// from the first step.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component, Clone, Debug)]
pub enum Recovering {
    /// Old tool outputs were cleared for it.
    Cleared,
    /// The conversation was compacted for it: if it does not fit either,
    /// the turn fails.
    Compacted,
}

/// The summary call of a compaction, a [`ModelRequest`] of the turn, and
/// a hold ([`HoldOf`]) on the model call it compacts for, if any; it is
/// [`ProvidedBy`] this plugin, when loaded as one. Sent again
/// ([`resend`](rig_harness::successor::resend)), it moves to the
/// successor with its hold, whose summary is the one taken.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The first message the compaction keeps, by id, as messages may move
    /// while the summary is written; `None` summarizes every message. A
    /// summary that arrives after it was taken out replaces nothing.
    pub keep_from: Option<MessageId>,
    /// The state the summary goes into.
    pub state: SummaryState,
    /// The summary, read from the model's reply in
    /// [`ReviewSystems::Edit`] once the call is answered: `None` before,
    /// and when the reply has none. A reviewer in [`ReviewSystems::Act`]
    /// reads it, replaces it, or sets it to `None` to cancel the
    /// compaction: nothing is replaced and the call it held is sent as
    /// it was.
    pub summary: Option<String>,
}

/// On a summary call whose [`Summarizing::summary`] was read from its
/// reply, so a reviewer's edit stays.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct SummaryRead;

/// An agent as compacting it reads it: its messages and its model.
#[derive(QueryData)]
#[query_data(mutable)]
struct Compacting {
    conversation: &'static mut Conversation,
    connection: Option<&'static Connection>,
}

/// Plans one summary call of `conversation` with `policy` for `reason`,
/// written by the model `spec`, building on the conversation's summary
/// state ([`summary_state`]): the [`Summarizing`] to spawn it with and its
/// request, one of its own with the summarizer's system prompt and the
/// older messages as text. `None` when no message would be summarized that
/// is not already; an error says why the policy cannot plan it.
/// rig-compaction plans `/compact` and the summary after an overflow so,
/// and near the window asks at the end of the held turn call's own request
/// instead; a plugin that summarizes with another policy or model, such as
/// a fallback, replaces a summary call's [`Summarizing`] and
/// [`ModelRequest`] with what this returns.
pub fn plan(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    spec: &ModelSpec,
) -> Result<Option<(Summarizing, CompletionRequest)>, String> {
    let state = summary_state(conversation);
    let planned = policy.plan_summary(conversation, &state, &reason, spec);
    let Some(SummaryPlan {
        keep_from,
        state,
        request,
        ..
    }) = planned.map_err(|why| why.to_string())?
    else {
        return Ok(None);
    };
    let summarizing = Summarizing {
        reason,
        keep_from,
        state,
        summary: None,
    };
    Ok(Some((summarizing, request)))
}

/// Plans one summary call of `conversation` for `reason` as `held`, the
/// turn's own call as prepared for `spec`, with the instruction of
/// `policy`'s summarizer appended ([`Summarizer::instruction`]): the same
/// system prompt, tools, reasoning setting and cache key, so the provider
/// reads the conversation from its prompt cache and the model writes from
/// the real tool outputs. `None` when nothing new would be summarized.
fn plan_in_place(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    (held, spec): (&ModelRequest, &ModelSpec),
) -> Option<(Summarizing, ModelRequest)> {
    let state = summary_state(conversation);
    let SummaryCut {
        upto,
        keep_from,
        state,
    } = policy.cut_summary(conversation, &state, &reason, spec)?;
    let kept = conversation.messages().get(upto).map(|kept| &**kept);
    let focus = match &reason {
        CompactReason::Asked { focus } => focus.trim(),
        _ => "",
    };
    let ask = completion::Message::user(policy.summarizer.instruction(kept, focus));
    let mut request = held.clone();
    match &mut request.history {
        Some(history) => {
            history.push(Entry::new(ask).by(summary_author()));
        }
        None => request.request_mut().chat_history.push(ask.into()),
    }
    request.request_mut().max_tokens = Some(policy.summarizer.max_tokens(spec));
    let summarizing = Summarizing {
        reason,
        keep_from,
        state,
        summary: None,
    };
    Some((summarizing, request))
}

/// How much of the conversation a summary near the window must take out,
/// as a divisor: at least a quarter. A summary rewrites every message the
/// prompt cache holds after the system prompt and tools, which the
/// provider then reads at full price, so one that frees little costs more
/// than it saves.
const WORTH: u64 = 4;

/// Whether a summary of `conversation` near the window, by `policy` for
/// `spec`, would take out at least a [`WORTH`]th of its `history` tokens,
/// as the request sends them. When nothing would be summarized it is, so
/// the start says so.
fn worth_summarizing(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    spec: &ModelSpec,
    history: u64,
) -> bool {
    let messages = conversation.messages();
    let from = summarized_prefix(conversation);
    let reason = CompactReason::Threshold;
    let Some(upto) = policy.cut(messages, from, &reason, Some(spec)) else {
        return true;
    };
    let older = estimate_tokens(messages.get(from.min(upto)..upto).unwrap_or_default());
    older.saturating_mul(WORTH) >= history
}

/// How much of the room below the reserve the tool outputs kept by a
/// clearing near the window may fill, as a divisor: a tenth. Clearing
/// rewrites what the prompt cache holds from the first output it clears,
/// and the provider reads what it keeps again at full price, so it keeps
/// little and comes seldom.
const CLEAR_TO: u64 = 10;

/// The clearing near the window of `spec` by `policy`: the newest tool
/// outputs within the policy's `keep_tokens` or a [`CLEAR_TO`]th of the
/// room below the reserve, whichever is less, counted as the provider
/// counts them, `scale` provider tokens per estimated one.
fn clearing_near_the_window(
    policy: &CompactionPolicy,
    spec: &ModelSpec,
    scale: f64,
) -> ClearToolOutputs {
    let reserve = policy.summarizer.limits.reserve;
    let room = spec
        .context_window
        .map_or(u64::MAX, |window| u64::from(window).saturating_sub(reserve));
    let keep = (policy.clearing.keep_tokens as u64).min(room / CLEAR_TO);
    ClearToolOutputs::new((keep as f64 / scale) as usize)
}

/// How many tokens the provider counted for each one estimated: its
/// `reported` count of the last request over the `estimated` tokens of
/// this one, between 1 and 4. The estimate takes four bytes a token, and
/// dense text such as numbers or minified code takes two or less.
fn provider_scale(reported: Option<u64>, estimated: u64) -> f64 {
    match reported {
        Some(reported) if estimated > 0 => (reported as f64 / estimated as f64).clamp(1.0, 4.0),
        _ => 1.0,
    }
}

/// `tokens` estimated, as the provider counts them at `scale`.
fn scaled(tokens: u64, scale: f64) -> u64 {
    (tokens as f64 * scale) as u64
}

/// Tells `agent` that, in `situation`, `cleared` was cleared, then `then`.
fn cleared_notice(agent: Entity, situation: &str, cleared: &Cleared, then: &str) -> Notice {
    let tokens = tokens_label(cleared.tokens as u64);
    let results = cleared.results;
    Notice::info(
        agent,
        format!("{situation}: cleared {results} older tool outputs (about {tokens} tokens){then}."),
    )
}

/// Says that `older` messages of so many tokens were summarized, `kept`
/// were kept unchanged, and the model now gets `left` tokens.
fn compacted_text(
    (older, older_tokens): (usize, u64),
    (kept, kept_tokens): (usize, u64),
    left: u64,
) -> String {
    let (kept, as_it_was) = match kept {
        1 => ("the newest message".to_owned(), "as it was"),
        count => (format!("the newest {count} messages"), "as they were"),
    };
    format!(
        "Compacted {older} {} (about {} tokens) into a summary and kept {kept} (about {} tokens) \
         {as_it_was}; the model now gets about {} tokens of conversation.",
        if older == 1 { "message" } else { "messages" },
        tokens_label(older_tokens),
        tokens_label(kept_tokens),
        tokens_label(left)
    )
}

/// What [`CompactingItem::start`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SummaryStart {
    /// Nothing: the agent has no model to write the summary.
    NoModel,
    /// Nothing: no message would be summarized that is not already.
    NothingNew,
    /// Nothing: the summary could not be planned, and the agent was told
    /// why.
    Refused,
    /// The summary call started.
    Started,
}

impl CompactingItem<'_, '_> {
    /// Clears the older tool outputs of the agent's conversation as
    /// `clearing` says, by a splice of the messages that changed, which
    /// keeps their ids.
    fn clear_old_outputs(&mut self, clearing: &ClearToolOutputs) -> Cleared {
        clearing.clear_conversation(&mut self.conversation, summary_author())
    }

    /// Starts the summary call of a compaction of `agent` for `reason` on
    /// `turn`, holding the model call `call`, or on a turn of its own when
    /// `None`, when it has a model and anything new would be summarized.
    /// With `held`, the held call's prepared request and the model it goes
    /// to, the summary call is that request with the instruction appended
    /// ([`plan_in_place`]); without, a request of its own ([`plan`]).
    fn start(
        &self,
        (agent, turn): (Entity, Option<(Entity, Entity)>),
        (policy, reason): (&CompactionPolicy, CompactReason),
        held: Option<(&ModelRequest, &ModelSpec)>,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) -> SummaryStart {
        let Some(connection) = self.connection else {
            return SummaryStart::NoModel;
        };
        let planned = match held {
            Some(held) => Ok(plan_in_place(&self.conversation, policy, reason, held)),
            None => plan(&self.conversation, policy, reason, &connection.spec)
                .map(|planned| planned.map(|(it, request)| (it, ModelRequest::new(request)))),
        };
        let (summarizing, request) = match planned {
            Ok(Some(planned)) => planned,
            Ok(None) => return SummaryStart::NothingNew,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
                return SummaryStart::Refused;
            }
        };
        let summary = (
            Name::new("compacting"),
            summarizing,
            request,
            Effect::default().with_purpose("compaction"),
            ForAgent(agent),
        );
        match turn {
            Some((turn, call)) => commands.spawn((summary, CallOf(turn), HoldOf(call))),
            None => {
                let turn = commands
                    .spawn((Name::new("compaction"), TurnOf(agent)))
                    .id();
                commands.spawn((summary, CallOf(turn)))
            }
        };
        SummaryStart::Started
    }
}

/// Compacts an idle agent's conversation on the user's request, in a turn
/// of its own that ends with the summary.
fn on_compact(
    compact: On<Compact>,
    mut agents: Query<(Compacting, Has<ActiveTurn>), With<Agent>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok((compacting, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let reason = CompactReason::Asked {
        focus: compact.focus.clone(),
    };
    let refusal = match compacting.start(
        (agent, None),
        (&default.0, reason),
        None,
        &mut commands,
        &mut notices,
    ) {
        SummaryStart::NoModel => Notice::no_model(agent),
        SummaryStart::NothingNew => Notice::info(agent, "Nothing to compact yet."),
        SummaryStart::Refused | SummaryStart::Started => return,
    };
    notices.write(refusal);
}

/// About how many tokens `model`, a turn's call for `spec` with the system
/// prompt `prompt`, sends, anchored to `last` while the call starts as the
/// request `sent` did ([`next_request_tokens`]); and what it starts with.
fn measure(
    model: &ModelRequest,
    spec: &ModelSpec,
    prompt: Option<&str>,
    (last, sent): (&LastUsage, Option<RequestStart>),
) -> (u64, RequestStart) {
    let bare;
    let history = match &model.history {
        Some(history) => history,
        None => {
            let messages = model.request.chat_history.iter().cloned();
            bare = rig_core::conversation::Conversation::from_messages(messages);
            &bare
        }
    };
    // Start takes the tools off a model that calls none.
    let tools = if spec.tools {
        model.request.tools.as_slice()
    } else {
        &[]
    };
    let request = Prepared {
        model: &spec.id,
        prompt,
        tools,
        history,
    };
    let tokens = next_request_tokens(last.0.as_ref(), sent, &request);
    (tokens, request.start())
}

/// The room `spec`'s window leaves for the output of a request of
/// `tokens`, at most its own limit or the model's ([`output_room`]).
fn room(spec: &ModelSpec, asked: Option<u64>, tokens: u64) -> Option<u64> {
    let window = spec.context_window.map(u64::from);
    let most = asked.or(spec.max_output_tokens.map(u64::from));
    output_room(window, most, tokens)
}

/// Before a turn's request that leaves less than the reserve of the
/// window of the model it goes to free, or no room for an answer, takes
/// the next step of the call's recovery path ([`Recovering`]): clears old
/// tool outputs, which costs no model call, and asks for a summary only
/// when clearing was not enough. A request that goes clamps its output
/// limit to the room the window leaves.
fn compact_near_the_window(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &Adapters,
            &CallOf,
            &ForAgent,
            Option<&Recovering>,
        ),
        (Added<Admitted>, With<TurnCall>),
    >,
    mut agents: Query<(Compacting, &LastUsage, Option<&Sent>)>,
    starts: ModelStarts,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, adapters, &CallOf(turn), &ForAgent(agent), step) in &mut calls {
        let Ok((mut compacting, last, sent)) = agents.get_mut(agent) else {
            continue;
        };
        let Ok((connection, prompt)) = starts.prompt(&model, Some(adapters), Some(agent)) else {
            continue;
        };
        let (policy, spec) = (&default.0, &*connection.spec);
        let prompt = prompt.map(|prompt| prompt.text());
        let prompt = prompt.as_deref();
        let anchor = (last, sent.map(|sent| sent.0));
        let (mut used, mut start) = measure(&model, spec, prompt, anchor);
        let asked = model.request.max_tokens;
        let over = |used| policy.over_threshold(used, spec) || room(spec, asked, used) == Some(0);
        if !over(used) {
            fit_output(&mut model, spec, used);
            commands.entity(agent).try_insert(Sent(start));
            continue;
        }
        let mut step = step.copied();
        if step.is_none() {
            let whole = measure(&model, spec, prompt, (last, None)).0;
            let scale = provider_scale(Some(used), whole);
            let clearing = clearing_near_the_window(policy, spec, scale);
            let cleared = compacting.clear_old_outputs(&clearing);
            if cleared.results > 0 {
                // The request's copy too, as cleared by compaction.
                match &mut model.history {
                    Some(history) => {
                        clearing.clear_conversation(history, summary_author());
                    }
                    None => {
                        clearing.clear(&mut model.request_mut().chat_history);
                    }
                }
                step = Some(Recovering::Cleared);
                commands.entity(call).try_insert(Recovering::Cleared);
                used = used.saturating_sub(scaled(cleared.tokens as u64, scale));
                start = measure(&model, spec, prompt, anchor).1;
                let nears = "The conversation nears the model's context window";
                notices.write(cleared_notice(agent, nears, &cleared, ""));
            }
        }
        if over(used) && step < Some(Recovering::Compacted) && {
            let history = estimate_tokens(compacting.conversation.messages());
            worth_summarizing(&compacting.conversation, policy, spec, history)
        } {
            let reason = (policy, CompactReason::Threshold);
            let at = (agent, Some((turn, call)));
            let held = Some((&*model, spec));
            let started = compacting.start(at, reason, held, &mut commands, &mut notices);
            if started == SummaryStart::Started {
                // The call is held, and built again once the summary is in.
                continue;
            }
        }
        fit_output(&mut model, spec, used);
        commands.entity(agent).try_insert(Sent(start));
    }
}

/// Clamps the output limit of `model`, a call to `spec` of `used` tokens,
/// to the room the window leaves, when that is less than the limit and
/// more than nothing: no room is a call that does not fit, sent as it is
/// for the provider to refuse.
fn fit_output(model: &mut ModelRequest, spec: &ModelSpec, used: u64) {
    let asked = model.request.max_tokens;
    let most = asked.or(spec.max_output_tokens.map(u64::from));
    let Some(room) = room(spec, asked, used).filter(|room| *room > 0) else {
        return;
    };
    if most.is_none_or(|most| room < most) {
        model.request_mut().max_tokens = Some(room);
    }
}

/// When a turn's call that carries a step of its recovery path is
/// replaced, such as by a retry, gives the step to its successor, unless
/// that has one of its own.
fn carry_recovery(
    superseded: On<Add<SupersededBy>>,
    calls: Query<(&SupersededBy, &Recovering)>,
    mut commands: Commands,
) {
    if let Ok((by, &step)) = calls.get(superseded.entity) {
        commands.entity(by.successor()).try_insert_if_new(step);
    }
}

/// After the model refused a turn's request as too long, takes the next
/// step of the call's recovery path ([`Recovering`]): clears old tool
/// outputs and sends it again; when they were cleared for it already or
/// none were, summarizes the older messages first. On the last step the
/// turn fails.
fn recover_from_overflow(
    failed: Query<
        (
            Entity,
            &RetryVerdict,
            &Effect,
            &CallOf,
            &ForAgent,
            Option<&Recovering>,
        ),
        (ReadyToSettle, With<TurnCall>, Without<SupersededBy>),
    >,
    mut agents: Query<Compacting>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, verdict, effect, &CallOf(turn), &ForAgent(agent), step) in &failed {
        if verdict.0 != Verdict::Overflow || step == Some(&Recovering::Compacted) {
            continue;
        }
        let Ok(mut compacting) = agents.get_mut(agent) else {
            continue;
        };
        let policy = &default.0;
        let outgrew = "The conversation outgrew the model's context window";
        if step.is_none() {
            let cleared = compacting.clear_old_outputs(&policy.clearing);
            if cleared.results > 0 {
                let then = " and sending it again";
                notices.write(cleared_notice(agent, outgrew, &cleared, then));
                commands
                    .spawn((
                        TurnCall,
                        CallOf(turn),
                        ForAgent(agent),
                        Effect::with_parent_id(effect.parent),
                        Recovering::Cleared,
                    ))
                    .add_related::<SupersededBy>(&[call]);
                continue;
            }
        }
        let reason = (policy, CompactReason::Overflow);
        let at = (agent, Some((turn, call)));
        // The refused request does not fit, so neither would it with an
        // instruction: the summary gets a request of its own.
        if compacting.start(at, reason, None, &mut commands, &mut notices) == SummaryStart::Started
        {
            notices.write(Notice::info(
                agent,
                format!("{outgrew}: summarizing its older messages and sending it again."),
            ));
        }
    }
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

/// Reads each answered summary call's summary from its reply, once. A
/// summary call is never a turn's own call, which other reviewers edit.
fn read_summary(
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
        let response = outcome.completion();
        summarizing.summary = response.and_then(|reply| Summarizer::summary_text(reply).ok());
        commands.entity(call).try_insert(SummaryRead);
    }
}

/// Takes a settled summary: it replaces the summarized messages in the
/// conversation as an entry of its own, a splice its log records. A failed
/// summary, one a reviewer cancelled, or one whose kept message is gone,
/// replaces nothing. Either way the call it held is sent again, built
/// anew, or the turn ends when the user asked for the compaction.
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ForAgent, &Summarizing, &Outcome, Option<&HoldOf>)>,
    held: Query<&Effect>,
    mut agents: Query<&mut Conversation>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CallOf(turn), &ForAgent(agent), summarizing, outcome, hold)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let end = match summarizing.keep_from {
        None => Some(conversation.messages().len()),
        Some(kept) => conversation.index_of(kept),
    };
    let failed = |why: &str| NotTaken::Failed(why.to_owned());
    let summary = match (&outcome.0, outcome.completion(), end) {
        (Err(report), ..) => Err(NotTaken::Failed(report.to_string())),
        (Ok(_), None, _) => Err(failed("the model did not answer with a completion")),
        (Ok(_), Some(_), None) => Err(failed("the messages it kept were taken out")),
        (Ok(_), Some(response), Some(end)) => match &summarizing.summary {
            Some(summary) => Ok((summary.clone(), end)),
            // The reply had one, so a reviewer took it out.
            None => match Summarizer::summary_text(response) {
                Ok(_) => Err(NotTaken::Cancelled),
                Err(why) => Err(NotTaken::Failed(why.to_string())),
            },
        },
    };
    let summary = summary.and_then(|(summary, end)| {
        let messages = conversation.messages();
        let older = messages.get(..end).unwrap_or_default();
        let older = (older.len(), estimate_tokens(older));
        let kept = messages.get(end..).unwrap_or_default();
        let kept = (kept.len(), estimate_tokens(kept));
        let state = SummaryState {
            summary,
            ..summarizing.state.clone()
        };
        match apply_summary(&mut conversation, summarizing.keep_from, &state) {
            Some(_) => Ok((older, kept)),
            None => Err(failed("the summary was empty")),
        }
    });
    match summary {
        Ok(((older, older_tokens), (kept, kept_tokens))) => {
            let left = estimate_tokens(conversation.messages());
            let text = compacted_text((older, older_tokens), (kept, kept_tokens), left);
            notices.write(Notice::info(agent, text));
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
        }
        Err(NotTaken::Failed(why)) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
        }
    }
    match hold {
        // The call it held is built again, from the compacted conversation,
        // on the last step of its recovery path, whether or not the summary
        // went in.
        Some(&HoldOf(call)) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            commands
                .spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::with_parent_id(parent),
                    Recovering::Compacted,
                ))
                .add_related::<SupersededBy>(&[call]);
        }
        // The user's own compaction turn ends with it.
        None => {
            let compacted = TurnOutcome::Stopped(Some("compacted".to_owned()));
            commands.entity(turn).try_insert(compacted);
        }
    }
}

/// Why a summary call's summary replaced nothing.
enum NotTaken {
    /// A reviewer cancelled it.
    Cancelled,
    /// It failed, for this reason.
    Failed(String),
}

/// Undoes the agent's newest compaction on [`Uncompact`].
fn on_uncompact(
    uncompact: On<Uncompact>,
    mut agents: Query<(&AgentId, &mut Conversation, Has<ActiveTurn>)>,
    store: Option<Res<SessionStore>>,
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
