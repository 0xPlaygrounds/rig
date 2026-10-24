//! Compaction: keeps each agent's requests inside its model's input
//! limit.
//!
//! - **When:**
//!   - before a turn's request that nears the input limit, in
//!     [`PrepareSystems::Fit`], after every plugin's edits, by an estimate
//!     anchored to the usage the provider last reported (rig-memory's
//!     [`next_request_tokens`]); there too the request's output limit is
//!     clamped to the room the window leaves;
//!   - after a request refused as too long, in [`ReviewSystems::Act`], read
//!     from its [`RetryVerdict`];
//!   - on `/compact <focus>`.
//! - **Bounded by progress:** a request that does not fit goes through one
//!   recovery path ([`Recovering`]): as it is, then with old tool outputs
//!   cleared, then summarized, then the turn fails. A summary that fails,
//!   or is cut off at its output limit, is never asked again: old tool
//!   outputs are cleared instead, and the call goes only if it then fits.
//!   Each step happens once, carried through retries, until a model reply
//!   moves the conversation on, so a turn compacts as often as it needs but
//!   never twice without a reply in between.
//! - **How:** older tool outputs are cleared first, all but the newest
//!   20k tokens of them as the provider counts them, so a clearing, which
//!   rewrites what the prompt cache holds, comes seldom. Before that the
//!   model writes notes on the ones it could not get back ([`Noting`]),
//!   which stay after them; a file still on disk can be read again. After
//!   an overflow no notes are asked: outputs are cleared at once only when
//!   each can be read again, and the older messages summarized otherwise.
//!   If clearing would still leave more than half the room used, the model
//!   summarizes the older messages into a checkpoint instead, built on the
//!   [`summary_state`], when that takes out at least a quarter of the
//!   conversation. Each is a splice by [`summary_author`], so it is logged,
//!   restored and announced as [`ConversationChanged`]. A notes or summary
//!   call is a model call of the turn holding the call it is for
//!   ([`HoldOf`]), so stopping the turn cancels it; a call built from the
//!   changed conversation then supersedes the held one. It is the agent's
//!   own request cut where its reading ends, plus one instruction, so the
//!   provider reads it from its prompt cache and the model reads every
//!   output whole, when that fits the model with room for the answer.
//!   Otherwise, and always after an overflow, it is a summary request of
//!   its own ([`plan`]) to the same model, sized to fit. A summary asked
//!   in place gets the room its request leaves, up to the summarizer's
//!   limit, and is told how much. A notes or summary call that fails for a
//!   passing reason is sent again, as the same request, after the turn's
//!   retry wait, counted with its retries; one that timed out, once.
//! - **Review:** once answered, [`Summarizing::summary`] and
//!   [`Noting::notes`] hold the text in [`ReviewSystems::Act`]. A reviewer
//!   may hold the call, edit the text, or set it to `None` to cancel.
//! - **Undo:** every compaction keeps what it replaced
//!   ([`Compacted`](rig_memory::Compacted)); [`Uncompact`] restores the
//!   newest one with its message ids. The next compaction builds on
//!   [`summary_state`].
//! - **Policy:** [`CompactionDefault`], `[compaction]` in settings. Other
//!   compaction, per agent or by other rules, is a plugin that replaces this
//!   one.

use bevy_ecs::query::QueryData;
use bevy_ecs::system::{Command, SystemParam};
use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, tokens_label};
use rig_core::conversation::Entry;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_ecs::turn::RETRY_POLICY;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::rig_tools::workspace::Stat;
use rig_harness::services::{ProjectRoot, WorkDir};
use rig_harness::settings::AppSettingsExt;
use rig_harness::successor::resend;
use rig_harness::workspace::ProjectWorkspace;
use rig_harness::{PluginEntity, ProvidedBy};
use rig_memory::{
    ClearPlan, ClearToolOutputs, Cleared, CompactReason, CompactionPolicy, NearWindow,
    OUTPUT_MARGIN, Prepared, RequestStart, Summarizer, SummaryCut, SummaryPlan, SummaryState,
    TURN_OUTPUT, TrackArgument, apply_summary, estimate_tokens, next_request_tokens, undo_summary,
};
pub use rig_memory::{summarized_prefix, summary_author, summary_in, summary_state};
use serde::{Deserialize, Deserializer};

/// Compacts agents' conversations near their model's input limit, after
/// an overflow, and on `/compact`, by [`CompactionDefault`].
#[derive(Default)]
pub struct CompactionPlugin;

impl Plugin for CompactionPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<CompactionDefault>("compaction")
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
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
                (compact_near_the_window, ask_in_place)
                    .chain()
                    .in_set(PrepareSystems::Fit),
                read_replies.in_set(ReviewSystems::Edit),
                (retry_compaction_calls, recover_from_overflow).in_set(ReviewSystems::Act),
            ),
        )
        .add_observer(take_summary)
        .add_observer(take_notes)
        .add_observer(follow_successor::<Summarizing>)
        .add_observer(follow_successor::<Noting>)
        .add_observer(carry_recovery)
        .add_observer(provided_by_compaction::<Summarizing>)
        .add_observer(provided_by_compaction::<Noting>);
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

/// The files the built-in file tools (`coding_tools`') were called with:
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
/// cleared, then summarized, then the turn fails. A call that replaces one
/// (a retry, or the call rebuilt after a summary) carries its step, so it
/// is never cleared or summarized again; a call after a model reply starts
/// from the first step.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component, Clone, Debug)]
pub enum Recovering {
    /// Old tool outputs were cleared for it.
    Cleared,
    /// Its summary failed, with old tool outputs cleared instead, or was
    /// cancelled: if it does not fit, it is not sent and the turn fails.
    NotSummarized,
    /// The conversation was summarized for it.
    Compacted,
}

/// The summary call of a compaction, a [`ModelRequest`] of the turn, and
/// a hold ([`HoldOf`]) on the model call it compacts for, if any; it is
/// [`ProvidedBy`] this plugin, when loaded as one. Sent again
/// ([`resend`]), it moves to the successor with its hold, whose summary is
/// the one taken.
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
    /// it was, when it fits.
    pub summary: Option<String>,
}

/// The notes call of a clearing: a model call of the turn, and a hold
/// ([`HoldOf`]) on the model call it clears for, asking for notes on the
/// tool outputs of its plan that cannot be had again; it is [`ProvidedBy`]
/// this plugin, when loaded as one.
#[derive(Component, Clone, Debug)]
pub struct Noting {
    /// Why.
    pub reason: CompactReason,
    /// The outputs it clears.
    pub plan: ClearPlan,
    /// The notes, as [`Summarizing::summary`]: a reply with none clears
    /// the outputs without notes, and `None` from a reviewer clears nothing.
    pub notes: Option<String>,
}

impl Noting {
    /// The notes call for `reason` of `plan`, before any reply: with its
    /// name and purpose, `compaction`.
    fn call(reason: CompactReason, plan: ClearPlan) -> impl Bundle {
        let noting = Self {
            reason,
            plan,
            notes: None,
        };
        let effect = Effect::default().with_purpose("compaction");
        (Name::new(NOTING), noting, effect)
    }
}

/// On a summary or notes call whose text was read from its reply, so a
/// reviewer's edit stays.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct SummaryRead;

/// An agent as compacting it reads it: its messages, its model and the
/// directory it works in.
#[derive(QueryData)]
#[query_data(mutable)]
struct Compacting {
    conversation: &'static mut Conversation,
    connection: Option<&'static Connection>,
    dir: Option<&'static WorkDir>,
}

/// Where the agents' file tools find files.
#[derive(SystemParam)]
struct Disk<'w> {
    workspace: Res<'w, ProjectWorkspace>,
    root: Res<'w, ProjectRoot>,
}

impl Disk<'_> {
    /// Whether every output `plan` clears of `conversation`, an agent's
    /// working in `dir`, can be had again by calling its tool again: the
    /// call is of a tool `tracked` names, with the path of a file still
    /// there on this machine as its argument.
    fn all_again(
        &self,
        plan: &ClearPlan,
        conversation: &rig_core::conversation::Conversation,
        tracked: &[TrackArgument],
        dir: Option<&WorkDir>,
    ) -> bool {
        let workspace = &*self.workspace.0;
        // Asking a remote workspace would block the frame.
        if workspace.host().is_some() {
            return false;
        }
        let base = dir.map_or(&self.root.0, |dir| &dir.0);
        let call_of = |result: &rig_core::message::ToolResult| {
            let mut messages = conversation.messages().iter();
            messages.find_map(|message| match &**message {
                completion::Message::Assistant(reply) => {
                    reply.tool_calls().find(|call| call.id == result.call)
                }
                _ => None,
            })
        };
        plan.results(conversation).all(|result| {
            let Some(call) = call_of(result) else {
                return false;
            };
            let path = tracked.iter().find_map(|rule| rule.value(call));
            let there = |path| workspace.stat(&workspace.resolve(Some(base), path));
            path.is_some_and(|path| there(path).is_ok_and(|stat| !matches!(stat, Stat::Missing)))
        })
    }
}

/// Plans one summary call of `conversation` with `policy` for `reason`,
/// written by the model `spec`, building on the conversation's summary
/// state ([`summary_state`]): the [`Summarizing`] to spawn it with and its
/// request, one of its own with the summarizer's system prompt and the
/// older messages as text. `None` when no message would be summarized that
/// is not already; an error says why the policy cannot plan it.
/// `CompactionPlugin` plans the summary after an overflow so, and otherwise
/// asks at the end of the agent's own request instead; a plugin that
/// summarizes with another policy or model, such as a fallback, replaces a
/// summary call's [`Summarizing`] and [`ModelRequest`] with what this
/// returns.
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

/// What the summary for `reason` should keep above all; may be empty.
fn focus(reason: &CompactReason) -> &str {
    match reason {
        CompactReason::Asked { focus } => focus.trim(),
        _ => "",
    }
}

/// Where a request asked in place ([`ask_at`]) ends.
#[derive(Clone, Copy, Debug)]
enum Upto {
    /// Before this message, by id; `None` keeps every message.
    Before(Option<MessageId>),
    /// After this message, by id.
    Through(MessageId),
}

/// Makes `model`, a request the kernel built, ask `ask` in place: its
/// messages cut at `upto`, then `ask` by [`summary_author`], with the
/// output limit `max_tokens`. `None`, changing nothing, for a request a
/// plugin brought whole or one without the message `upto` names.
fn ask_at(model: &mut ModelRequest, upto: Upto, ask: String, max_tokens: u64) -> Option<()> {
    let history = model.history.as_mut()?;
    let end = match upto {
        Upto::Before(None) => history.messages().len(),
        Upto::Before(Some(id)) => history.index_of(id)?,
        Upto::Through(id) => history.index_of(id)?.saturating_add(1),
    };
    history.splice(end.., [], summary_author());
    history.push(Entry::new(completion::Message::user(ask)).by(summary_author()));
    model.request_mut().max_tokens = Some(max_tokens);
    Some(())
}

/// Makes `model` ask in place ([`ask_at`]) for a summary by `summarizer`
/// with `focus`, its output limit the room left ([`Measure::budget`]) and
/// stated in the instruction. `Some(false)` when the room is too small;
/// `None` as [`ask_at`].
fn ask_summary(
    model: &mut ModelRequest,
    upto: Upto,
    (summarizer, focus): (&Summarizer, &str),
    measure: Measure,
) -> Option<bool> {
    let most = summarizer.max_tokens(measure.spec);
    let mut asked = model.clone();
    ask_at(&mut asked, upto, summarizer.instruction(focus, most), most)?;
    let Some(budget) = measure.budget(&asked, most) else {
        return Some(false);
    };
    ask_at(model, upto, summarizer.instruction(focus, budget), budget)?;
    Some(true)
}

/// The summary call of `conversation` for `reason` by `policy` for
/// `spec`, before any reply, building on its [`summary_state`]; `None`
/// when nothing new would be summarized.
fn cut(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    spec: &ModelSpec,
) -> Option<Summarizing> {
    let state = summary_state(conversation);
    let SummaryCut { keep_from, state } =
        policy.cut_summary(conversation, &state, &reason, spec)?;
    Some(Summarizing {
        reason,
        keep_from,
        state,
        summary: None,
    })
}

/// A turn's model call a compaction call is for: its prepared request and
/// the model it goes to.
type Held<'a> = (&'a ModelRequest, &'a ModelSpec);

/// Plans one summary call of `conversation` for `reason` of its own
/// ([`plan`]), sized to fit `spec`, the model a turn's call goes to: to
/// `route`, that call's own model when not its agent's.
fn plan_alone(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    (route, spec): (Option<&Connection>, &ModelSpec),
) -> Result<Option<(Summarizing, ModelRequest)>, String> {
    let planned = plan(conversation, policy, reason, spec)?;
    Ok(planned.map(|(summarizing, request)| {
        let request = ModelRequest {
            connection: route.cloned(),
            ..ModelRequest::new(request)
        };
        (summarizing, request)
    }))
}

/// Plans one summary call of `conversation` for `reason` as `held`, cut
/// before the first message kept with the instruction of `policy`'s
/// summarizer appended ([`ask_at`]), when that fits as `measure` counts.
/// A request a plugin brought whole, or one that does not fit, gets a
/// request of its own ([`plan_alone`]). `None` when nothing new would be
/// summarized.
fn plan_in_place(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    (held, measure): (&ModelRequest, Measure),
) -> Result<Option<(Summarizing, ModelRequest)>, String> {
    let spec = measure.spec;
    let Some(summarizing) = cut(conversation, policy, reason, spec) else {
        return Ok(None);
    };
    let mut request = held.clone();
    let ask = (&policy.summarizer, focus(&summarizing.reason));
    let upto = Upto::Before(summarizing.keep_from);
    if ask_summary(&mut request, upto, ask, measure) != Some(true) {
        let route = held.connection.as_ref();
        return plan_alone(conversation, policy, summarizing.reason, (route, spec));
    }
    Ok(Some((summarizing, request)))
}

/// How much of the conversation a summary near the window must take out,
/// as a divisor: at least a quarter. A summary rewrites every message the
/// prompt cache holds after the system prompt and tools, which the
/// provider then reads at full price, so one that frees little costs more
/// than it saves.
const WORTH: u64 = 4;

/// Whether a summary of `conversation` near the window, by `policy` for
/// `spec`, would take out at least a [`WORTH`]th of its tokens. When
/// nothing would be summarized it is, so the start says so.
fn worth_summarizing(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    spec: &ModelSpec,
) -> bool {
    let messages = conversation.messages();
    let from = summarized_prefix(conversation);
    let reason = CompactReason::Threshold;
    let Some(upto) = policy.cut(messages, from, &reason, Some(spec)) else {
        return true;
    };
    let older = estimate_tokens(messages.get(from.min(upto)..upto).unwrap_or_default());
    older.saturating_mul(WORTH) >= estimate_tokens(messages)
}

/// How much of the room below the reserve a clearing near the window must
/// leave free, as a divisor: half. One that frees less comes again soon,
/// each time rewriting what the prompt cache holds, so the older messages
/// are summarized into a checkpoint instead.
const ENOUGH: u64 = 2;

/// `policy` as it applies near the window: the newest tool output and
/// messages it keeps counted as the provider counts them, `scale` provider
/// tokens per estimated one.
fn near_the_window(policy: &CompactionPolicy, scale: f64) -> CompactionPolicy {
    let mut policy = policy.clone();
    policy.clearing.keep_tokens = (policy.clearing.keep_tokens as f64 / scale) as usize;
    policy.keep_recent = (policy.keep_recent as f64 / scale) as usize;
    policy
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

/// Tells `agent` that `cleared` was cleared for `reason`, then `then`.
fn cleared_notice(agent: Entity, reason: &CompactReason, cleared: &Cleared, then: &str) -> Notice {
    let situation = match reason {
        CompactReason::Overflow => OUTGREW,
        _ => NEARS,
    };
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

/// What the agent is told a clearing near the window is for.
const NEARS: &str = "The conversation nears the model's context window";

/// What the agent is told a clearing or summary after an overflow is for.
const OUTGREW: &str = "The conversation outgrew the model's context window";

/// How a running notes or summary call is cancelled, as a notice says it.
const CANCEL: &str = "stopping the turn (Esc) cancels it";

/// The name of a summary call, which the agent's status shows.
const SUMMARIZING: &str = "summarizing older messages";

/// The name of a notes call, which the agent's status shows.
const NOTING: &str = "noting older tool outputs";

impl CompactingItem<'_, '_> {
    /// Whether clearing `plan` by `policy` asks the model for notes first:
    /// near the window it summarizes, and not every output it clears can
    /// be had again ([`Disk::all_again`]).
    fn notes_needed(&self, plan: &ClearPlan, policy: &CompactionPolicy, disk: &Disk) -> bool {
        policy.near_window == NearWindow::Summarize
            && !disk.all_again(plan, &self.conversation, &policy.tracked, self.dir)
    }

    /// Starts the summary call of a compaction of `agent` for `reason` on
    /// `turn`, holding its model call `call` (`held`), when anything new
    /// would be summarized; says whether it started. With `measure`, the
    /// summary call is the held request asked in place when that fits
    /// ([`plan_in_place`]); without, a request of its own
    /// ([`plan_alone`]).
    fn start(
        &self,
        (agent, turn, call): (Entity, Entity, Entity),
        (policy, reason): (&CompactionPolicy, CompactReason),
        (held, measure): (Held, Option<Measure>),
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) -> bool {
        let conversation = &self.conversation;
        let planned = match measure {
            Some(measure) => plan_in_place(conversation, policy, reason, (held.0, measure)),
            None => plan_alone(
                conversation,
                policy,
                reason,
                (held.0.connection.as_ref(), held.1),
            ),
        };
        let (summarizing, request) = match planned {
            Ok(Some(planned)) => planned,
            Ok(None) => return false,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
                return false;
            }
        };
        commands.spawn((
            Name::new(SUMMARIZING),
            summarizing,
            request,
            Effect::default().with_purpose("compaction"),
            ForAgent(agent),
            CallOf(turn),
            HoldOf(call),
            Fallback(policy.clearing.clone()),
        ));
        true
    }
}

/// On a summary call that holds a turn's call: the clearing of the policy
/// it was started with, as it applies there, that stands in for a failed
/// summary.
#[derive(Component)]
struct Fallback(ClearToolOutputs);

/// Compacts an idle agent's conversation on the user's request, in a turn
/// of its own that ends with the summary. The summary call is built by the
/// kernel like the agent's next request ([`SideCall`]) and asked in place
/// ([`ask_in_place`]), so the provider reads it from its prompt cache.
fn on_compact(
    compact: On<Compact>,
    agents: Query<(Compacting, Has<ActiveTurn>), With<Agent>>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok((compacting, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(connection) = compacting.connection else {
        notices.write(Notice::no_model(agent));
        return;
    };
    let reason = CompactReason::Asked {
        focus: compact.focus.clone(),
    };
    let spec = &connection.spec;
    let Some(summarizing) = cut(compacting.conversation, &default.0, reason, spec) else {
        notices.write(Notice::info(agent, "Nothing to compact yet."));
        return;
    };
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent)))
        .id();
    commands.spawn((
        Name::new(SUMMARIZING),
        summarizing,
        SideCall(agent),
        ForAgent(agent),
        CallOf(turn),
        Effect::default().with_purpose("compaction"),
    ));
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
    prepared(model, spec, prompt, |request| {
        let tokens = next_request_tokens(last.0.as_ref(), sent, request);
        (tokens, request.start())
    })
}

/// The estimate of `model`, a call for `spec` with the system prompt
/// `prompt`, by the bytes it sends ([`Prepared::whole_tokens`]).
fn estimate(model: &ModelRequest, spec: &ModelSpec, prompt: Option<&str>) -> u64 {
    prepared(model, spec, prompt, |request| request.whole_tokens())
}

/// What `read` makes of `model`, a call for `spec` with the system prompt
/// `prompt`, as it is sent.
fn prepared<R>(
    model: &ModelRequest,
    spec: &ModelSpec,
    prompt: Option<&str>,
    read: impl FnOnce(&Prepared) -> R,
) -> R {
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
    read(&Prepared {
        model: &spec.id,
        prompt,
        tools,
        history,
    })
}

/// How a compaction call cut from a turn's call is measured: by the bytes
/// it sends to `spec` with the system prompt `prompt`, at `scale`, the
/// provider tokens per estimated one of the turn's call.
#[derive(Clone, Copy)]
struct Measure<'a> {
    spec: &'a ModelSpec,
    prompt: Option<&'a str>,
    scale: f64,
}

impl<'a> Measure<'a> {
    /// The measure of calls cut from `model`, a turn's call to `spec` with
    /// the system prompt `prompt`, that the provider counts as `used`.
    fn of(model: &ModelRequest, spec: &'a ModelSpec, prompt: Option<&'a str>, used: u64) -> Self {
        let scale = provider_scale(Some(used), estimate(model, spec, prompt));
        Self {
            spec,
            prompt,
            scale,
        }
    }

    /// The output limit of `request`, a compaction call: the room left, up
    /// to `most`, when it fits ([`fits`]).
    fn budget(&self, request: &ModelRequest, most: u64) -> Option<u64> {
        let tokens = scaled(estimate(request, self.spec, self.prompt), self.scale);
        fits(self.spec, tokens, most).then(|| room(self.spec, most, tokens))
    }
}

/// The least room for an answer a request needs, unless it asks for less.
/// A compaction call cut from a turn's call with less room is a summary
/// request of its own.
const LEAST_OUTPUT: u64 = 12_000;

/// The explicit output limit, or the default answer budget plus thinking
/// within the model's output limit; `None`, for the server's default, when
/// the model's window and output limit are both unknown.
fn output_budget(request: &CompletionRequest, spec: &ModelSpec) -> Option<u64> {
    let thinking = match request.options.reasoning {
        Some(completion::Reasoning::Budget { tokens }) => u64::from(tokens) + 1024,
        _ => 0,
    };
    let wanted = TURN_OUTPUT + thinking;
    let default = match spec.max_output_tokens {
        Some(most) => Some(wanted.min(most.into())),
        None => spec.context_window.map(|_| wanted),
    };
    request.max_tokens.or(default)
}

/// The room `spec`'s window leaves for the output of a request of
/// `tokens`, at most its output budget.
fn room(spec: &ModelSpec, asked: u64, tokens: u64) -> u64 {
    spec.context_window.map_or(asked, |window| {
        asked.min(u64::from(window).saturating_sub(tokens.saturating_add(OUTPUT_MARGIN)))
    })
}

/// Whether a request of `tokens` to `spec` that asks for `asked` tokens of
/// output fits: within the model's input limit, with room for at least
/// [`LEAST_OUTPUT`] of them. A limit not known is taken to fit.
fn fits(spec: &ModelSpec, tokens: u64, asked: u64) -> bool {
    spec.input_limit()
        .is_none_or(|limit| tokens <= u64::from(limit))
        && room(spec, asked, tokens) >= asked.clamp(1, LEAST_OUTPUT)
}

/// Before a turn's request that leaves less than the reserve of the input
/// limit of the model it goes to free, or no room for an answer, takes
/// the next step of the call's recovery path ([`Recovering`]): clears old
/// tool outputs, after the model's notes on those it could not get back
/// ([`Noting`]), or, when clearing would leave more than half the room
/// used, or was not enough, summarizes the older messages. Notes too big
/// to ask give way to a summary of its own. A request whose summary
/// failed goes only if it fits; otherwise the turn fails. A request that
/// goes clamps its output limit to the room the window leaves.
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
    (default, disk): (Res<CompactionDefault>, Disk),
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
        let spec = &*connection.spec;
        let prompt = prompt.map(|prompt| prompt.text());
        let prompt = prompt.as_deref();
        let anchor = (last, sent.map(|sent| sent.0));
        let (mut used, mut start) = measure(&model, spec, prompt, anchor);
        let asked = output_budget(&model.request, spec);
        let over = |used| {
            default.0.over_threshold(used, spec)
                || asked.is_some_and(|asked| room(spec, asked, used) == 0)
        };
        if !over(used) {
            fit_output(&mut model, spec, used);
            commands.entity(agent).try_insert(Sent(start));
            continue;
        }
        let measured = Measure::of(&model, spec, prompt, used);
        let scale = measured.scale;
        let policy = &near_the_window(&default.0, scale);
        let conversation = &compacting.conversation;
        let plan = policy.clearing.plan(conversation);
        let after = used.saturating_sub(scaled(plan.cleared.tokens as u64, scale));
        let reserve = policy.summarizer.limits.reserve;
        let below =
            (spec.input_limit()).map_or(u64::MAX, |limit| u64::from(limit).saturating_sub(reserve));
        // Clearing is the first step, unless it would free too little.
        let checkpoint = match step {
            None => after > below / ENOUGH,
            Some(step) => *step < Recovering::NotSummarized,
        };
        if checkpoint
            && worth_summarizing(conversation, policy, spec)
            && compacting.start(
                (agent, turn, call),
                (policy, CompactReason::Threshold),
                ((&model, spec), Some(measured)),
                &mut commands,
                &mut notices,
            )
        {
            // The call is held, and built again once the summary is in.
            continue;
        }
        if step.is_none() && plan.cleared.results > 0 {
            let mut without = "";
            if compacting.notes_needed(&plan, policy, &disk)
                && let Some(through) = plan.through
            {
                let summarizer = &policy.summarizer;
                let (mut request, most) = ((*model).clone(), summarizer.max_tokens(spec));
                let ask = summarizer.prompts.notes.clone().into_owned();
                if ask_at(&mut request, Upto::Through(through), ask, most).is_some() {
                    if let Some(budget) = measured.budget(&request, most) {
                        request.request_mut().max_tokens = Some(budget);
                        commands.spawn((
                            Noting::call(CompactReason::Threshold, plan),
                            request,
                            ForAgent(agent),
                            CallOf(turn),
                            HoldOf(call),
                        ));
                        // The call is held, and built again once the notes are in.
                        continue;
                    }
                    // Notes too big to ask give way to a summary of its own.
                    if compacting.start(
                        (agent, turn, call),
                        (policy, CompactReason::Threshold),
                        ((&model, spec), None),
                        &mut commands,
                        &mut notices,
                    ) {
                        continue;
                    }
                    without = " without notes, as they would not fit";
                }
            }
            let cleared = plan.apply(&mut compacting.conversation, None, summary_author());
            // The request's copy too, as cleared by compaction.
            match &mut model.history {
                Some(history) => {
                    plan.apply(history, None, summary_author());
                }
                None => {
                    policy.clearing.clear(&mut model.request_mut().chat_history);
                }
            }
            commands.entity(call).try_insert(Recovering::Cleared);
            used = after;
            start = measure(&model, spec, prompt, anchor).1;
            notices.write(cleared_notice(
                agent,
                &CompactReason::Threshold,
                &cleared,
                without,
            ));
        }
        let answer = asked.unwrap_or(LEAST_OUTPUT);
        if step == Some(&Recovering::NotSummarized) && !fits(spec, used, answer) {
            let why = "The conversation does not fit the model's context window, and was not \
                       summarized";
            let why = ErrorReport::new(ErrorKind::Request, why);
            commands.entity(call).try_insert_if_new(Outcome(Err(why)));
            continue;
        }
        fit_output(&mut model, spec, used);
        commands.entity(agent).try_insert(Sent(start));
    }
}

/// Sets the output budget of `model`, a call to `spec` of `used` tokens,
/// clamped to the remaining room. With no room, leaves the request unchanged
/// for the provider to refuse.
fn fit_output(model: &mut ModelRequest, spec: &ModelSpec, used: u64) {
    let Some(asked) = output_budget(&model.request, spec) else {
        return;
    };
    let room = room(spec, asked, used);
    if room > 0 {
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
/// outputs and sends it again, when each can be had again or the policy
/// clears without notes; otherwise, or when they were cleared for it
/// already or none were, summarizes the older messages first, in a request
/// of its own to the same model, as the refused one would not fit even
/// cut. When nothing is left to summarize, it clears the outputs without
/// notes. Once a summary was tried, the turn fails.
fn recover_from_overflow(
    failed: Query<
        (
            Entity,
            (&ModelRequest, &Adapters),
            (&RetryVerdict, &Effect),
            (&CallOf, &ForAgent),
            Option<&Recovering>,
        ),
        (ReadyToSettle, With<TurnCall>, Without<SupersededBy>),
    >,
    mut agents: Query<Compacting>,
    starts: ModelStarts,
    (default, disk): (Res<CompactionDefault>, Disk),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, (model, adapters), (verdict, effect), (&CallOf(turn), &ForAgent(agent)), step) in
        &failed
    {
        if verdict.0 != Verdict::Overflow || step >= Some(&Recovering::NotSummarized) {
            continue;
        }
        let Ok(mut compacting) = agents.get_mut(agent) else {
            continue;
        };
        let Ok((connection, _)) = starts.prompt(model, Some(adapters), Some(agent)) else {
            continue;
        };
        let policy = &default.0;
        let plan = policy.clearing.plan(&compacting.conversation);
        let clears = step.is_none() && plan.cleared.results > 0;
        // A clearing that loses nothing the model could not read again
        // comes first, as it costs no model call.
        let summarizes = !clears || compacting.notes_needed(&plan, policy, &disk);
        let reason = (policy, CompactReason::Overflow);
        let held = ((model, &*connection.spec), None);
        let at = (agent, turn, call);
        if summarizes && compacting.start(at, reason, held, &mut commands, &mut notices) {
            notices.write(Notice::warn(
                agent,
                format!(
                    "{OUTGREW}: summarizing its older messages and sending it again; {CANCEL}."
                ),
            ));
            continue;
        }
        if clears {
            let cleared = plan.apply(&mut compacting.conversation, None, summary_author());
            let then = " and sending it again";
            notices.write(cleared_notice(
                agent,
                &CompactReason::Overflow,
                &cleared,
                then,
            ));
            let successor = (turn, agent, call);
            supersede(&mut commands, successor, effect.parent, Recovering::Cleared);
        }
    }
}

/// Sends a failed notes or summary call again once the wait passed, when
/// its verdict is a retry and its turn goes on: a hold with the
/// [`Backoff`] keeps it from settling until then, and the call sent again
/// ([`resend`]) takes its marker and its hold on the call it is for. The
/// retry counts toward its turn's [`Recovery`], whose count the verdict
/// weighs, so it waits as the turn's calls wait. A call that timed out is
/// sent again once ([`TimedOutBefore`]). An overflow is never sent again
/// as it was.
fn retry_compaction_calls(
    failed: Query<
        (
            Entity,
            &Outcome,
            &RetryVerdict,
            &CallOf,
            &ForAgent,
            Has<TimedOutBefore>,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            Without<TurnCall>,
            Without<SupersededBy>,
            Or<(With<Summarizing>, With<Noting>)>,
        ),
    >,
    ended: Query<(), With<TurnOutcome>>,
    mut commands: Commands,
) {
    for (call, Outcome(outcome), RetryVerdict(verdict), &CallOf(turn), &ForAgent(agent), again) in
        &failed
    {
        let (Err(report), Verdict::Retry(delay)) = (outcome, verdict) else {
            continue;
        };
        // Timed out again after a retry for a timeout: a third try, as big,
        // would likely time out too.
        let timed_out = report.kind == ErrorKind::Timeout;
        if ended.contains(turn) || (timed_out && again) {
            continue;
        }
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
            let tries = match timed_out {
                true => "once, as it timed out".to_owned(),
                false => format!("{}/{}", backoff.attempt, RETRY_POLICY.max_retries),
            };
            world.write_message(Notice::warn(
                agent,
                format!(
                    "The compaction call failed: {report}. Retrying in {}s ({tries}).",
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
                    let successor = world.get::<SupersededBy>(call).map(SupersededBy::successor);
                    if let Some(mut successor) = successor
                        .filter(|_| timed_out || again)
                        .and_then(|successor| world.get_entity_mut(successor).ok())
                    {
                        successor.insert(TimedOutBefore);
                    }
                });
        });
    }
}

/// On a notes or summary call sent again after one of its tries timed
/// out, whatever failed in between: one that times out again is not
/// retried.
#[derive(Component, Clone, Copy, Debug)]
struct TimedOutBefore;

/// Spawns the call that supersedes `call`, a turn's model call held while
/// the conversation was compacted for it, built anew from the conversation
/// on the recovery path's `step`, under the effect `parent`.
fn supersede(
    commands: &mut Commands,
    (turn, agent, call): (Entity, Entity, Entity),
    parent: Option<rig_core::effect::EffectId>,
    step: Recovering,
) {
    commands
        .spawn((
            TurnCall,
            CallOf(turn),
            ForAgent(agent),
            Effect::with_parent_id(parent),
            step,
        ))
        .add_related::<SupersededBy>(&[call]);
}

/// Asks each summary call the kernel built like a turn's request
/// ([`SideCall`]) in place ([`ask_at`]): cut before the first message it
/// keeps, with the instruction appended, so the provider reads the agent's
/// conversation from its prompt cache and the model reads every output
/// whole. One that would not fit the model with room for its answer, as
/// the provider counted the agent's last request, is a summary of its own
/// instead ([`plan_alone`]), sent again as it is when retried; with none to
/// be, it is not sent. A call whose kept message was taken out is answered
/// with that; one a plugin brought whole is left as it is.
fn ask_in_place(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &Adapters,
            &SideCall,
            &Summarizing,
        ),
        Added<Admitted>,
    >,
    agents: Query<(&Conversation, &LastUsage, Option<&Sent>)>,
    starts: ModelStarts,
    default: Res<CompactionDefault>,
    mut commands: Commands,
) {
    for (call, mut model, adapters, &SideCall(agent), summarizing) in &mut calls {
        if model.history.is_none() {
            // A plugin brought it whole, such as from `plan`: it asks itself.
            continue;
        }
        let Ok((connection, prompt)) = starts.prompt(&model, Some(adapters), Some(agent)) else {
            continue;
        };
        let Ok((conversation, last, sent)) = agents.get(agent) else {
            continue;
        };
        let spec = &*connection.spec;
        let prompt = prompt.map(|prompt| prompt.text());
        let prompt = prompt.as_deref();
        let used = measure(&model, spec, prompt, (last, sent.map(|sent| sent.0))).0;
        let measured = Measure::of(&model, spec, prompt, used);
        let route = model.connection.clone();
        let policy = &default.0;
        let ask = (&policy.summarizer, focus(&summarizing.reason));
        let upto = Upto::Before(summarizing.keep_from);
        match ask_summary(&mut model, upto, ask, measured) {
            Some(true) => continue,
            Some(false) => {}
            None => {
                let why = "the messages it was to read were taken out";
                let why = ErrorReport::new(ErrorKind::Internal, why);
                commands.entity(call).try_insert_if_new(Outcome(Err(why)));
                continue;
            }
        }
        let reason = summarizing.reason.clone();
        let why = match plan_alone(conversation, policy, reason, (route.as_ref(), spec)) {
            Ok(Some((summarizing, request))) => {
                *model = request;
                // Without it a retry is built again, as the request that did not fit.
                commands
                    .entity(call)
                    .try_remove::<SideCall>()
                    .try_insert(summarizing);
                continue;
            }
            Ok(None) => "nothing older is left to summarize on its own".to_owned(),
            Err(why) => why,
        };
        let why = format!("its request would not fit the model, and {why}");
        let why = ErrorReport::new(ErrorKind::Request, why);
        commands.entity(call).try_insert_if_new(Outcome(Err(why)));
    }
}

/// Marks each summary or notes call (`M`) as this plugin's.
fn provided_by_compaction<M: Component>(
    added: On<Add<M>>,
    me: Option<Res<PluginEntity<CompactionPlugin>>>,
    mut commands: Commands,
) {
    if let Some(me) = me {
        commands
            .entity(added.entity)
            .try_insert_if_new(ProvidedBy(me.entity()));
    }
}

/// Reads each answered summary or notes call's text from its reply, once.
/// Neither is ever a turn's own call, which other reviewers edit.
fn read_replies(
    mut calls: Query<
        (
            Entity,
            Option<&mut Summarizing>,
            Option<&mut Noting>,
            &Outcome,
        ),
        (
            ReadyToSettle,
            ModelCalls,
            Without<TurnCall>,
            Without<SummaryRead>,
            Or<(With<Summarizing>, With<Noting>)>,
        ),
    >,
    mut commands: Commands,
) {
    for (call, summarizing, noting, outcome) in &mut calls {
        let response = outcome.completion();
        let text = response.and_then(|reply| Summarizer::summary_text(reply).ok());
        if let Some(mut summarizing) = summarizing {
            summarizing.summary = text;
        } else if let Some(mut noting) = noting {
            noting.notes = text;
        }
        commands.entity(call).try_insert(SummaryRead);
    }
}

/// The text a summary or notes call settled with, `read` from its reply or
/// left by a reviewer, or why there is none.
fn taken(outcome: &Outcome, read: Option<&String>) -> Result<String, NotTaken> {
    let response = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => return Err(NotTaken::Failed(report.to_string())),
        (Ok(_), None) => {
            let why = "the model did not answer with a completion";
            return Err(NotTaken::Failed(why.to_owned()));
        }
        (Ok(_), Some(response)) => response,
    };
    match read {
        Some(text) => Ok(text.clone()),
        // The reply had one, so a reviewer took it out.
        None => match Summarizer::summary_text(response) {
            Ok(_) => Err(NotTaken::Cancelled),
            Err(why) => Err(NotTaken::Failed(why.to_string())),
        },
    }
}

/// Takes a settled summary: it replaces the summarized messages in the
/// conversation as an entry of its own, a splice its log records. A failed
/// summary, one a reviewer cancelled, one whose turn was stopped or failed
/// first, even once answered, or one whose kept message is gone, replaces
/// nothing, and is never asked again; a failed one that held a call clears
/// old tool outputs instead. Either way the call it held is sent again,
/// built anew, or the turn ends when the user asked for the compaction.
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(
        (&CallOf, &ForAgent),
        (&Summarizing, &Outcome),
        Option<&HoldOf>,
        Option<&Fallback>,
    )>,
    held: Query<&Effect>,
    ended: Query<(), With<TurnOutcome>>,
    mut agents: Query<&mut Conversation>,
    default: Res<CompactionDefault>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(((&CallOf(turn), &ForAgent(agent)), (summarizing, outcome), hold, fallback)) =
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
    let ended = ended.contains(turn);
    let summary = match ended {
        true => Err(NotTaken::Cancelled),
        false => taken(outcome, summarizing.summary.as_ref()),
    };
    let summary = summary.and_then(|summary| {
        let end = end.ok_or_else(|| failed("the messages it kept were taken out"))?;
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
    let (step, changed) = match summary {
        Ok(((older, older_tokens), (kept, kept_tokens))) => {
            let left = estimate_tokens(conversation.messages());
            let text = compacted_text((older, older_tokens), (kept, kept_tokens), left);
            notices.write(Notice::info(agent, text));
            (Recovering::Compacted, true)
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
            (Recovering::NotSummarized, false)
        }
        Err(NotTaken::Failed(why)) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
            let clearing = fallback.map_or(&default.0.clearing, |it| &it.0);
            let plan = hold.map(|_| clearing.plan(&conversation));
            let plan = plan.filter(|plan| plan.cleared.results > 0);
            if let Some(plan) = &plan {
                let cleared = plan.apply(&mut conversation, None, summary_author());
                let reason = &summarizing.reason;
                notices.write(cleared_notice(agent, reason, &cleared, " instead"));
            }
            (Recovering::NotSummarized, plan.is_some())
        }
    };
    match hold {
        // Built again unchanged, the refused call would be refused again:
        // it fails as it is.
        Some(&HoldOf(call)) if !changed && summarizing.reason == CompactReason::Overflow => {
            commands.entity(call).try_insert(step);
        }
        Some(&HoldOf(call)) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            supersede(&mut commands, (turn, agent, call), parent, step);
        }
        // The user's own compaction turn ends with it, unless it ended.
        None if !ended => {
            let compacted = TurnOutcome::Stopped(Some("compacted".to_owned()));
            commands.entity(turn).try_insert(compacted);
        }
        None => {}
    }
}

/// Takes a settled notes call: its plan clears the outputs, with the notes
/// after them, by one splice its log records. A failed call clears them
/// without notes; one a reviewer cancelled, or one whose turn was stopped
/// or failed first, even once answered, clears nothing. Either way the call
/// it held is sent again, built anew.
fn take_notes(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ForAgent, &Noting, &Outcome, Option<&HoldOf>)>,
    held: Query<&Effect>,
    ended: Query<(), With<TurnOutcome>>,
    mut agents: Query<&mut Conversation>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CallOf(turn), &ForAgent(agent), noting, outcome, hold)) = calls.get(settled.entity)
    else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let (by, plan) = (summary_author(), &noting.plan);
    let notes = match ended.contains(turn) {
        true => Err(NotTaken::Cancelled),
        false => taken(outcome, noting.notes.as_ref()),
    };
    let step = match notes {
        Ok(notes) => {
            let cleared = plan.apply(&mut conversation, Some(&notes), by);
            let then = ", after the model noted what they held";
            notices.write(cleared_notice(agent, &noting.reason, &cleared, then));
            Recovering::Cleared
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Clearing cancelled."));
            Recovering::NotSummarized
        }
        Err(NotTaken::Failed(why)) => {
            let cleared = plan.apply(&mut conversation, None, by);
            let then = format!(" without notes: {why}");
            notices.write(cleared_notice(agent, &noting.reason, &cleared, &then));
            Recovering::Cleared
        }
    };
    if let Some(&HoldOf(call)) = hold {
        let parent = held.get(call).ok().and_then(|effect| effect.parent);
        supersede(&mut commands, (turn, agent, call), parent, step);
    }
}

/// Why a summary or notes call's text went in nowhere.
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
