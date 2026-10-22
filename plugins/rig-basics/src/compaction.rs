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
//! - **How:** older tool outputs are cleared first, all but the newest
//!   20k tokens of them as the provider counts them, so a clearing, which
//!   rewrites what the prompt cache holds, comes seldom. Before that the
//!   model writes notes on the ones it could not get back ([`Noting`]),
//!   which stay after them; a file still on disk can be read again. If
//!   clearing would still leave more than half the room used, the model
//!   summarizes the older messages into a checkpoint instead, built on the
//!   [`summary_state`], when that takes out at least a quarter of the
//!   conversation. Each is a splice by [`summary_author`], so it is logged,
//!   restored and announced as [`ConversationChanged`]. A notes or summary
//!   call is a model call of the turn holding the call it is for
//!   ([`HoldOf`]), so stopping the turn cancels it; a call built from the
//!   changed conversation then supersedes the held one. It is the agent's
//!   own request cut where its reading ends, plus one instruction, so the
//!   provider reads it from its prompt cache and the model reads every
//!   output whole; only the summary after an overflow is a request of its
//!   own ([`plan`]).
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
use bevy_ecs::system::SystemParam;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{self, CompletionRequest, tokens_label};
use rig_core::conversation::Entry;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::rig_tools::workspace::Stat;
use rig_harness::services::{ProjectRoot, WorkDir};
use rig_harness::settings::AppSettingsExt;
use rig_harness::workspace::ProjectWorkspace;
use rig_harness::{PluginEntity, ProvidedBy};
use rig_memory::{
    ClearPlan, Cleared, CompactReason, CompactionPolicy, NearWindow, Prepared, RequestStart,
    Summarizer, SummaryCut, SummaryPlan, SummaryState, TrackArgument, apply_summary,
    estimate_tokens, next_request_tokens, output_room, undo_summary,
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
                recover_from_overflow.in_set(ReviewSystems::Act),
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
        (Name::new("noting"), noting, effect)
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

/// Plans one summary call of `conversation` for `reason` as `held`, the
/// turn's own call as prepared for `spec`, cut before the first message
/// kept with the instruction of `policy`'s summarizer appended
/// ([`ask_at`]). A request a plugin brought whole gets a request of its
/// own ([`plan`]). `None` when nothing new would be summarized.
fn plan_in_place(
    conversation: &rig_core::conversation::Conversation,
    policy: &CompactionPolicy,
    reason: CompactReason,
    (held, spec): (&ModelRequest, &ModelSpec),
) -> Option<(Summarizing, ModelRequest)> {
    let summarizing = cut(conversation, policy, reason, spec)?;
    let mut request = held.clone();
    let ask = policy.summarizer.instruction(focus(&summarizing.reason));
    let most = policy.summarizer.max_tokens(spec);
    if ask_at(&mut request, Upto::Before(summarizing.keep_from), ask, most).is_none() {
        let (summarizing, request) =
            plan(conversation, policy, summarizing.reason, spec).ok()??;
        return Some((summarizing, ModelRequest::new(request)));
    }
    Some((summarizing, request))
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

/// What the agent is told a clearing near the window is for.
const NEARS: &str = "The conversation nears the model's context window";

/// What the agent is told a clearing or summary after an overflow is for.
const OUTGREW: &str = "The conversation outgrew the model's context window";

impl CompactingItem<'_, '_> {
    /// Whether clearing `plan` by `policy` asks the model for notes first:
    /// near the window it summarizes, and not every output it clears can
    /// be had again ([`Disk::all_again`]).
    fn notes_needed(&self, plan: &ClearPlan, policy: &CompactionPolicy, disk: &Disk) -> bool {
        policy.near_window == NearWindow::Summarize
            && !disk.all_again(plan, &self.conversation, &policy.tracked, self.dir)
    }

    /// Starts the summary call of a compaction of `agent` for `reason` on
    /// `turn`, holding its model call `call`, when the agent has a model
    /// and anything new would be summarized; says whether it started. With
    /// `held`, the held call's prepared request and the model it goes to,
    /// the summary call is that request asked in place
    /// ([`plan_in_place`]); without, a request of its own ([`plan`]).
    fn start(
        &self,
        (agent, turn, call): (Entity, Entity, Entity),
        (policy, reason): (&CompactionPolicy, CompactReason),
        held: Option<(&ModelRequest, &ModelSpec)>,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) -> bool {
        let Some(connection) = self.connection else {
            return false;
        };
        let planned = match held {
            Some(held) => Ok(plan_in_place(&self.conversation, policy, reason, held)),
            None => plan(&self.conversation, policy, reason, &connection.spec)
                .map(|planned| planned.map(|(it, request)| (it, ModelRequest::new(request)))),
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
            Name::new("compacting"),
            summarizing,
            request,
            Effect::default().with_purpose("compaction"),
            ForAgent(agent),
            CallOf(turn),
            HoldOf(call),
        ));
        true
    }
}

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
        Name::new("compacting"),
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

/// Before a turn's request that leaves less than the reserve of the input
/// limit of the model it goes to free, or no room for an answer, takes
/// the next step of the call's recovery path ([`Recovering`]): clears old
/// tool outputs, after the model's notes on those it could not get back
/// ([`Noting`]), or, when clearing would leave more than half the room
/// used, or was not enough, summarizes the older messages. A request that
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
        let asked = model.request.max_tokens;
        let over =
            |used| default.0.over_threshold(used, spec) || room(spec, asked, used) == Some(0);
        if !over(used) {
            fit_output(&mut model, spec, used);
            commands.entity(agent).try_insert(Sent(start));
            continue;
        }
        let whole = measure(&model, spec, prompt, (last, None)).0;
        let scale = provider_scale(Some(used), whole);
        let policy = &near_the_window(&default.0, scale);
        let conversation = &compacting.conversation;
        let plan = policy.clearing.plan(conversation);
        let after = used.saturating_sub(scaled(plan.cleared.tokens as u64, scale));
        let reserve = policy.summarizer.limits.reserve;
        let below = (spec.input_limit())
            .map_or(u64::MAX, |limit| u64::from(limit).saturating_sub(reserve));
        // Clearing is the first step, unless it would free too little.
        let checkpoint = match step {
            None => after > below / ENOUGH,
            Some(step) => *step < Recovering::Compacted,
        };
        if checkpoint
            && worth_summarizing(conversation, policy, spec)
            && compacting.start(
                (agent, turn, call),
                (policy, CompactReason::Threshold),
                Some((&model, spec)),
                &mut commands,
                &mut notices,
            )
        {
            // The call is held, and built again once the summary is in.
            continue;
        }
        if step.is_none() && plan.cleared.results > 0 {
            if compacting.notes_needed(&plan, policy, &disk)
                && let Some(through) = plan.through
            {
                let summarizer = &policy.summarizer;
                let (mut request, most) = ((*model).clone(), summarizer.max_tokens(spec));
                let ask = summarizer.prompts.notes.clone().into_owned();
                if ask_at(&mut request, Upto::Through(through), ask, most).is_some() {
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
            notices.write(cleared_notice(agent, NEARS, &cleared, ""));
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
/// outputs, after the model's notes on those it could not get back
/// ([`Noting`]), and sends it again; when they were cleared for it already
/// or none were, summarizes the older messages first. On the last step the
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
    (default, disk): (Res<CompactionDefault>, Disk),
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
        if step.is_none() {
            let plan = policy.clearing.plan(&compacting.conversation);
            if plan.cleared.results > 0 {
                if compacting.notes_needed(&plan, policy, &disk) {
                    // Built like the refused request and cut after the newest
                    // output it clears, so the newest output, which usually
                    // made it too long, is left out.
                    commands.spawn((
                        Noting::call(CompactReason::Overflow, plan),
                        SideCall(agent),
                        ForAgent(agent),
                        CallOf(turn),
                        HoldOf(call),
                    ));
                    continue;
                }
                let cleared = plan.apply(&mut compacting.conversation, None, summary_author());
                let then = " and sending it again";
                notices.write(cleared_notice(agent, OUTGREW, &cleared, then));
                let successor = (turn, agent, call);
                supersede(&mut commands, successor, effect.parent, Recovering::Cleared);
                continue;
            }
        }
        let reason = (policy, CompactReason::Overflow);
        let at = (agent, turn, call);
        // The refused request does not fit, so neither would it with an
        // instruction: the summary gets a request of its own.
        if compacting.start(at, reason, None, &mut commands, &mut notices) {
            notices.write(Notice::info(
                agent,
                format!("{OUTGREW}: summarizing its older messages and sending it again."),
            ));
        }
    }
}

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

/// Asks each summary or notes call the kernel built like a turn's request
/// ([`SideCall`]) in place ([`ask_at`]): cut at its boundary, with the
/// instruction appended, so the provider reads the agent's conversation
/// from its prompt cache and the model reads every output whole. A call
/// whose boundary was taken out is answered with that; one a plugin
/// brought whole is left as it is.
fn ask_in_place(
    mut calls: Query<
        (
            Entity,
            &mut ModelRequest,
            &Adapters,
            &SideCall,
            Option<&Summarizing>,
            Option<&Noting>,
        ),
        (Added<Admitted>, Or<(With<Summarizing>, With<Noting>)>),
    >,
    starts: ModelStarts,
    default: Res<CompactionDefault>,
    mut commands: Commands,
) {
    for (call, mut model, adapters, &SideCall(agent), summarizing, noting) in &mut calls {
        if model.history.is_none() {
            // A plugin brought it whole, such as from `plan`: it asks itself.
            continue;
        }
        let Ok((connection, _)) = starts.prompt(&model, Some(adapters), Some(agent)) else {
            continue;
        };
        let summarizer = &default.0.summarizer;
        let most = summarizer.max_tokens(&connection.spec);
        let asked = if let Some(it) = summarizing {
            let ask = summarizer.instruction(focus(&it.reason));
            ask_at(&mut model, Upto::Before(it.keep_from), ask, most)
        } else {
            let through = noting.and_then(|it| it.plan.through);
            let ask = summarizer.prompts.notes.clone().into_owned();
            through.and_then(|through| ask_at(&mut model, Upto::Through(through), ask, most))
        };
        if asked.is_none() {
            let why = "the messages it was to read were taken out";
            let why = ErrorReport::new(ErrorKind::Internal, why);
            commands.entity(call).try_insert_if_new(Outcome(Err(why)));
        }
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
/// nothing. Either way the call it held is sent again, built anew, or the
/// turn ends when the user asked for the compaction.
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ForAgent, &Summarizing, &Outcome, Option<&HoldOf>)>,
    held: Query<&Effect>,
    ended: Query<(), With<TurnOutcome>>,
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
            supersede(
                &mut commands,
                (turn, agent, call),
                parent,
                Recovering::Compacted,
            );
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
    let situation = match noting.reason {
        CompactReason::Overflow => OUTGREW,
        _ => NEARS,
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
            notices.write(cleared_notice(agent, situation, &cleared, then));
            Recovering::Cleared
        }
        Err(NotTaken::Cancelled) => {
            notices.write(Notice::info(agent, "Clearing cancelled."));
            Recovering::Compacted
        }
        Err(NotTaken::Failed(why)) => {
            let cleared = plan.apply(&mut conversation, None, by);
            let then = format!(" without notes: {why}");
            notices.write(cleared_notice(agent, situation, &cleared, &then));
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
