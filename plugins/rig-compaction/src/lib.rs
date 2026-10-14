//! Compaction: when a turn's next request nears the model's context window,
//! older tool outputs are cleared, and when that is not enough the older
//! messages are summarized by the model into rig-memory's checkpoint, which
//! takes their place in the conversation. Both are splices of the
//! conversation by [`summary_author`], so they are logged, restored and
//! announced as [`Committed`]; the summary is an entry of its own by that
//! author. The same happens after the model refused a
//! request as too long, and `/compact <focus>` does it now. It is stage systems over the turn's model calls
//! ([`TurnCall`]): a Prepare system measures each request against the
//! window, and a Review system takes over a call whose [`RetryVerdict`]
//! says it overflowed. The summary is a [`ModelRequest`] of the turn, so
//! interrupting the turn cancels it, that holds the call it compacts for
//! until the summary is in; a new call built from the compacted
//! conversation then supersedes it. The Prepare system is in
//! [`PrepareSystems::Fit`], after every plugin's edits, so the request is
//! measured as it will be sent; the Review system is in
//! [`ReviewSystems::Act`].
//!
//! A plugin reviews a summary before it replaces anything: once the call
//! is answered, its [`Summarizing::summary`] holds the model's text in
//! [`ReviewSystems::Act`], where a reviewer may hold the call
//! ([`HoldOf`]) until someone decided, replace the text, or take it out to
//! cancel the compaction. Every compaction keeps what it replaced
//! (rig-memory's [`Compacted`](rig_memory::Compacted)), and [`Uncompact`]
//! puts the newest one's messages back with their ids.

use bevy_ecs::query::QueryData;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, Usage, tokens_label};
use rig_ecs::journal::SessionStore;
use rig_ecs::prelude::*;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::successor::follow_successor;
use rig_harness::{AppKeepsExt, PluginEntity, ProvidedBy};
use rig_memory::{
    Cleared, CompactReason, CompactionPolicy, Summarizer, SummaryPlan, SummaryState, TrackArgument,
    apply_summary, undo_summary,
};
/// Who writes a compaction's summary, and how many of a conversation's
/// first entries are summaries already: rig-memory's, which this plugin
/// compacts by.
pub use rig_memory::{summarized_prefix, summary_author, summary_in};
use serde::{Deserialize, Serialize};

/// Compacts agents' conversations near their model's window, after an
/// overflow, and on `/compact`, by the [`Compaction`] policy.
#[derive(Default)]
pub struct CompactionPlugin;

impl Plugin for CompactionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Compaction(CompactionPolicy {
            tracked: TRACKED.to_vec(),
            ..CompactionPolicy::default()
        }))
        .register_required_components::<TurnOf, Compactions>()
        .keeps_saved::<Summarized>()
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
        .add_observer(provided_by_compaction);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// How agents are compacted: rig-memory's default policy, tracking the
/// files the built-in file tools read and changed. A plugin whose tool
/// reads or changes files tracks its argument the same way, by adding a
/// [`TrackArgument`] to `tracked` (from `finish` or a startup system). A
/// plugin chooses where a compaction cuts with the policy's `cut` (a
/// [`Cut`](rig_memory::Cut)), given the conversation's messages and how
/// many of the first are summaries ([`summarized_prefix`]), and how many
/// summaries a turn may ask for with `per_turn`.
#[derive(Resource, Clone, Debug)]
pub struct Compaction(pub CompactionPolicy);

/// The files the built-in file tools (rig-coding-tools') were called with:
/// `read`'s are read, `edit`'s and `write`'s changed (a later set wins in
/// the summary).
const TRACKED: &[TrackArgument<'static>] = &[
    TrackArgument {
        tool: "read",
        argument: "path",
        set: "read-files",
    },
    TrackArgument {
        tool: "edit",
        argument: "path",
        set: "modified-files",
    },
    TrackArgument {
        tool: "write",
        argument: "path",
        set: "modified-files",
    },
];

/// Compact the agent's conversation now: its messages up to the small tail
/// the [`Compaction`] policy keeps (by default the newest reply alone) are
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
/// ([`undo_summary`]); the summary state goes back to the compaction's
/// before it. Refused while a turn runs or in a session that is not
/// saved.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Uncompact {
    /// The agent.
    pub entity: Entity,
}

/// The agent's summary so far and the files it tracks, which the next
/// compaction builds on. Saved with the session. A compaction inserts it
/// after it spliced the summary into the conversation, so an
/// `On<Insert<Summarized>>` observer sees the conversation with it.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Summarized(pub SummaryState);

/// What a turn did to fit the window: whether it cleared tool outputs
/// after an overflow, and how many summaries it asked for.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Compactions {
    /// Whether an overflow cleared tool outputs.
    pub cleared: bool,
    /// Summaries asked for, at most the policy's `per_turn`.
    pub summaries: u32,
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
    pub upto: Option<MessageId>,
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

/// An agent as compacting it reads it: its messages, its summary so far,
/// and its model.
#[derive(QueryData)]
#[query_data(mutable)]
struct Compacting {
    conversation: &'static mut Conversation,
    summarized: Option<&'static Summarized>,
    connection: Option<&'static Connection>,
}

/// Plans one summary call of `conversation` with `policy`, for `reason`,
/// written by the model `spec`, building on `summarized`: the
/// [`Summarizing`] to spawn it with and its request. `None` when no message
/// would be summarized that is not already; an error says why the policy
/// cannot plan it. rig-compaction's own compactions are planned so with the
/// [`Compaction`] policy; a plugin that summarizes with another, such as a
/// fallback, replaces a summary call's [`Summarizing`] and
/// [`ModelRequest`] with what this returns.
pub fn plan(
    conversation: &rig_core::conversation::Conversation,
    summarized: Option<&Summarized>,
    policy: &CompactionPolicy,
    reason: CompactReason,
    spec: &ModelSpec,
) -> Result<Option<(Summarizing, CompletionRequest)>, String> {
    let none = SummaryState::default();
    let state = summarized.map_or(&none, |summarized| &summarized.0);
    let planned = policy.plan_summary(conversation, state, &reason, spec);
    let Some(SummaryPlan {
        upto,
        state,
        request,
    }) = planned.map_err(|why| why.to_string())?
    else {
        return Ok(None);
    };
    let summarizing = Summarizing {
        reason,
        upto,
        state,
        summary: None,
    };
    Ok(Some((summarizing, request)))
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
enum Start {
    /// Nothing: the agent has no model to write the summary.
    NoModel,
    /// Nothing: no message would be summarized that is not already.
    NothingNew,
    /// Nothing: the summary could not be planned, and the agent was told
    /// why. It counts as a compaction of the turn.
    Refused,
    /// The summary call started.
    Started,
}

impl CompactingItem<'_, '_> {
    /// Clears the older tool outputs of the agent's conversation, by a
    /// splice of the messages that changed, which keeps their ids.
    fn clear_old_outputs(&mut self, policy: &CompactionPolicy) -> Cleared {
        let clearing = &policy.clearing;
        clearing.clear_conversation(&mut self.conversation, summary_author())
    }

    /// Starts the summary call of a compaction of `agent` for `reason` on
    /// `turn`, holding the model call `call`, or on a turn of its own when
    /// `None`, when it has a model and anything new would be summarized.
    fn start(
        &self,
        (agent, turn): (Entity, Option<(Entity, Entity)>),
        (policy, reason): (&CompactionPolicy, CompactReason),
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) -> Start {
        let Some(connection) = self.connection else {
            return Start::NoModel;
        };
        let planned = plan(
            &self.conversation,
            self.summarized,
            policy,
            reason,
            &connection.spec,
        );
        let (summarizing, request) = match planned {
            Ok(Some(planned)) => planned,
            Ok(None) => return Start::NothingNew,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
                return Start::Refused;
            }
        };
        let summary = (
            Name::new("compacting"),
            summarizing,
            ModelRequest::new(request),
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
        Start::Started
    }
}

/// Compacts an idle agent's conversation on the user's request, in a turn
/// of its own that ends with the summary.
fn on_compact(
    compact: On<Compact>,
    mut agents: Query<(Compacting, Has<ActiveTurn>), With<Agent>>,
    policy: Res<Compaction>,
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
        (&policy.0, reason),
        &mut commands,
        &mut notices,
    ) {
        Start::NoModel => Notice::no_model(agent),
        Start::NothingNew => Notice::info(agent, "Nothing to compact yet."),
        Start::Refused | Start::Started => return,
    };
    notices.write(refusal);
}

/// Before a turn's request that leaves less than the reserve of the
/// window of the model it goes to free, clears old tool outputs, which
/// costs no model call, and asks for a summary only when clearing was not
/// enough. The context in use is the last reply's or the request's
/// estimate, as that model gets it, whichever is larger.
fn compact_near_the_window(
    mut calls: Query<
        (Entity, &mut ModelRequest, &Adapters, &CallOf, &ForAgent),
        (Added<Admitted>, With<TurnCall>),
    >,
    mut agents: Query<(Compacting, &mut LastUsage)>,
    mut turns: Query<&mut Compactions>,
    starts: ModelStarts,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, adapters, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok(mut compactions) = turns.get_mut(turn) else {
            continue;
        };
        let Ok((mut compacting, mut last)) = agents.get_mut(agent) else {
            continue;
        };
        if compactions.summaries >= policy.0.per_turn {
            continue;
        }
        let Ok((connection, sent, _)) = starts.sent(&model, Some(adapters), Some(agent)) else {
            continue;
        };
        let (policy, spec) = (&policy.0, &*connection.spec);
        let sent = policy.estimate(&sent.chat_history);
        let used = last.context().unwrap_or(0).max(sent);
        if !policy.over_threshold(used, spec) {
            continue;
        }
        let cleared = compacting.clear_old_outputs(policy);
        let left = used.saturating_sub(cleared.tokens as u64);
        if cleared.results > 0 {
            // The request's copy too, as cleared by compaction.
            match &mut model.history {
                Some(history) => {
                    policy
                        .clearing
                        .clear_conversation(history, summary_author());
                }
                None => {
                    policy.clearing.clear(&mut model.request.chat_history);
                }
            }
            *last = LastUsage(Some(Usage::new().total_tokens(left)));
            let nears = "The conversation nears the model's context window";
            notices.write(cleared_notice(agent, nears, &cleared, ""));
        }
        if !policy.over_threshold(left, spec) {
            continue;
        }
        let reason = (policy, CompactReason::Threshold);
        let at = (agent, Some((turn, call)));
        let started = compacting.start(at, reason, &mut commands, &mut notices);
        if matches!(started, Start::Refused | Start::Started) {
            compactions.summaries += 1;
        }
    }
}

/// After the model refused a turn's request as too long, clears old tool
/// outputs and sends it again; when that was done already or cleared
/// nothing, summarizes the older messages first. Otherwise the turn fails.
fn recover_from_overflow(
    failed: Query<
        (Entity, &RetryVerdict, &Effect, &CallOf, &ForAgent),
        (Unreviewed, With<TurnCall>, Without<SupersededBy>),
    >,
    mut agents: Query<Compacting>,
    mut turns: Query<&mut Compactions>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, verdict, effect, &CallOf(turn), &ForAgent(agent)) in &failed {
        if verdict.0 != Verdict::Overflow {
            continue;
        }
        let Ok(mut compactions) = turns.get_mut(turn) else {
            continue;
        };
        let Ok(mut compacting) = agents.get_mut(agent) else {
            continue;
        };
        let policy = &policy.0;
        let outgrew = "The conversation outgrew the model's context window";
        if !compactions.cleared {
            compactions.cleared = true;
            let cleared = compacting.clear_old_outputs(policy);
            if cleared.results > 0 {
                let then = " and sending it again";
                notices.write(cleared_notice(agent, outgrew, &cleared, then));
                commands
                    .spawn((
                        TurnCall,
                        CallOf(turn),
                        ForAgent(agent),
                        Effect::child_of(effect.parent),
                    ))
                    .add_related::<SupersededBy>(&[call]);
                continue;
            }
        }
        if compactions.summaries >= policy.per_turn {
            continue;
        }
        let reason = (policy, CompactReason::Overflow);
        let at = (agent, Some((turn, call)));
        match compacting.start(at, reason, &mut commands, &mut notices) {
            Start::NoModel | Start::NothingNew => {}
            Start::Refused => compactions.summaries += 1,
            Start::Started => {
                compactions.summaries += 1;
                notices.write(Notice::info(
                    agent,
                    format!("{outgrew}: summarizing its older messages and sending it again."),
                ));
            }
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
            Unreviewed,
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
    mut agents: Query<(&mut Conversation, &mut LastUsage)>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&CallOf(turn), &ForAgent(agent), summarizing, outcome, hold)) =
        calls.get(settled.entity)
    else {
        return;
    };
    let Ok((mut conversation, mut last)) = agents.get_mut(agent) else {
        return;
    };
    let end = match summarizing.upto {
        None => Some(conversation.messages().len()),
        Some(kept) => conversation.index_of(kept),
    };
    // Why nothing is replaced; `None` when a reviewer cancelled.
    let failed = |why: &str| Some(why.to_owned());
    let summary = match (&outcome.0, outcome.completion(), end) {
        (Err(report), ..) => Err(Some(report.to_string())),
        (Ok(_), None, _) => Err(failed("the model did not answer with a completion")),
        (Ok(_), Some(_), None) => Err(failed("the messages it kept were taken out")),
        (Ok(_), Some(response), Some(end)) => match &summarizing.summary {
            Some(summary) => Ok((summary.clone(), end)),
            // The reply had one, so a reviewer took it out.
            None => match Summarizer::summary_text(response) {
                Ok(_) => Err(None),
                Err(why) => Err(Some(why.to_string())),
            },
        },
    };
    let summary = summary.and_then(|(summary, end)| {
        let messages = conversation.messages();
        let older = messages.get(..end).unwrap_or_default();
        let older = (older.len(), policy.0.estimate(older));
        let kept = messages.get(end..).unwrap_or_default();
        let kept = (kept.len(), policy.0.estimate(kept));
        let state = SummaryState {
            summary,
            ..summarizing.state.clone()
        };
        match apply_summary(&mut conversation, summarizing.upto, &state) {
            Some(_) => Ok((state, older, kept)),
            None => Err(failed("the summary was empty")),
        }
    });
    match summary {
        Ok((state, (older, older_tokens), (kept, kept_tokens))) => {
            let left = policy.0.estimate(conversation.messages());
            *last = LastUsage(Some(Usage::new().total_tokens(left)));
            commands.entity(agent).insert(Summarized(state));
            let text = compacted_text((older, older_tokens), (kept, kept_tokens), left);
            notices.write(Notice::info(agent, text));
        }
        Err(None) => {
            notices.write(Notice::info(agent, "Compaction cancelled."));
        }
        Err(Some(why)) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
        }
    }
    match hold {
        // The call it held is built again, from the compacted conversation.
        Some(&HoldOf(call)) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            commands
                .spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::child_of(parent),
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

/// Undoes the agent's newest compaction on [`Uncompact`].
fn on_uncompact(
    uncompact: On<Uncompact>,
    mut agents: Query<(&AgentId, &mut Conversation, &mut LastUsage, Has<ActiveTurn>)>,
    store: Option<Res<SessionStore>>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = uncompact.entity;
    let Ok((id, mut conversation, mut last, busy)) = agents.get_mut(agent) else {
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
            let left = policy.0.estimate(conversation.messages());
            *last = LastUsage(Some(Usage::new().total_tokens(left)));
            commands.entity(agent).insert(Summarized(undone.state));
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
