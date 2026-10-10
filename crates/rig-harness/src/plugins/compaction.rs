//! Compaction: when a turn's next request nears the model's context window,
//! older tool outputs are cleared, and when that is not enough the older
//! messages are summarized by the model into rig-memory's checkpoint, which
//! requests send in their place as the agent's [`Condensed`]. The same
//! happens after the model refused a request as too long, and `/compact
//! <focus>` does it now. It uses the kernel's turn hooks only:
//! [`PrepareRequest`] for the window, [`ModelFailed`] for a refused request,
//! and a [`ModelRequest`] for the summary, a call of the turn, so
//! interrupting the turn cancels it.

use rig_core::catalog::ModelSpec;
use rig_core::completion::{Message, UnsupportedOption, Usage};
use rig_ecs::prelude::*;
use rig_ecs::usage;
use rig_memory::{
    Cleared, CompactReason, CompactionPolicy, Summarizer, SummaryState, TrackArgument,
};
use serde::{Deserialize, Serialize};

/// Compactions a turn may make, asked for or not.
pub const MAX_COMPACTIONS: u32 = 2;

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
        .add_command(
            "compact",
            "Summarize all but the newest reply to free context; /compact <focus> says what to keep",
            compact,
        )
        .add_observer(on_compact)
        .add_observer(compact_near_the_window)
        .add_observer(recover_from_overflow)
        .add_observer(take_summary);
    }
}

/// How agents are compacted: rig-memory's default policy, tracking the
/// files the built-in file tools read and changed.
#[derive(Resource, Clone, Debug)]
pub struct Compaction(pub CompactionPolicy);

/// The files the built-in file tools were called with: `read`'s are read,
/// `edit`'s and `write`'s changed (a later set wins in the summary).
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

/// The agent's summary so far and the files it tracks, which the next
/// compaction builds on. Saved with the session.
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
    /// Summaries asked for, at most [`MAX_COMPACTIONS`].
    pub summaries: u32,
}

/// The summary call of a compaction, a [`ModelRequest`] of the turn.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The first message the compaction keeps.
    pub upto: usize,
    /// The state the summary goes into.
    pub state: SummaryState,
}

fn compact(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Compact {
        entity: args.agent,
        focus: args.args,
    });
}

/// What compacting an agent reads: its messages, what requests already
/// leave out, its summary so far, and its model.
struct Compacting<'a> {
    messages: &'a [Message],
    condensed: Option<&'a Condensed>,
    summarized: Option<&'a Summarized>,
    spec: &'a ModelSpec,
}

/// The first message requests send as it is, after what `condensed`
/// summarized.
fn first_live(condensed: Option<&Condensed>) -> usize {
    condensed.map_or(0, |condensed| condensed.upto)
}

/// Clears the older tool outputs of the messages requests send as they are.
fn clear_old_outputs(
    policy: &CompactionPolicy,
    conversation: &mut Conversation,
    condensed: Option<&Condensed>,
) -> Cleared {
    let live = conversation.messages_mut().get_mut(first_live(condensed)..);
    policy.clearing.clear(live.unwrap_or_default())
}

impl Compacting<'_> {
    /// Where a compaction for `reason` should end, if anything new would be
    /// summarized.
    fn cut(&self, policy: &CompactionPolicy, reason: &CompactReason) -> Option<usize> {
        let from = first_live(self.condensed);
        policy.cut(self.messages, from, reason, Some(self.spec))
    }

    /// Starts the summary call of a compaction for `reason` ending at
    /// `upto`, on `turn`.
    fn start(
        &self,
        commands: &mut Commands,
        turn: Entity,
        policy: &CompactionPolicy,
        (reason, upto): (CompactReason, usize),
    ) -> Result<(), UnsupportedOption> {
        let (state, request) = policy.plan(
            self.summarized
                .map(|summarized| &summarized.0)
                .unwrap_or(&SummaryState::default()),
            self.messages,
            first_live(self.condensed),
            upto,
            self.spec,
            &reason,
        )?;
        commands.spawn((
            Name::new("compacting"),
            Summarizing {
                reason,
                upto,
                state,
            },
            ModelRequest { request },
            CallOf(turn),
        ));
        Ok(())
    }
}

/// Compacts an idle agent's conversation on the user's request, in a turn
/// of its own that ends with the summary.
fn on_compact(
    compact: On<Compact>,
    agents: Query<
        (
            &Conversation,
            Option<&Condensed>,
            Option<&Summarized>,
            Option<&Connection>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = compact.entity;
    let Ok((conversation, condensed, summarized, connection, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::info(agent, "A turn is running; stop it first."));
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::info(
            agent,
            "No model is connected; pick one first.",
        ));
        return;
    };
    let compacting = Compacting {
        messages: conversation.messages(),
        condensed,
        summarized,
        spec: &connection.spec,
    };
    let reason = CompactReason::Asked {
        focus: compact.focus.clone(),
    };
    let Some(upto) = compacting.cut(&policy.0, &reason) else {
        notices.write(Notice::info(agent, "Nothing to compact yet."));
        return;
    };
    let turn = commands
        .spawn((Name::new("compaction"), TurnOf(agent)))
        .id();
    if let Err(why) = compacting.start(&mut commands, turn, &policy.0, (reason, upto)) {
        notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
        commands.entity(turn).despawn();
    }
}

/// Before a request that leaves less than the reserve of the model's window
/// free, clears old tool outputs, which costs no model call, and asks for a
/// summary only when clearing was not enough. The context in use is the
/// last reply's or the request's estimate, whichever is larger.
fn compact_near_the_window(
    mut prepare: On<PrepareRequest>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Condensed>,
        Option<&Summarized>,
        Option<&Connection>,
        &mut LastUsage,
    )>,
    mut turns: Query<&mut Compactions>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (turn, agent) = (prepare.entity, prepare.agent);
    let Ok(mut compactions) = turns.get_mut(turn) else {
        return;
    };
    let Ok((mut conversation, condensed, summarized, connection, mut last)) = agents.get_mut(agent)
    else {
        return;
    };
    let Some(connection) = connection.filter(|_| compactions.summaries < MAX_COMPACTIONS) else {
        return;
    };
    let (policy, spec) = (&policy.0, &*connection.spec);
    let used = last
        .context()
        .unwrap_or(0)
        .max(policy.estimate(&prepare.messages));
    if !policy.over_threshold(used, spec) {
        return;
    }
    let cleared = clear_old_outputs(policy, &mut conversation, condensed);
    let left = used.saturating_sub(cleared.tokens as u64);
    if cleared.results > 0 {
        policy.clearing.clear(&mut prepare.event_mut().messages);
        *last = LastUsage(Some(Usage::new().total_tokens(left)));
        notices.write(Notice::info(
            agent,
            format!(
                "The conversation nears the model's context window: cleared {} older tool \
                 outputs (about {} tokens).",
                cleared.results,
                usage::tokens(cleared.tokens as u64)
            ),
        ));
    }
    if !policy.over_threshold(left, spec) {
        return;
    }
    let compacting = Compacting {
        messages: conversation.messages(),
        condensed,
        summarized,
        spec,
    };
    let Some(upto) = compacting.cut(policy, &CompactReason::Threshold) else {
        return;
    };
    compactions.summaries += 1;
    let started = compacting.start(
        &mut commands,
        turn,
        policy,
        (CompactReason::Threshold, upto),
    );
    if let Err(why) = started {
        notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
    }
}

/// After the model refused a request as too long, clears old tool outputs
/// and sends it again; when that was done already or cleared nothing,
/// summarizes the older messages first. Otherwise the turn fails.
fn recover_from_overflow(
    mut failed: On<ModelFailed>,
    mut agents: Query<(
        &mut Conversation,
        Option<&Condensed>,
        Option<&Summarized>,
        Option<&Connection>,
    )>,
    mut turns: Query<&mut Compactions>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if failed.handled || !failed.report.is_context_overflow() {
        return;
    }
    let (turn, agent) = (failed.entity, failed.agent);
    let Ok(mut compactions) = turns.get_mut(turn) else {
        return;
    };
    let Ok((mut conversation, condensed, summarized, connection)) = agents.get_mut(agent) else {
        return;
    };
    let policy = &policy.0;
    if !compactions.cleared {
        compactions.cleared = true;
        let cleared = clear_old_outputs(policy, &mut conversation, condensed);
        if cleared.results > 0 {
            notices.write(Notice::info(
                agent,
                format!(
                    "The conversation outgrew the model's context window: cleared {} older \
                     tool outputs (about {} tokens) and sending it again.",
                    cleared.results,
                    usage::tokens(cleared.tokens as u64)
                ),
            ));
            commands.trigger(CallModel { entity: turn });
            failed.event_mut().handled = true;
            return;
        }
    }
    let Some(connection) = connection.filter(|_| compactions.summaries < MAX_COMPACTIONS) else {
        return;
    };
    let compacting = Compacting {
        messages: conversation.messages(),
        condensed,
        summarized,
        spec: &connection.spec,
    };
    let Some(upto) = compacting.cut(policy, &CompactReason::Overflow) else {
        return;
    };
    compactions.summaries += 1;
    match compacting.start(&mut commands, turn, policy, (CompactReason::Overflow, upto)) {
        Ok(()) => {
            notices.write(Notice::info(
                agent,
                "The conversation outgrew the model's context window: summarizing its older \
                 messages and sending it again.",
            ));
            failed.event_mut().handled = true;
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
        }
    }
}

/// Takes a finished summary: the agent's [`Condensed`] now sends it in
/// place of the summarized messages, which its log records. A failed
/// summary replaces nothing. Either way the turn carries on with its model
/// call, or ends when the user asked for the compaction.
fn take_summary(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Summarizing, &Done<ModelReply>)>,
    turns: Query<&TurnOf>,
    mut agents: Query<(&Conversation, Option<&Condensed>, &mut LastUsage)>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), summarizing, Done(reply))) = calls.get(call) else {
        return;
    };
    commands.entity(call).despawn();
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((conversation, condensed, mut last)) = agents.get_mut(agent) else {
        return;
    };
    let summary = reply
        .as_ref()
        .map_err(ToString::to_string)
        .and_then(|response| Summarizer::summary_text(response).map_err(|why| why.to_string()));
    match summary {
        Ok(summary) => {
            let messages = conversation.messages();
            let upto = summarizing.upto;
            let older = messages.get(first_live(condensed).min(upto)..upto);
            let kept = messages.get(upto..).unwrap_or_default();
            let state = SummaryState {
                summary,
                ..summarizing.state.clone()
            };
            let condensed = Condensed {
                upto,
                summary: state.message().unwrap_or_default(),
            };
            let left = policy.0.estimate(&condensed.request(messages));
            *last = LastUsage(Some(Usage::new().total_tokens(left)));
            commands
                .entity(agent)
                .insert((condensed, Summarized(state)));
            let older = older.unwrap_or_default();
            notices.write(Notice::info(
                agent,
                format!(
                    "Compacted {} messages (about {} tokens) into a summary and kept {} \
                     (about {} tokens) as it was; the model now gets about {} tokens of \
                     conversation.",
                    older.len(),
                    usage::tokens(policy.0.estimate(older)),
                    match kept.len() {
                        1 => "the newest message".to_owned(),
                        kept => format!("the newest {kept} messages"),
                    },
                    usage::tokens(policy.0.estimate(kept)),
                    usage::tokens(left)
                ),
            ));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
        }
    }
    match summarizing.reason {
        CompactReason::Asked { .. } => {
            commands.entity(turn).despawn();
        }
        CompactReason::Threshold | CompactReason::Overflow => {
            commands.trigger(CallModel { entity: turn });
        }
    }
}

#[cfg(test)]
mod tests;
