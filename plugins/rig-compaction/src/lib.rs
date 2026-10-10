//! Compaction: when a turn's next request nears the model's context window,
//! older tool outputs are cleared, and when that is not enough the older
//! messages are summarized by the model into rig-memory's checkpoint, which
//! requests send in their place as the agent's [`Condensed`]. The same
//! happens after the model refused a request as too long, and `/compact
//! <focus>` does it now. It uses the kernel's turn hooks only:
//! [`PrepareRequest`] for the window, [`ModelFailed`] for a refused request,
//! and a [`ModelRequest`] for the summary, a call of the turn, so
//! interrupting the turn cancels it.

use bevy_ecs::query::QueryData;
use rig_core::completion::{Usage, tokens_label};
use rig_ecs::prelude::*;
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
            |In(args): In<CommandArgs>, mut commands: Commands| {
                let (entity, focus) = (args.agent, args.args);
                commands.trigger(Compact { entity, focus });
            },
        )
        .add_observer(on_compact)
        .add_observer(compact_near_the_window)
        .add_observer(recover_from_overflow)
        .add_observer(take_summary);
    }
}

/// How agents are compacted: rig-memory's default policy, tracking the
/// files the built-in file tools read and changed. A plugin whose tool
/// reads or changes files tracks its argument the same way, by adding a
/// [`TrackArgument`] to `tracked` (from `finish` or a startup system).
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

/// An agent as compacting it reads it: its messages, what requests already
/// leave out, its summary so far, and its model.
#[derive(QueryData)]
#[query_data(mutable)]
struct Compacting {
    conversation: &'static mut Conversation,
    condensed: Option<&'static Condensed>,
    summarized: Option<&'static Summarized>,
    connection: Option<&'static Connection>,
}

/// The first message requests send as it is, after what `condensed`
/// summarized.
fn first_live(condensed: Option<&Condensed>) -> usize {
    condensed.map_or(0, |condensed| condensed.upto)
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
    /// Clears the older tool outputs of the messages requests send as they
    /// are.
    fn clear_old_outputs(&mut self, policy: &CompactionPolicy) -> Cleared {
        let from = first_live(self.condensed);
        let live = self.conversation.messages_mut().get_mut(from..);
        policy.clearing.clear(live.unwrap_or_default())
    }

    /// Starts the summary call of a compaction of `agent` for `reason` on
    /// `turn`, or on a turn of its own when `None`, when it has a model and
    /// anything new would be summarized.
    fn start(
        &self,
        (agent, turn): (Entity, Option<Entity>),
        (policy, reason): (&CompactionPolicy, CompactReason),
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) -> Start {
        let Some(connection) = self.connection else {
            return Start::NoModel;
        };
        let (messages, spec) = (self.conversation.messages(), &*connection.spec);
        let from = first_live(self.condensed);
        let Some(upto) = policy.cut(messages, from, &reason, Some(spec)) else {
            return Start::NothingNew;
        };
        let none = SummaryState::default();
        let planned = policy.plan(
            self.summarized.map_or(&none, |summarized| &summarized.0),
            messages,
            from,
            upto,
            spec,
            &reason,
        );
        let (state, request) = match planned {
            Ok(planned) => planned,
            Err(why) => {
                notices.write(Notice::error(agent, format!("Cannot compact: {why}.")));
                return Start::Refused;
            }
        };
        let turn = turn.unwrap_or_else(|| {
            let turn = commands.spawn((Name::new("compaction"), TurnOf(agent)));
            turn.id()
        });
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

/// Before a request that leaves less than the reserve of the model's window
/// free, clears old tool outputs, which costs no model call, and asks for a
/// summary only when clearing was not enough. The context in use is the
/// last reply's or the request's estimate, whichever is larger.
fn compact_near_the_window(
    mut prepare: On<PrepareRequest>,
    mut agents: Query<(Compacting, &mut LastUsage)>,
    mut turns: Query<&mut Compactions>,
    policy: Res<Compaction>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (turn, agent) = (prepare.entity, prepare.agent);
    let Ok(mut compactions) = turns.get_mut(turn) else {
        return;
    };
    let Ok((mut compacting, mut last)) = agents.get_mut(agent) else {
        return;
    };
    let connection = compacting.connection;
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
    let cleared = compacting.clear_old_outputs(policy);
    let left = used.saturating_sub(cleared.tokens as u64);
    if cleared.results > 0 {
        policy.clearing.clear(&mut prepare.event_mut().messages);
        *last = LastUsage(Some(Usage::new().total_tokens(left)));
        let nears = "The conversation nears the model's context window";
        notices.write(cleared_notice(agent, nears, &cleared, ""));
    }
    if !policy.over_threshold(left, spec) {
        return;
    }
    let reason = (policy, CompactReason::Threshold);
    let started = compacting.start((agent, Some(turn)), reason, &mut commands, &mut notices);
    if matches!(started, Start::Refused | Start::Started) {
        compactions.summaries += 1;
    }
}

/// After the model refused a request as too long, clears old tool outputs
/// and sends it again; when that was done already or cleared nothing,
/// summarizes the older messages first. Otherwise the turn fails.
fn recover_from_overflow(
    mut failed: On<ModelFailed>,
    mut agents: Query<Compacting>,
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
    let Ok(mut compacting) = agents.get_mut(agent) else {
        return;
    };
    let policy = &policy.0;
    let outgrew = "The conversation outgrew the model's context window";
    if !compactions.cleared {
        compactions.cleared = true;
        let cleared = compacting.clear_old_outputs(policy);
        if cleared.results > 0 {
            let then = " and sending it again";
            notices.write(cleared_notice(agent, outgrew, &cleared, then));
            commands.trigger(CallModel { entity: turn });
            failed.event_mut().handled = true;
            return;
        }
    }
    if compactions.summaries >= MAX_COMPACTIONS {
        return;
    }
    let reason = (policy, CompactReason::Overflow);
    match compacting.start((agent, Some(turn)), reason, &mut commands, &mut notices) {
        Start::NoModel | Start::NothingNew => {}
        Start::Refused => compactions.summaries += 1,
        Start::Started => {
            compactions.summaries += 1;
            notices.write(Notice::info(
                agent,
                format!("{outgrew}: summarizing its older messages and sending it again."),
            ));
            failed.event_mut().handled = true;
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
                    tokens_label(policy.0.estimate(older)),
                    match kept.len() {
                        1 => "the newest message".to_owned(),
                        kept => format!("the newest {kept} messages"),
                    },
                    tokens_label(policy.0.estimate(kept)),
                    tokens_label(left)
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
