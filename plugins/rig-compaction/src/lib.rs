//! Compaction: when a turn's next request nears the model's context window,
//! older tool outputs are cleared, and when that is not enough the older
//! messages are summarized by the model into rig-memory's checkpoint, which
//! takes their place in the conversation. Both are splices of the
//! conversation ([`Commit::splice`]), so they are logged, restored and
//! announced as [`Committed::Splice`]; the summary's content item has the
//! origin [`summary_origin`]. The same happens after the model refused a
//! request as too long, and `/compact <focus>` does it now. It is stage systems over the turn's model calls
//! ([`TurnCall`]): a Prepare system measures each request against the
//! window, and a Review system takes over a call whose [`RetryVerdict`]
//! says it overflowed. The summary is a [`ModelRequest`] of the turn, so
//! interrupting the turn cancels it, that holds the call it compacts for
//! until the summary is in; a new call built from the compacted
//! conversation then supersedes it. Both systems are in public
//! [`CompactionSystems`] sets: a plugin that routes a call to another
//! model or edits its request runs before [`CompactionSystems::Prepare`],
//! so the request is measured as it will be sent.

use std::slice;
use std::sync::Arc;

use bevy_ecs::query::QueryData;
use rig_basics::ProjectContextSystems;
use rig_coding_tools::AttachSystems;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, Message, Usage, tokens_label};
use rig_ecs::prelude::*;
use rig_ecs::prompt::tool_rules;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
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
        .configure_sets(
            AgentPass,
            (
                // It measures the request as it goes out: after the
                // files a message names were attached and the default
                // prompt went in.
                CompactionSystems::Prepare
                    .in_set(Stage::Prepare)
                    .after(AttachSystems)
                    .after(ProjectContextSystems),
                CompactionSystems::Recover.in_set(Stage::Review),
            ),
        )
        .add_systems(
            AgentPass,
            (
                compact_near_the_window.in_set(CompactionSystems::Prepare),
                recover_from_overflow.in_set(CompactionSystems::Recover),
            ),
        )
        .add_observer(take_summary);
    }
}

/// The compaction plugin's stage systems, to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactionSystems {
    /// In [`Stage::Prepare`]: measures each turn's request against the
    /// window of the model it goes to, and clears old tool outputs from it
    /// or holds it for a summary, after [`AttachSystems`] and
    /// [`ProjectContextSystems`]. A system that changes the request or its
    /// model runs before it.
    Prepare,
    /// In [`Stage::Review`]: takes over a turn's call the model refused as
    /// too long.
    Recover,
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

/// The summary call of a compaction, a [`ModelRequest`] of the turn, and
/// a hold ([`HoldOf`]) on the model call it compacts for, if any.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// Where the summarized messages end.
    pub until: Until,
    /// The state the summary goes into.
    pub state: SummaryState,
}

/// Where a compaction's summarized messages end, by message id, as
/// messages may move while the summary is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Until {
    /// Every message.
    All,
    /// The messages before this one, which the compaction keeps. A summary
    /// that arrives after it was taken out replaces nothing.
    Before(MessageId),
}

/// The origin of a compaction's summary in the conversation.
pub fn summary_origin() -> Origin {
    Origin::plugin("compaction")
}

/// An agent as compacting it reads it: its messages, its summary so far,
/// and its model.
#[derive(QueryData)]
#[query_data(mutable)]
struct Compacting {
    conversation: &'static mut Conversation,
    summarized: Option<&'static Summarized>,
    connection: Option<&'static Connection>,
}

/// The messages of `conversation` without the summary it holds, which the
/// next summary builds on from [`Summarized`] instead of reading it again,
/// and how many of the first messages held nothing else.
fn unsummarized(conversation: &Conversation) -> (Vec<Message>, usize) {
    let summary = summary_origin();
    let mut from = 0;
    let mut messages = conversation.messages().to_vec();
    for (at, message) in messages.iter_mut().enumerate() {
        let Message::User { content } = message else {
            continue;
        };
        let mut item = 0;
        content.retain(|_| {
            item += 1;
            conversation.origin(at, item - 1) != Some(&summary)
        });
        if content.is_empty() && from == at {
            from += 1;
        }
    }
    (messages, from)
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
    /// Clears the older tool outputs of `agent`'s conversation, by a splice
    /// of the messages that changed, which keeps their ids.
    fn clear_old_outputs(
        &mut self,
        agent: Entity,
        policy: &CompactionPolicy,
        commit: &mut Commit,
    ) -> Cleared {
        let mut messages = self.conversation.messages().to_vec();
        let cleared = policy.clearing.clear(&mut messages);
        let was = self.conversation.messages();
        let changed =
            |(at, message): (usize, &Message)| (was.get(at) != Some(message)).then_some(at);
        let changed: Vec<usize> = messages.iter().enumerate().filter_map(changed).collect();
        let (Some(&first), Some(&last)) = (changed.first(), changed.last()) else {
            return cleared;
        };
        let entries = (first..=last).filter_map(|at| {
            let mut entry = self.conversation.entry(at)?;
            entry.message = messages.get(at)?.clone();
            Some(entry)
        });
        let entries: Vec<Entry> = entries.collect();
        commit.splice(agent, &mut self.conversation, first..=last, entries);
        cleared
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
        let spec = &*connection.spec;
        let (messages, from) = unsummarized(&self.conversation);
        let Some(upto) = policy.cut(&messages, from, &reason, Some(spec)) else {
            return Start::NothingNew;
        };
        let until = match self.conversation.id(upto) {
            Some(kept) => Until::Before(kept),
            None => Until::All,
        };
        let none = SummaryState::default();
        let planned = policy.plan(
            self.summarized.map_or(&none, |summarized| &summarized.0),
            &messages,
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
        let summary = (
            Name::new("compacting"),
            Summarizing {
                reason,
                until,
                state,
            },
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

/// The model `model` goes to, as Start will send it: its own connection,
/// else its agent's `own`, as its adapters change it.
fn routed(model: &ModelRequest, own: Option<&Connection>) -> Option<Arc<ModelSpec>> {
    let connection = model.connection.as_ref().or(own)?.clone();
    let adapters = model.adapters.iter();
    Some(
        adapters
            .fold(connection, |connection, adapter| adapter.apply(connection))
            .spec,
    )
}

/// The policy's estimate of `request` as a model of `spec` gets it: one
/// that calls no tools gets them taken out at Start, and with them their
/// rules from the system prompt.
fn sent_tokens(
    policy: &CompactionPolicy,
    request: &CompletionRequest,
    spec: &ModelSpec,
    tools: &Query<(&ToolDef, &ToolRules)>,
) -> u64 {
    let messages = &request.chat_history;
    let estimate = policy.estimate(messages);
    if spec.tools {
        return estimate;
    }
    let carried = request.tools.iter().filter_map(|tool| {
        let rules = tools.iter().find(|(def, _)| def.0.name == tool.name);
        rules.map(|(_, rules)| rules)
    });
    let rules = tool_rules(carried);
    match messages.first() {
        Some(system @ Message::System { content }) if !rules.is_empty() => {
            let without = Message::system(content.replacen(&rules, "", 1));
            estimate.saturating_sub(policy.estimate(slice::from_ref(system)))
                + policy.estimate(&[without])
        }
        _ => estimate,
    }
}

/// Before a turn's request that leaves less than the reserve of the
/// window of the model it goes to free, clears old tool outputs, which
/// costs no model call, and asks for a summary only when clearing was not
/// enough. The context in use is the last reply's or the request's
/// estimate, as that model gets it, whichever is larger.
fn compact_near_the_window(
    mut calls: Query<
        (Entity, &mut ModelRequest, &CallOf, &ForAgent),
        (Added<Admitted>, With<TurnCall>),
    >,
    mut agents: Query<(Compacting, &mut LastUsage)>,
    mut turns: Query<&mut Compactions>,
    tools: Query<(&ToolDef, &ToolRules)>,
    policy: Res<Compaction>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, mut model, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok(mut compactions) = turns.get_mut(turn) else {
            continue;
        };
        let Ok((mut compacting, mut last)) = agents.get_mut(agent) else {
            continue;
        };
        if compactions.summaries >= MAX_COMPACTIONS {
            continue;
        }
        let Some(spec) = routed(&model, compacting.connection) else {
            continue;
        };
        let (policy, spec) = (&policy.0, &*spec);
        let sent = sent_tokens(policy, &model.request, spec, &tools);
        let used = last.context().unwrap_or(0).max(sent);
        let messages = &mut model.request.chat_history;
        if !policy.over_threshold(used, spec) {
            continue;
        }
        let cleared = compacting.clear_old_outputs(agent, policy, &mut commit);
        let left = used.saturating_sub(cleared.tokens as u64);
        if cleared.results > 0 {
            policy.clearing.clear(messages);
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
    mut commit: Commit,
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
            let cleared = compacting.clear_old_outputs(agent, policy, &mut commit);
            if cleared.results > 0 {
                let then = " and sending it again";
                notices.write(cleared_notice(agent, outgrew, &cleared, then));
                commands.spawn((
                    TurnCall,
                    Supersedes(call),
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::child_of(effect.parent),
                ));
                continue;
            }
        }
        if compactions.summaries >= MAX_COMPACTIONS {
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

/// Takes a settled summary: it replaces the summarized messages in the
/// conversation, a splice its log records, merged into the first kept
/// message when that is the user's, so user and model keep taking turns.
/// A failed summary, or one whose kept message is gone, replaces nothing.
/// Either way the call it held is sent again, built anew, or the turn ends
/// when the user asked for the compaction.
fn take_summary(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ForAgent, &Summarizing, &Outcome, Option<&HoldOf>)>,
    held: Query<&Effect>,
    mut agents: Query<(&mut Conversation, &mut LastUsage)>,
    policy: Res<Compaction>,
    mut commit: Commit,
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
    let end = match summarizing.until {
        Until::All => Some(conversation.messages().len()),
        Until::Before(kept) => conversation.index_of(kept),
    };
    let summary = match (&outcome.0, outcome.completion(), end) {
        (Err(report), ..) => Err(report.to_string()),
        (Ok(_), None, _) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(_), None) => Err("the messages it kept were taken out".to_owned()),
        (Ok(_), Some(response), Some(end)) => Summarizer::summary_text(response)
            .map(|summary| (summary, end))
            .map_err(|why| why.to_string()),
    };
    match summary {
        Ok((summary, end)) => {
            let messages = conversation.messages();
            let older = messages.get(..end).unwrap_or_default();
            let (older, older_tokens) = (older.len(), policy.0.estimate(older));
            let kept = messages.get(end..).unwrap_or_default();
            let (kept, kept_tokens) = (kept.len(), policy.0.estimate(kept));
            let state = SummaryState {
                summary,
                ..summarizing.state.clone()
            };
            let text = Message::user(state.message().unwrap_or_default());
            let summary = Entry::from(text, summary_origin());
            commit.splice(agent, &mut conversation, ..end, [summary]);
            let left = policy.0.estimate(conversation.messages());
            *last = LastUsage(Some(Usage::new().total_tokens(left)));
            commands.entity(agent).insert(Summarized(state));
            notices.write(Notice::info(
                agent,
                format!(
                    "Compacted {older} messages (about {} tokens) into a summary and kept {} \
                     (about {} tokens) as it was; the model now gets about {} tokens of \
                     conversation.",
                    tokens_label(older_tokens),
                    match kept {
                        1 => "the newest message".to_owned(),
                        kept => format!("the newest {kept} messages"),
                    },
                    tokens_label(kept_tokens),
                    tokens_label(left)
                ),
            ));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Compaction failed: {why}.")));
        }
    }
    match hold {
        // The call it held is built again, from the compacted conversation.
        Some(&HoldOf(call)) => {
            let parent = held.get(call).ok().and_then(|effect| effect.parent);
            commands.spawn((
                TurnCall,
                Supersedes(call),
                CallOf(turn),
                ForAgent(agent),
                Effect::child_of(parent),
            ));
        }
        // The user's own compaction turn ends with it.
        None => {
            let compacted = TurnOutcome::Stopped(Some("compacted".to_owned()));
            commands.entity(turn).try_insert(compacted);
        }
    }
}

#[cfg(test)]
mod tests;
