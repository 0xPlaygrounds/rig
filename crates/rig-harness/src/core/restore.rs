//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done. Each log is
//! read from its newest compaction on, with a torn last line cut off and
//! records of unknown types skipped. Each fix the reconcile pass makes is an
//! ordinary record, so the next restore finds nothing left to fix; the
//! request-build repair that answers any call still without a result stays
//! as the backstop.
//!
//! Plugins re-arm their own obligations from their saved components when
//! [`Restored`] is triggered on each agent, and may keep an agent from
//! carrying its work on:
//!
//! ```ignore
//! fn rearm(mut restored: On<Restored>, waits: Query<&MyWait>) {
//!     if waits.contains(restored.entity) {
//!         restored.event_mut().resume = false;
//!     }
//! }
//! ```

use std::collections::{BTreeMap, HashMap};
use std::error::Error;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_core::completion::Message;
use rig_core::message::{ToolCall, ToolResult};
use rig_core::transcript::pending_calls;
use rig_memory::SummaryState;
use serde::Deserialize;
use serde::de::IgnoredAny;

use super::agent::{
    Agent, AgentId, CallOf, Conversation, ModelChoice, Notice, SpawnedBy, SystemPrompt, ToolAccess,
    ToolCallRun, TurnOf,
};
use super::compaction::Compacted;
use super::journal::{
    AgentLog, COMPONENT_VERSION, Header, Line, Record, SavedComponents, SavedValue, SessionLog,
    Settings, UsageRecord, load_blobs,
};
use super::store::{JournalStore, SessionStore};
use super::tools::failed;
use super::turn::{CallModel, ToolStarter, tool_name};

/// The result of a call that may change something and was running when
/// the session stopped.
const INTERRUPTED: &str = "interrupted by a restart; it may have partly run";

/// The first fields of a line, read from every line to find the newest
/// compaction.
#[derive(Deserialize)]
struct Envelope {
    seq: u64,
    #[serde(rename = "type")]
    kind: String,
}

/// An agent log folded into the agent's state.
struct Folded {
    header: Header,
    conversation: Conversation,
    message_seqs: Vec<u64>,
    halted: bool,
    settings: Option<Settings>,
    usage: UsageRecord,
    components: BTreeMap<String, SavedValue>,
    compacted: Compacted,
    next_seq: u64,
}

/// A restored agent, for the reconcile pass.
struct RestoredAgent {
    entity: Entity,
    parent: Option<String>,
    depth: usize,
    halted: bool,
}

/// The agents [`restore_session`] spawned, until [`reconcile`] took them.
#[derive(Resource)]
pub(crate) struct RestoredAgents(Vec<RestoredAgent>);

/// An agent was restored: its conversation with its origins, its settings,
/// its saved components and the agent that spawned it are back, and the
/// agents above it were reconciled. Triggered once per agent at startup,
/// before the core carries it on. Plugins re-arm what they owe from their
/// own saved components here.
#[derive(EntityEvent, Clone, Debug)]
pub struct Restored {
    /// The agent.
    pub entity: Entity,
    /// Whether the core carries the agent on: runs again its calls that
    /// can run again, and sends a conversation that ends in a message the
    /// model has not answered to the model. An observer sets it to `false`
    /// to keep the agent idle; every call left without a result is then
    /// answered as interrupted, and nothing runs again by itself.
    pub resume: bool,
}

/// Spawns the agents of the session's logs, with their conversations,
/// settings, usage, compactions and saved plugin components, links each
/// agent to the agent that spawned it, and starts logging. Anything
/// that does not load is skipped with a notice.
pub(crate) fn restore_session(world: &mut World) {
    let (Some(log), Some(SessionStore(store))) = (
        world.get_resource::<SessionLog>().cloned(),
        world.get_resource::<SessionStore>().cloned(),
    ) else {
        return;
    };
    let mut notices: Vec<String> = Vec::new();
    let mut folded: Vec<Folded> = Vec::new();
    let agents = store.agents().unwrap_or_else(|failure| {
        notices.push(format!("Could not list the agent logs: {failure}."));
        Vec::new()
    });
    for agent in agents {
        match read_log(&*store, &agent) {
            Ok(agent) => folded.push(agent),
            Err(failure) => notices.push(format!(
                "Could not restore the log of agent {agent}: {failure}."
            )),
        }
    }
    let parents: HashMap<String, Option<String>> = folded
        .iter()
        .map(|agent| (agent.header.agent.clone(), agent.header.parent.clone()))
        .collect();
    let saved_components = world
        .get_resource::<SavedComponents>()
        .map(|saved| saved.0.clone())
        .unwrap_or_default();
    let mut restored = Vec::new();
    let mut logs = Vec::new();
    for agent in folded {
        let Folded {
            header,
            conversation,
            message_seqs,
            halted,
            settings,
            usage,
            components,
            compacted,
            next_seq,
        } = agent;
        let id = AgentId(header.agent.clone());
        let applied = settings.clone().unwrap_or_default();
        let mut spawned = world.spawn((
            Name::new("agent"),
            Agent,
            id.clone(),
            conversation,
            compacted,
            usage.total(),
            applied.effort,
            applied
                .prompt
                .map_or_else(SystemPrompt::default, SystemPrompt),
            applied.tools.map_or(ToolAccess::All, ToolAccess::Only),
        ));
        if let Some(model) = applied.model {
            spawned.insert(ModelChoice(model));
        }
        let entity = spawned.id();
        for (path, saved) in &components {
            let outcome = match saved_components.get(path.as_str()) {
                _ if saved.v > COMPONENT_VERSION => Err(format!(
                    "it was saved by a newer build (version {})",
                    saved.v
                )),
                None => Err("no plugin saves it any more".to_owned()),
                Some(insert) => insert(&mut world.entity_mut(entity), saved.value.clone())
                    .map_err(|failure| format!("its saved value no longer fits: {failure}")),
            };
            if let Err(failure) = outcome {
                notices.push(format!("Skipped saved component `{path}`: {failure}."));
            }
        }
        let depth = depth(&header.agent, &parents);
        restored.push(RestoredAgent {
            entity,
            parent: header.parent.clone(),
            depth,
            halted,
        });
        logs.push((
            id.0,
            AgentLog {
                depth,
                next_seq,
                started: true,
                pending: Vec::new(),
                message_seqs,
                halted,
                settings,
                usage,
                components,
            },
        ));
    }
    let entities: HashMap<String, Entity> = logs
        .iter()
        .zip(&restored)
        .map(|((id, _), agent)| (id.clone(), agent.entity))
        .collect();
    for agent in &restored {
        if let Some(parent) = agent
            .parent
            .as_ref()
            .and_then(|parent| entities.get(parent))
            && let Ok(mut child) = world.get_entity_mut(agent.entity)
        {
            child.insert(SpawnedBy(*parent));
        }
    }
    // What restoring set off, such as connecting each model, is not logged.
    world.flush();
    log.resume(logs);
    world.insert_resource(RestoredAgents(restored));
    for notice in notices {
        world.write_message(Notice::error(None, notice));
    }
}

/// How many agents above `agent` started it.
fn depth(agent: &str, parents: &HashMap<String, Option<String>>) -> usize {
    let mut depth = 0;
    let mut at = agent;
    while let Some(Some(parent)) = parents.get(at) {
        depth += 1;
        // A loop cannot happen, but a bound costs nothing.
        if depth > parents.len() {
            break;
        }
        at = parent;
    }
    depth
}

/// Reads and folds the log of `agent` in `store`, cutting off a torn last
/// line there first. Images come back from the store's blobs.
fn read_log(store: &dyn JournalStore, agent: &str) -> Result<Folded, Box<dyn Error>> {
    let bytes = store.read(agent)?;
    let complete = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at + 1);
    let mut lines: Vec<&[u8]> = bytes
        .get(..complete)
        .unwrap_or_default()
        .split(|byte| *byte == b'\n')
        .collect();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    let mut keep = complete;
    if let Some(last) = lines.last()
        && serde_json::from_slice::<IgnoredAny>(last).is_err()
    {
        keep = keep.saturating_sub(last.len() + 1);
        lines.pop();
    }
    if keep < bytes.len() {
        store.truncate(agent, u64::try_from(keep)?)?;
    }
    lines.retain(|line| !line.is_empty());
    let header = match lines
        .first()
        .and_then(|first| serde_json::from_slice::<Line>(first).ok())
    {
        Some(Line {
            record: Record::Header(header),
            ..
        }) => header,
        _ => return Err("it does not start with a header".into()),
    };
    let envelopes: Vec<Option<Envelope>> = lines
        .iter()
        .map(|line| serde_json::from_slice(line).ok())
        .collect();
    let mut folded = Folded {
        header,
        conversation: Conversation::default(),
        message_seqs: Vec::new(),
        halted: false,
        settings: None,
        usage: UsageRecord::default(),
        components: BTreeMap::new(),
        compacted: Compacted::default(),
        next_seq: envelopes
            .iter()
            .flatten()
            .map(|envelope| envelope.seq + 1)
            .max()
            .unwrap_or(1),
    };
    // Messages are read from the first one the newest compaction kept, the
    // rest from the compaction on.
    let compaction = envelopes
        .iter()
        .rposition(|envelope| matches!(envelope, Some(envelope) if envelope.kind == "compaction"))
        .and_then(|at| {
            let line = serde_json::from_slice::<Line>(lines.get(at)?).ok()?;
            match line.record {
                Record::Compaction(record) => Some((at, record)),
                _ => None,
            }
        });
    let (from_messages, from_rest) = match compaction {
        Some((at, record)) => {
            let first_kept = envelopes
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, envelope)| {
                    matches!(envelope, Some(envelope) if envelope.seq >= record.first_kept)
                })
                .map_or(at, |(index, _)| index.min(at));
            folded.settings = record.snapshot.settings;
            folded.usage = record.snapshot.usage;
            folded.components = record.snapshot.components;
            folded.compacted = Compacted(SummaryState {
                upto: 0,
                summary: record.summary,
                tracked: record.tracked,
            });
            (first_kept, at + 1)
        }
        None => (1, 1),
    };
    for (index, line) in lines.iter().enumerate().skip(from_messages) {
        // Unknown record types are skipped and left on disk.
        let Ok(line) = serde_json::from_slice::<Line>(line) else {
            continue;
        };
        let superseded = index < from_rest;
        match line.record {
            Record::Message {
                mut message,
                origin,
            } => {
                load_blobs(&mut message, store);
                if !folded.conversation.append(message, origin) {
                    folded.message_seqs.push(line.seq);
                }
                folded.halted = false;
            }
            Record::Retract => {
                if folded.conversation.retract().is_some() {
                    folded.message_seqs.pop();
                }
                folded.halted = false;
            }
            Record::Halt => folded.halted = true,
            _ if superseded => {}
            Record::Settings(settings) => folded.settings = Some(settings),
            Record::Usage(usage) => folded.usage = usage,
            Record::Component {
                component,
                v,
                value,
            } => match value {
                Some(value) => {
                    folded.components.insert(component, SavedValue { v, value });
                }
                None => {
                    folded.components.remove(&component);
                }
            },
            Record::Header(_) | Record::Compaction(_) => {}
        }
    }
    Ok(folded)
}

/// Settles what the restored agents left half done, parents before the
/// agents they spawned. Each agent first gets [`Restored`]; then, by
/// appending records, a tool call without a result starts again when its
/// tool is an ordinary read-only one, and is answered as interrupted
/// otherwise; and an agent whose conversation ends
/// in the user's message, or in a full set of tool results, calls its
/// model again. An agent an observer kept idle only gets the interrupted
/// results.
pub(crate) fn reconcile(world: &mut World) {
    let Some(RestoredAgents(mut agents)) = world.remove_resource::<RestoredAgents>() else {
        return;
    };
    agents.sort_by_key(|agent| agent.depth);
    for agent in agents {
        let mut restored = Restored {
            entity: agent.entity,
            resume: true,
        };
        world.trigger_ref(&mut restored);
        let carry = (agent.entity, restored.resume, !agent.halted);
        if let Err(error) = world.run_system_cached_with(settle, carry) {
            warn!("could not reconcile a restored agent: {error}");
        }
    }
}

/// Settles one restored agent, as [`reconcile`] says: `resume` is what its
/// [`Restored`] observers left, and `answer` whether a conversation ending
/// in a message the model has not answered goes to the model.
fn settle(
    In((agent, resume, answer)): In<(Entity, bool, bool)>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    starter: ToolStarter,
    log: Res<SessionLog>,
    mut commands: Commands,
) {
    let Ok((id, mut conversation)) = agents.get_mut(agent) else {
        return;
    };
    let mut results: Vec<ToolResult> = Vec::new();
    let mut reruns: Vec<ToolCall> = Vec::new();
    for call in pending_calls(conversation.messages()) {
        if resume && starter.reruns(call.function.name.as_str()) {
            reruns.push(call);
        } else {
            results.push(failed(&call, INTERRUPTED.to_owned()));
        }
    }
    if !results.is_empty() {
        log.commit(id, &mut conversation, Message::tool_results(results), None);
    }
    if !resume {
        return;
    }
    if !reruns.is_empty() {
        let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
        let runs: Vec<(Entity, ToolCallRun)> = reruns
            .into_iter()
            .map(|call| {
                let run = starter.run(call, None);
                let entity = commands
                    .spawn((tool_name(&run), CallOf(turn), run.clone()))
                    .id();
                (entity, run)
            })
            .collect();
        for (entity, run) in runs {
            starter.start(&mut commands, entity, agent, &run);
        }
    } else if answer && matches!(conversation.messages().last(), Some(Message::User { .. })) {
        let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
        commands.trigger(CallModel { entity: turn });
    }
}
