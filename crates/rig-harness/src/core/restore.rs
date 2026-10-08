//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done. Each log is
//! read from its newest compaction on, with a torn last line cut off and
//! records of unknown types skipped. Each fix the reconcile pass makes is an
//! ordinary record, so the next restore finds nothing left to fix; the
//! request-build repair that answers any call still without a result stays
//! as the backstop.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{ReflectFromReflect, TypeRegistry};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{CallId, ToolCall, ToolResult, ToolResultContent, UserContent};
use serde::Deserialize;
use serde::de::{DeserializeSeed, IgnoredAny};
use serde_json::Value;

use super::agent::{
    Agent, AgentId, CallOf, Conversation, ModelChoice, Notice, Spawned, SpawnedBy, SystemPrompt,
    ToolAccess, ToolCallRun, TurnOf,
};
use super::compaction::Compacted;
use super::inbox::{Deliver, DeliveryMode, Origin, OriginKind, RequestId};
use super::journal::{
    AgentLog, COMPONENT_VERSION, Header, Line, ParentRef, Record, ReflectSaved, SavedValue,
    SessionLog, SessionPaths, Settings, UsageRecord, load_blobs,
};
use super::subagents::{self, Assignment, Delegated, TASK};
use super::tools::failed;
use super::turn::{CallModel, ToolStarter, tool_name};

/// The result of a call that may change something and was running when
/// the session stopped.
const INTERRUPTED: &str = "interrupted by a restart; it may have partly run";

/// The first fields of a line, read from every line to find the newest
/// compaction and the subagent answers delivered.
#[derive(Deserialize)]
struct Envelope {
    seq: u64,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    origin: Option<Origin>,
}

/// An agent log folded into the agent's state.
struct Folded {
    path: PathBuf,
    header: Header,
    conversation: Conversation,
    message_seqs: Vec<u64>,
    halted: bool,
    settings: Option<Settings>,
    usage: UsageRecord,
    components: BTreeMap<String, SavedValue>,
    compacted: Compacted,
    next_seq: u64,
    /// The agents whose output a message of this log delivered.
    delivered: BTreeSet<String>,
}

/// A restored agent, for the reconcile pass.
struct RestoredAgent {
    entity: Entity,
    id: AgentId,
    parent: Option<ParentRef>,
    task: Option<String>,
    depth: usize,
    halted: bool,
    delivered: BTreeSet<String>,
}

/// The agents [`restore_session`] spawned, until [`reconcile`] took them.
#[derive(Resource)]
pub(crate) struct Restored(Vec<RestoredAgent>);

/// Spawns the agents of the session's logs, with their conversations,
/// settings, usage, compactions and saved plugin components, links each
/// subagent to the agent that started it, and starts logging. Anything
/// that does not load is skipped with a notice.
pub(crate) fn restore_session(world: &mut World) {
    let (Some(log), Some(paths)) = (
        world.get_resource::<SessionLog>().cloned(),
        world.get_resource::<SessionPaths>().cloned(),
    ) else {
        return;
    };
    let blobs = paths.blobs();
    let mut notices: Vec<String> = Vec::new();
    let mut folded: Vec<Folded> = Vec::new();
    for path in paths.agent_logs() {
        match read_log(&path, &blobs) {
            Ok(agent) => folded.push(agent),
            Err(failure) => notices.push(format!(
                "Could not restore the agent log {}: {failure}.",
                path.display()
            )),
        }
    }
    let parents: HashMap<String, Option<String>> = folded
        .iter()
        .map(|agent| {
            let parent = agent
                .header
                .parent
                .as_ref()
                .map(|parent| parent.agent.clone());
            (agent.header.agent.clone(), parent)
        })
        .collect();
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let mut restored = Vec::new();
    let mut logs = Vec::new();
    for agent in folded {
        let Folded {
            path,
            header,
            conversation,
            message_seqs,
            halted,
            settings,
            usage,
            components,
            compacted,
            next_seq,
            delivered,
        } = agent;
        let id = AgentId(header.agent.clone());
        let applied = settings.clone().unwrap_or_default();
        let name = match &header.task {
            Some(task) => format!("subagent: {task}"),
            None => "agent".to_owned(),
        };
        let mut spawned = world.spawn((
            Name::new(name),
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
        if let Some(task) = &header.task {
            spawned.insert(Delegated { task: task.clone() });
        }
        if let Some(model) = applied.model {
            spawned.insert(ModelChoice(model));
        }
        let entity = spawned.id();
        if let Some(registry) = &registry {
            let registry = registry.read();
            for (path, saved) in &components {
                let outcome = if saved.v > COMPONENT_VERSION {
                    Err(format!("it was saved by a newer build (version {})", saved.v).into())
                } else {
                    restore_component(world, entity, &registry, path, saved.value.clone())
                };
                if let Err(failure) = outcome {
                    notices.push(format!("Skipped saved component `{path}`: {failure}."));
                }
            }
        }
        let depth = depth(&header.agent, &parents);
        restored.push(RestoredAgent {
            entity,
            id: id.clone(),
            parent: header.parent.clone(),
            task: header.task.clone(),
            depth,
            halted,
            delivered,
        });
        logs.push((
            id.0,
            AgentLog {
                path,
                depth,
                next_seq,
                started: true,
                file: None,
                pending: Vec::new(),
                message_seqs,
                halted,
                settings,
                usage,
                components,
            },
        ));
    }
    let entities: HashMap<String, Entity> = restored
        .iter()
        .map(|agent| (agent.id.0.clone(), agent.entity))
        .collect();
    for agent in &restored {
        if let Some(parent) = agent
            .parent
            .as_ref()
            .and_then(|parent| entities.get(&parent.agent))
            && let Ok(mut child) = world.get_entity_mut(agent.entity)
        {
            child.insert(SpawnedBy(*parent));
        }
    }
    // What restoring set off, such as connecting each model, is not logged.
    world.flush();
    log.resume(logs);
    world.insert_resource(Restored(restored));
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

/// Reads and folds the agent log at `path`, cutting off a torn last line
/// on disk first. Images come back from `blobs`.
fn read_log(path: &Path, blobs: &Path) -> Result<Folded, Box<dyn Error>> {
    let bytes = fs::read(path)?;
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
        OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(u64::try_from(keep)?)?;
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
        path: path.to_owned(),
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
        delivered: envelopes
            .iter()
            .flatten()
            .filter_map(|envelope| envelope.origin.as_ref())
            .filter(|origin| origin.kind == OriginKind::Agent)
            .filter_map(|origin| origin.from.as_ref())
            .map(|from| from.0.clone())
            .collect(),
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
            folded.compacted = Compacted {
                upto: 0,
                summary: record.summary,
                read: record.read,
                modified: record.modified,
            };
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
                load_blobs(&mut message, blobs);
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

fn restore_component(
    world: &mut World,
    entity: Entity,
    registry: &TypeRegistry,
    path: &str,
    value: Value,
) -> Result<(), Box<dyn Error>> {
    let registration = registry
        .get_with_type_path(path)
        .ok_or("its plugin is not loaded")?;
    let component = registration
        .data::<ReflectComponent>()
        .filter(|_| registration.data::<ReflectSaved>().is_some())
        .ok_or("it is no longer a saved component")?;
    let value = TypedReflectDeserializer::new(registration, registry).deserialize(value)?;
    let value = match registration.data::<ReflectFromReflect>() {
        Some(from_reflect) => from_reflect
            .from_reflect(value.as_ref())
            .ok_or("its saved value no longer fits the type")?
            .into_partial_reflect(),
        None => value,
    };
    let mut entity = world.get_entity_mut(entity)?;
    component.insert(&mut entity, value.as_ref(), registry);
    Ok(())
}

/// Settles what the restored agents left half done, by appending records:
/// a tool call without a result runs again when its tool is an ordinary
/// one that only reads, is answered as started when it is a `task` call
/// whose subagent's log names it, and is answered as interrupted
/// otherwise, an open tool's call included; an agent whose conversation ends
/// in the user's message, or in a full set of tool results, calls its
/// model again; and a subagent's answer that no message of the agent that
/// started it delivered goes to that agent now, or once the subagent is
/// done.
pub(crate) fn reconcile(
    restored: Option<Res<Restored>>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    children: Query<&Spawned>,
    starter: ToolStarter,
    log: Res<SessionLog>,
    mut commands: Commands,
) {
    let Some(restored) = restored else {
        return;
    };
    commands.remove_resource::<Restored>();
    let restored = &restored.0;
    // Agents whose turn starts now.
    let mut working: HashSet<Entity> = HashSet::new();
    for agent in restored {
        let Ok((id, mut conversation)) = agents.get_mut(agent.entity) else {
            continue;
        };
        let mut results: Vec<ToolResult> = Vec::new();
        let mut reruns: Vec<ToolCall> = Vec::new();
        for call in dangling(conversation.messages()) {
            let name = call.function.name.as_str();
            if name == TASK {
                let child = restored.iter().find(|child| {
                    child
                        .parent
                        .as_ref()
                        .is_some_and(|parent| parent.agent == id.0 && parent.call == call.id)
                });
                results.push(match child {
                    Some(child) => call.result(vec![ToolResultContent::text(subagents::started(
                        &child.id,
                        child.task.as_deref().unwrap_or_default(),
                    ))]),
                    None => failed(&call, INTERRUPTED.to_owned()),
                });
            } else if starter.reruns(name) {
                reruns.push(call);
            } else {
                results.push(failed(&call, INTERRUPTED.to_owned()));
            }
        }
        if !results.is_empty() {
            log.commit(id, &mut conversation, Message::tool_results(results), None);
        }
        if !reruns.is_empty() {
            let turn = commands
                .spawn((Name::new("turn"), TurnOf(agent.entity)))
                .id();
            let runs: Vec<(Entity, ToolCallRun)> = reruns
                .into_iter()
                .map(|call| {
                    let run = ToolCallRun {
                        touch: starter
                            .footprint(call.function.name.as_str())
                            .of(&call.function.arguments),
                        call,
                        parent: None,
                    };
                    let entity = commands
                        .spawn((tool_name(&run), CallOf(turn), run.clone()))
                        .id();
                    (entity, run)
                })
                .collect();
            for (entity, run) in runs {
                starter.start(&mut commands, entity, agent.entity, &run);
            }
            working.insert(agent.entity);
        } else if !agent.halted
            && matches!(conversation.messages().last(), Some(Message::User { .. }))
        {
            let turn = commands
                .spawn((Name::new("turn"), TurnOf(agent.entity)))
                .id();
            commands.trigger(CallModel { entity: turn });
            working.insert(agent.entity);
        }
    }
    // Deepest first, so a subagent knows whether its own subagents still
    // work. The answers go after every turn above started, so a parent
    // that works queues them.
    let mut order: Vec<&RestoredAgent> = restored.iter().collect();
    order.sort_by_key(|agent| std::cmp::Reverse(agent.depth));
    let mut assigned: HashSet<Entity> = HashSet::new();
    for agent in order {
        let Some(parent) = agent.parent.as_ref().and_then(|parent| {
            restored
                .iter()
                .find(|restored| restored.id.0 == parent.agent)
        }) else {
            continue;
        };
        if parent.delivered.contains(&agent.id.0) {
            continue;
        }
        let waits = working.contains(&agent.entity)
            || children
                .get(agent.entity)
                .is_ok_and(|children| children.iter().any(|child| assigned.contains(&child)));
        let Some(call) = agent.parent.as_ref().map(|parent| parent.call.to_string()) else {
            continue;
        };
        let request = RequestId(call);
        if waits {
            commands.entity(agent.entity).insert(Assignment {
                effect: None,
                request,
            });
            assigned.insert(agent.entity);
        } else if let Ok((id, conversation)) = agents.get(agent.entity) {
            commands.trigger(Deliver {
                entity: parent.entity,
                text: subagents::answer(
                    agent.task.as_deref().unwrap_or_default(),
                    subagents::final_answer(conversation.messages()),
                ),
                origin: Origin::agent(id.clone(), Some(request)),
                mode: DeliveryMode::Queue,
            });
        }
    }
}

/// The tool calls of the conversation's last reply that have no result.
fn dangling(messages: &[Message]) -> Vec<ToolCall> {
    let Some(at) = messages
        .iter()
        .rposition(|message| matches!(message, Message::Assistant(_)))
    else {
        return Vec::new();
    };
    let Some(Message::Assistant(reply)) = messages.get(at) else {
        return Vec::new();
    };
    let answered: HashSet<&CallId> = messages
        .iter()
        .skip(at + 1)
        .flat_map(|message| match message {
            Message::User { content } => content.as_slice(),
            _ => &[],
        })
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(&result.call),
            _ => None,
        })
        .collect();
    reply
        .content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) if !answered.contains(&call.id) => Some(call.clone()),
            _ => None,
        })
        .collect()
}
