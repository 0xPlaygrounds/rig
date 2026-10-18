//! What each agent's model is given: the system prompt and the tools, as
//! one `context` event at an agent's first model call and again whenever
//! either changes, so the journal tells what the model saw without a copy
//! per turn. The hash of the last one sent is saved with the agent, so a
//! restarted worker sends none for a context the journal holds.

use std::hash::{DefaultHasher, Hash, Hasher};

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use rig_core::message::Message;
use rig_ecs::agent::AgentId;
use rig_ecs::session_log::ReflectSaved;
use rig_ecs::stage::{ForAgent, Started};
use rig_ecs::turn::{ModelRequest, TurnCall};
use rig_protocol::methods::WorkerEvent;

use super::WorkerOutbox;

/// The kind of the event.
pub const CONTEXT: &str = "context";

/// On an agent: the hash of the last context the channel was told of.
#[derive(Component, Reflect, Default, Clone, Debug, PartialEq, Eq)]
#[reflect(Component, Saved)]
pub struct ContextSent(pub String);

/// Sends the context of each model call that starts, when it is not the
/// one its agent sent last.
pub(crate) fn export_context(
    started: On<Add<Started>>,
    calls: Query<(&ModelRequest, &ForAgent), With<TurnCall>>,
    agents: Query<(&AgentId, Option<&ContextSent>)>,
    outbox: Option<ResMut<WorkerOutbox>>,
    mut commands: Commands,
) {
    let (Ok((call, &ForAgent(agent))), Some(mut outbox)) = (calls.get(started.entity), outbox)
    else {
        return;
    };
    let Ok((id, sent)) = agents.get(agent) else {
        return;
    };
    let request = &call.request;
    let system: Vec<&str> = request
        .chat_history
        .iter()
        .filter_map(|message| match &**message {
            Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    let system = system.join("\n\n");
    let tools: Vec<serde_json::Value> = request
        .tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name.to_string(),
                "description": tool.description,
                "schema": tool.parameters,
            })
        })
        .collect();
    let mut hasher = DefaultHasher::new();
    system.hash(&mut hasher);
    serde_json::to_string(&tools)
        .unwrap_or_default()
        .hash(&mut hasher);
    let hash = format!("{:016x}", hasher.finish());
    if sent.is_some_and(|sent| sent.0 == hash) {
        return;
    }
    outbox.events.push(WorkerEvent {
        kind: CONTEXT.into(),
        data: serde_json::json!({
            "agentId": id.0,
            "system": system,
            "tools": tools,
            "hash": hash,
        }),
        source: None,
    });
    commands.entity(agent).insert(ContextSent(hash));
}
