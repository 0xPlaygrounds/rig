//! Who may answer an ask, and how an ask a parent may not answer escalates
//! up the tree.

use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_protocol::{AgentId, Answerers, AskId};

use super::tree::ParentSession;
use crate::channel::Caller;

/// An approval or question waiting for an answer.
#[derive(Component)]
pub struct OpenAsk {
    pub id: AskId,
    pub session: Entity,
    pub agent: AgentId,
    pub deadline: Instant,
    pub answerers: Answerers,
}

/// Bubbles `ask` up the tree to whoever may answer it.
#[derive(EntityEvent)]
#[entity_event(propagate = &'static ParentSession, auto_propagate)]
pub struct Escalate {
    #[event_target]
    pub session: Entity,
    pub ask: Entity,
}

/// Whether `caller` may answer `ask`.
pub fn may_answer(world: &World, ask: &OpenAsk, caller: &Caller) -> bool {
    let _ = (world, ask, caller);
    false
}

/// Routes a newly opened `ask`: answers it by rule, or escalates it.
pub fn on_ask(world: &mut World, ask: Entity) {
    let _ = (world, ask);
}
