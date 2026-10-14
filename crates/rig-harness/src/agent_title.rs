//! An agent's title, saved where a library program reads it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_ecs::session_log::ReflectSaved;

/// An agent's title, such as a subagent's task, which names it in views.
/// Saved with the session as a string under rig-memory's `TITLE` (its type
/// path, `rig::title`), so a library program reads it with
/// `SavedAgent::title`.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
#[type_path = "rig"]
#[type_name = "title"]
pub struct AgentTitle(pub String);
