//! The one way an agent is made. A [`Spawner`], or [`spawn`] with the world,
//! makes it from an [`Origin`] and an [`AgentSpec`]: what the spec leaves out
//! is the origin's. [`spawn_root`] cannot fail.
//!
//! ```
//! use rig_ecs::prelude::*;
//! use rig_ecs::spawn::{AgentSpec, Origin, Spawner};
//!
//! /// Each agent whose turn ended gets a reviewer below it.
//! fn add_reviewer(
//!     settled: On<Add<Settled>>,
//!     turns: Query<&ForAgent, With<TurnOutcome>>,
//!     mut spawner: Spawner,
//! ) {
//!     if let Ok(&ForAgent(agent)) = turns.get(settled.entity) {
//!         let spec = AgentSpec { name: "reviewer".into(), ..AgentSpec::default() };
//!         spawner.spawn(Origin::ChildOf(agent), spec).ok();
//!     }
//! }
//! # let _ = add_reviewer;
//! ```

use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryData;
use bevy_ecs::system::{EntityCommands, SystemParam, SystemState};
use rig_core::effect::EffectId;

use super::agent::{Agent, AgentId, EffectParent, NotYours, SpawnedBy, SystemPrompt, ToolAccess};
use super::model::{Connection, ModelChoice, Models, ReasoningSetting};
use super::tools::ToolDef;

/// Where a new agent comes from: whose settings it takes, and where it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// An agent nothing spawned, with the defaults.
    Root,
    /// A child of the agent: below it ([`SpawnedBy`]).
    ChildOf(Entity),
    /// Beside the agent, under its parent if any, such as its successor.
    Like(Entity),
}

/// What a new agent is given; each setting left out is its origin's, the
/// connection and reasoning setting with the model.
#[derive(Clone, Debug, Default)]
pub struct AgentSpec {
    /// Its name, shown in views; `agent` when empty.
    pub name: String,
    /// Its id, a fresh one when `None`.
    pub id: Option<AgentId>,
    /// A catalog model that calls tools, as `vendor/model`.
    pub model: Option<String>,
    /// An effort, such as `low`, which becomes its reasoning setting.
    pub effort: Option<String>,
    /// Its tools, by name: each a registered tool its origin keeps.
    pub tools: Option<Vec<String>>,
    /// Its own part of its system prompt.
    pub system_prompt: Option<String>,
    /// The effect its model calls are recorded under ([`EffectParent`]).
    pub under: Option<EffectId>,
}

/// Why no agent was made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnRefused {
    /// The origin is not an agent.
    NotAnAgent,
    /// An agent has the id asked for.
    IdTaken(AgentId),
    /// A tool asked for is not one its origin keeps.
    Tool(NotYours),
    /// The model or effort asked for cannot be used, for the reason given.
    Model(String),
}

impl std::fmt::Display for SpawnRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnAgent => f.write_str("the agent to spawn from is gone"),
            Self::IdTaken(id) => write!(f, "an agent with the id `{}` exists", id.0),
            Self::Tool(refused) => refused.fmt(f),
            Self::Model(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for SpawnRefused {}

/// What a new agent takes from its origin.
#[derive(QueryData)]
struct Settings {
    model: Option<&'static ModelChoice>,
    effort: &'static ReasoningSetting,
    prompt: &'static SystemPrompt,
    access: &'static ToolAccess,
    connection: Option<&'static Connection>,
    under: Option<&'static EffectParent>,
}

/// Makes agents from a system, by the rule of the [module](self).
#[derive(SystemParam)]
pub struct Spawner<'w, 's> {
    origins: Query<'w, 's, Settings, With<Agent>>,
    lineage: Query<'w, 's, &'static SpawnedBy>,
    ids: Query<'w, 's, &'static AgentId>,
    tools: Query<'w, 's, &'static ToolDef>,
    models: Option<Res<'w, Models>>,
    commands: Commands<'w, 's>,
}

impl Spawner<'_, '_> {
    /// Spawns an agent from `origin` with `spec`, and returns it to insert
    /// more; or why it may not be made.
    pub fn spawn(
        &mut self,
        origin: Origin,
        spec: AgentSpec,
    ) -> Result<EntityCommands<'_>, SpawnRefused> {
        let (from, parent) = match origin {
            Origin::Root => (None, None),
            Origin::ChildOf(agent) => (Some(agent), Some(agent)),
            Origin::Like(agent) => (
                Some(agent),
                self.lineage.get(agent).ok().map(SpawnedBy::parent),
            ),
        };
        let from = from.map(|agent| self.origins.get(agent)).transpose();
        let from = from.map_err(|_| SpawnRefused::NotAnAgent)?;
        let id = spec.id.unwrap_or_default();
        if self.ids.iter().any(|taken| *taken == id) {
            return Err(SpawnRefused::IdTaken(id));
        }
        let model = from.as_ref().and_then(|from| from.model);
        let effort = from.as_ref().map(|from| *from.effort).unwrap_or_default();
        let fallback = Models::default();
        let models = self.models.as_deref().unwrap_or(&fallback);
        let (choice, effort) = ModelChoice::inherit(
            &models.0,
            (model, effort),
            spec.model.as_deref(),
            spec.effort.as_deref(),
        )
        .map_err(SpawnRefused::Model)?;
        let access = from
            .as_ref()
            .map(|from| from.access.clone())
            .unwrap_or_default();
        let access = match &spec.tools {
            None => access,
            Some(asked) => {
                let registered = self.tools.iter().map(|def| def.0.name.as_str());
                access
                    .grant(asked, registered)
                    .map_err(SpawnRefused::Tool)?
            }
        };
        let prompt = spec.system_prompt.map(SystemPrompt);
        let prompt = prompt.or_else(|| from.as_ref().map(|from| from.prompt.clone()));
        let same = choice.is_some() && choice.as_ref() == model;
        let connection = from
            .as_ref()
            .and_then(|from| from.connection)
            .filter(|_| same)
            .cloned();
        let under = spec
            .under
            .or(from.and_then(|from| from.under).map(|under| under.0));
        let name = match spec.name.is_empty() {
            true => "agent".to_owned(),
            false => spec.name,
        };
        // Each before `Agent`, so what observes a new agent sees it: the
        // connection before the model choice, so connecting keeps it, and
        // the parent after the id, so the agent's log opens naming it.
        let mut agent = self.commands.spawn(id);
        if let Some(connection) = connection {
            agent.insert(connection);
        }
        if let Some(choice) = choice {
            agent.insert(choice);
        }
        if let Some(parent) = parent {
            agent.insert(SpawnedBy::new(parent));
        }
        if let Some(under) = under {
            agent.insert(EffectParent(under));
        }
        let prompt = prompt.unwrap_or_default();
        agent.insert((Agent::new(), Name::new(name), prompt, access, effort));
        Ok(agent)
    }
}

/// Spawns an agent from `origin` with `spec`, as a [`Spawner`] does, and
/// returns it; or why it may not be made.
pub fn spawn(world: &mut World, origin: Origin, spec: AgentSpec) -> Result<Entity, SpawnRefused> {
    let mut state = SystemState::<Spawner>::new(world);
    let mut spawner = state.get_mut(world).map_err(|_| SpawnRefused::NotAnAgent)?;
    let spawned = spawner.spawn(origin, spec).map(|agent| agent.id());
    state.apply(world);
    spawned
}

/// Spawns an agent nothing spawned, with the defaults, named `agent`. It
/// keeps every tool, those registered later too.
pub fn spawn_root(world: &mut World) -> EntityWorldMut<'_> {
    world.spawn((Name::new("agent"), Agent::new(), ToolAccess::all()))
}

#[cfg(test)]
mod tests;
