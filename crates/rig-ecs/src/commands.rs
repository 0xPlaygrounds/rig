//! Owned construction values for the native ECS graph.
//!
//! Use [`Agent::spawn`] and [`Prompt::spawn`] with a `World`, or import
//! [`RigCommands`] in a system. Commands reserve an entity immediately and
//! initialize it when Bevy applies deferred commands. Expected failures are
//! retained in [`CommandFailures`]; drain them in the host's error reporting
//! system. Failed initialization removes its newly constructed graph; unrelated
//! observer effects are not rolled back. Child relationships are established
//! before payload insertion: relationship observers can see an attached child
//! before its Grant or Utterance payload exists.
//!
//! The host still owns the loop and drives `run_to_quiescence`.

use bevy_ecs::prelude::*;
use rig_core::{completion::message::ToolChoice, effect::EffectFamily};

use crate::{
    agent::{
        AdditionalParams, DefaultMaxTurns, InvalidCalls, MaxTokens, MaxTurns, MessageParts,
        OrderCounter, Output, Owner, Preamble, Run, RunCounter, Temperature, ToolChoiceSpec, Turn,
        UsesModel,
    },
    bus::{Bus, Policy, ServingPolicy},
};

/// A construction or lifecycle operation that could not be applied.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OperationError {
    /// Install the bus and agent systems before constructing work.
    #[error("the agent runtime is not installed; call rig_ecs::commands::install first")]
    NotInstalled,
    /// Installation is explicit and occurs once per world.
    #[error("a bus or agent runtime is already installed in this world")]
    AlreadyInstalled,
    /// The entity was removed before the operation applied.
    #[error("entity {0:?} no longer exists")]
    MissingEntity(Entity),
    /// The target is not an agent graph.
    #[error("entity {0:?} is not an agent with Owner and UsesModel components")]
    NotAgent(Entity),
    /// A lifecycle operation requires a run graph.
    #[error("entity {0:?} is not a run")]
    NotRun(Entity),
    /// The target is not a turn below a live run.
    #[error("entity {0:?} is not a turn below a run")]
    NotTurn(Entity),
    /// The operation is outside its supported scheduling window.
    #[error("cannot {operation} entity {entity:?} in its current phase")]
    InvalidPhase {
        /// The target whose phase does not permit the operation.
        entity: Entity,
        /// The operation that was refused.
        operation: &'static str,
    },
    /// A different decision already occupies the same unconsumed slot.
    #[error("entity {0:?} already has a different retry decision")]
    ConflictingRetry(Entity),
    /// Forking would duplicate unfinished work or external memory ownership.
    #[error(
        "run {0:?} cannot fork at this boundary; finish the current turn and use a non-remembering run"
    )]
    UnsafeFork(Entity),
    /// No additional unique run sequence or graph order can be allocated.
    #[error("the world's run sequence or graph order is exhausted")]
    SequenceExhausted,
    /// The referenced handler is missing or serves another family.
    #[error("entity {entity:?} is not registered for {expected:?} effects")]
    HandlerFamily {
        /// The referenced handler entity.
        entity: Entity,
        /// The family the operation requires.
        expected: EffectFamily,
    },
}

/// Install the bus and agent systems without scheduling or starting a host loop.
///
/// Like `Bus::install`, this initializes the shared task pool if the host has
/// not already done so. Calling twice returns an error without modifying state.
pub fn install(world: &mut World, policy: ServingPolicy) -> Result<(), OperationError> {
    if world.contains_resource::<Policy>() || world.contains_resource::<RunCounter>() {
        return Err(OperationError::AlreadyInstalled);
    }
    Bus::with_policy(policy).install(world);
    crate::systems::install_agent(world);
    world.init_resource::<CommandFailures>();
    Ok(())
}

pub(crate) fn installed(world: &World) -> Result<(), OperationError> {
    if world.contains_resource::<Policy>()
        && world.contains_resource::<RunCounter>()
        && world.contains_resource::<OrderCounter>()
        && world.contains_resource::<crate::systems::AgentRuntimeInstalled>()
    {
        Ok(())
    } else {
        Err(OperationError::NotInstalled)
    }
}

pub(crate) fn handler(
    world: &World,
    entity: Entity,
    expected: EffectFamily,
) -> Result<(), OperationError> {
    if crate::bus::handlers::registered_family(world, entity) == Some(expected) {
        Ok(())
    } else {
        Err(OperationError::HandlerFamily { entity, expected })
    }
}

pub(crate) fn entity(
    world: &mut World,
    entity: Entity,
) -> Result<EntityWorldMut<'_>, OperationError> {
    world
        .get_entity_mut(entity)
        .map_err(|_| OperationError::MissingEntity(entity))
}

// Insertion observers flush before spawn/insert returns. A removed parent must
// not leave an unattached child behind or allow further graph construction.
pub(crate) fn child(
    world: &mut World,
    parent: Entity,
    bundle: impl Bundle,
) -> Result<Entity, OperationError> {
    entity(world, parent)?;
    // Establish the relationship before payload observers can remove its parent.
    // Bevy's relationship hooks queue parent bookkeeping during insertion.
    let child = world.spawn(ChildOf(parent)).id();
    child_present(world, parent, child)?;
    entity(world, child)?.insert(bundle);
    child_present(world, parent, child)?;
    Ok(child)
}

fn child_present(world: &mut World, parent: Entity, child: Entity) -> Result<(), OperationError> {
    if world.get_entity(parent).is_err() {
        if world.get_entity(child).is_ok() {
            world.despawn(child);
        }
        return Err(OperationError::MissingEntity(parent));
    }
    entity(world, child)?;
    Ok(())
}

pub(crate) fn finish_construction(
    world: &mut World,
    target: Entity,
    result: Result<(), OperationError>,
) -> Result<(), OperationError> {
    if result.is_err()
        && let Ok(entity) = world.get_entity_mut(target)
    {
        entity.despawn();
    }
    result
}

pub(crate) fn order(world: &mut World) -> Result<crate::agent::Order, OperationError> {
    let mut counter = world.resource_mut::<OrderCounter>();
    let order = counter.0;
    counter.0 = order
        .checked_add(1)
        .ok_or(OperationError::SequenceExhausted)?;
    Ok(crate::agent::Order(order))
}

/// Temporary agent configuration; spawning consumes it into ordinary components.
#[derive(Debug, Clone)]
pub struct Agent {
    model: Entity,
    owner: String,
    preamble: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
    tool_choice: Option<ToolChoice>,
    output: Output,
    max_turns: usize,
    tools: Vec<Entity>,
    memory: Option<(Entity, String)>,
}

impl Agent {
    /// An agent using `model`, with one model turn and no tools or preamble.
    pub fn new(model: Entity) -> Self {
        Self {
            model,
            owner: "agent".to_owned(),
            preamble: None,
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            output: Output::default(),
            max_turns: 1,
            tools: Vec::new(),
            memory: None,
        }
    }

    /// Name used in the run's recording scope.
    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = owner.into();
        self
    }

    /// Set a system preamble, including an explicitly empty one.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Limit response tokens.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set provider-specific request parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set the tool-selection policy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set structured-output mode and schema.
    pub fn output(mut self, output: Output) -> Self {
        self.output = output;
        self
    }

    /// Set the model-call budget inherited by runs.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Append grants in the supplied order.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Entity>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Load and append conversation history through a registered memory handler.
    pub fn memory(mut self, handler: Entity, conversation: impl Into<String>) -> Self {
        self.memory = Some((handler, conversation.into()));
        self
    }

    fn validate(&self, world: &World) -> Result<(), OperationError> {
        installed(world)?;
        handler(world, self.model, EffectFamily::Completion)?;
        for tool in &self.tools {
            handler(world, *tool, EffectFamily::Tool)?;
        }
        if let Some((memory, _)) = &self.memory {
            handler(world, *memory, EffectFamily::Memory)?;
        }
        Ok(())
    }

    /// Validate before creating an agent. Pending handler bindings may apply
    /// first, including their application observers; those effects are not rolled back.
    pub fn spawn(self, world: &mut World) -> Result<Entity, OperationError> {
        self.validate(world)?;
        self.materialize(world)?;
        let entity = world.spawn_empty().id();
        self.insert(world, entity)?;
        Ok(entity)
    }

    fn materialize(&self, world: &mut World) -> Result<(), OperationError> {
        for entity in std::iter::once(self.model)
            .chain(self.tools.iter().copied())
            .chain(self.memory.as_ref().map(|(entity, _)| *entity))
        {
            crate::bus::handlers::materialize_registration(world, entity)
                .map_err(|_| OperationError::MissingEntity(entity))?;
        }
        self.validate(world)
    }

    fn insert(self, world: &mut World, target: Entity) -> Result<(), OperationError> {
        let result = self.initialize(world, target);
        finish_construction(world, target, result)
    }

    fn initialize(self, world: &mut World, target: Entity) -> Result<(), OperationError> {
        entity(world, target)?.insert((
            Owner(self.owner),
            Preamble(self.preamble),
            Temperature(self.temperature),
            MaxTokens(self.max_tokens),
            AdditionalParams(self.additional_params),
            ToolChoiceSpec(self.tool_choice),
            self.output,
            DefaultMaxTurns(Some(self.max_turns)),
            MaxTurns(self.max_turns),
            InvalidCalls::default(),
            UsesModel(self.model),
        ));
        entity(world, target)?;
        if let Some((memory, conversation)) = self.memory {
            entity(world, target)?.insert((
                crate::agent::Remembers(memory),
                crate::agent::Conversation(conversation),
            ));
        }
        entity(world, target)?;
        for tool in self.tools {
            let order = order(world)?;
            child(world, target, (crate::agent::Grant(tool), order))?;
        }
        Ok(())
    }
}

/// An owned prompt with named run options. It is consumed into the run graph.
#[derive(Debug, Clone)]
pub struct Prompt {
    agent: Entity,
    text: String,
    history: Vec<MessageParts>,
    streaming: bool,
    max_turns: Option<usize>,
}

impl Prompt {
    /// Submit text with no supplied history and the agent's turn limit.
    pub fn new(agent: Entity, text: impl Into<String>) -> Self {
        Self {
            agent,
            text: text.into(),
            history: Vec::new(),
            streaming: false,
            max_turns: None,
        }
    }

    /// Prepend conversation history. Empty history retains configured memory loading.
    pub fn history(mut self, history: impl IntoIterator<Item = MessageParts>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Request streaming delivery.
    pub fn streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Override the agent's model-call budget for this run.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    fn validate(&self, world: &World) -> Result<(), OperationError> {
        installed(world)?;
        let agent = world
            .get_entity(self.agent)
            .map_err(|_| OperationError::MissingEntity(self.agent))?;
        if agent.contains::<Run>() || agent.contains::<Turn>() || !agent.contains::<Owner>() {
            return Err(OperationError::NotAgent(self.agent));
        }
        let model = agent
            .get::<UsesModel>()
            .ok_or(OperationError::NotAgent(self.agent))?;
        handler(world, model.0, EffectFamily::Completion)
    }

    /// Validate before spawning. Application observers may mutate or remove the
    /// graph during construction; their effects are not rolled back.
    pub fn spawn(self, world: &mut World) -> Result<Entity, OperationError> {
        self.validate(world)?;
        let entity = world.spawn_empty().id();
        self.insert(world, entity)?;
        Ok(entity)
    }

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), OperationError> {
        crate::systems::spawn_run_into(
            world,
            entity,
            self.agent,
            &self.history,
            &self.text,
            self.streaming,
            self.max_turns,
        )
    }
}

/// A failed deferred operation. Failure does not imply the target can be deleted:
/// lifecycle operations refer to existing agents, runs, or turns.
#[derive(Debug, Clone)]
pub struct CommandFailure {
    /// For construction, the reserved destination; for lifecycle operations, the
    /// existing target. The entity may have been removed before application.
    pub entity: Entity,
    /// Why the operation failed.
    pub error: OperationError,
}

/// Deferred errors retained until the host drains them. These are diagnostics,
/// not execution state. Inspect this resource after command application.
#[derive(Resource, Debug, Default)]
pub struct CommandFailures(Vec<CommandFailure>);

impl CommandFailures {
    /// Consume failures since the last drain.
    pub fn drain(&mut self) -> impl Iterator<Item = CommandFailure> + '_ {
        self.0.drain(..)
    }

    /// Inspect failures without consuming them.
    pub fn iter(&self) -> impl Iterator<Item = &CommandFailure> {
        self.0.iter()
    }
}

fn failed(world: &mut World, entity: Entity, error: OperationError) {
    world.init_resource::<CommandFailures>();
    world
        .resource_mut::<CommandFailures>()
        .0
        .push(CommandFailure { entity, error });
}

/// Native Bevy command extensions. Returned IDs are reserved immediately;
/// components and relationships become visible when deferred commands apply.
pub trait RigCommands {
    /// Queue agent construction. See [`Agent::spawn`] for the immediate form.
    fn spawn_agent(&mut self, agent: Agent) -> Entity;
    /// Queue prompt submission. See [`Prompt::spawn`] for the immediate form.
    fn prompt(&mut self, prompt: Prompt) -> Entity;
    /// Queue cancellation; expected errors are retained in CommandFailures.
    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>);
    /// Grant a tool to an agent for future turns; errors go to CommandFailures.
    fn grant_tool(&mut self, agent: Entity, tool: Entity);
    /// Remove this tool's agent grants for future turns.
    fn revoke_tool(&mut self, agent: Entity, tool: Entity);
    /// Request a tool-free turn retry before Materialise consumes it.
    fn retry_turn(&mut self, turn: Entity, retry: crate::agent::Retry);
    /// Merge a request patch while the turn is Fresh, before Assemble.
    fn patch_turn(&mut self, turn: Entity, patch: crate::agent::RequestPatch);
}

impl RigCommands for Commands<'_, '_> {
    fn spawn_agent(&mut self, agent: Agent) -> Entity {
        let entity = self.spawn_empty().id();
        self.queue(move |world: &mut World| {
            let result = if world.get_entity(entity).is_err() {
                Err(OperationError::MissingEntity(entity))
            } else {
                agent.validate(world)
            };
            match result {
                Ok(()) => match agent.materialize(world) {
                    Ok(()) if world.get_entity(entity).is_ok() => {
                        if let Err(error) = agent.insert(world, entity) {
                            failed(world, entity, error);
                        }
                    }
                    Ok(()) => failed(world, entity, OperationError::MissingEntity(entity)),
                    Err(error) => failed(world, entity, error),
                },
                Err(error) => failed(world, entity, error),
            }
        });
        entity
    }

    fn grant_tool(&mut self, agent: Entity, tool: Entity) {
        self.queue(move |world: &mut World| {
            if let Err(error) = crate::lifecycle::grant_tool(world, agent, tool) {
                failed(world, agent, error);
            }
        });
    }

    fn revoke_tool(&mut self, agent: Entity, tool: Entity) {
        self.queue(move |world: &mut World| {
            if let Err(error) = crate::lifecycle::revoke_tool(world, agent, tool) {
                failed(world, agent, error);
            }
        });
    }

    fn retry_turn(&mut self, turn: Entity, retry: crate::agent::Retry) {
        self.queue(move |world: &mut World| {
            if let Err(error) = crate::lifecycle::retry_turn(world, turn, retry) {
                failed(world, turn, error);
            }
        });
    }

    fn patch_turn(&mut self, turn: Entity, patch: crate::agent::RequestPatch) {
        self.queue(move |world: &mut World| {
            if let Err(error) = crate::lifecycle::patch_turn(world, turn, patch) {
                failed(world, turn, error);
            }
        });
    }

    fn cancel_run(&mut self, run: Entity, reason: impl Into<String>) {
        let reason = reason.into();
        self.queue(move |world: &mut World| {
            if let Err(error) = crate::lifecycle::cancel(world, run, reason) {
                failed(world, run, error);
            }
        });
    }

    fn prompt(&mut self, prompt: Prompt) -> Entity {
        let entity = self.spawn_empty().id();
        self.queue(move |world: &mut World| {
            let result = if world.get_entity(entity).is_err() {
                Err(OperationError::MissingEntity(entity))
            } else {
                prompt.validate(world)
            };
            match result {
                Ok(()) => {
                    if let Err(error) = prompt.insert(world, entity) {
                        failed(world, entity, error);
                    }
                }
                Err(error) => failed(world, entity, error),
            }
        });
        entity
    }
}
