//! The header from components: a run's program identity as the log names
//! it, computed from the agent entity — never from a spec struct.

use bevy_ecs::prelude::*;
use rig_core::effect::{EffectFamily, EffectRow, HandlerKey};
use rig_effect_log::{EffectLogRecorder, stable_hash};

use crate::{
    agent::{
        AdditionalParams, Context, DefaultMaxTurns, DocumentId, DocumentProps, DocumentText, Grant,
        MaxTokens, Order, Output, OutputKind, Preamble, Remembers, Retrieves, Route, RunOf,
        Temperature, ToolChoiceSpec, UsesModel,
    },
    bus::{Bound, Scope},
};

/// The JSON the header's `run_spec` hashes for `agent`: the shape
/// `CONTRACT.md` names, built from components, canonicalised by
/// [`stable_hash`]. The builder's identity: the default turn budget, no
/// retries, `fail` — a run's own overrides are not identity.
pub fn spec_json(world: &mut World, agent: Entity) -> serde_json::Value {
    let preamble = world.get::<Preamble>(agent).and_then(|p| p.0.clone());
    let temperature = world.get::<Temperature>(agent).and_then(|t| t.0);
    let max_tokens = world.get::<MaxTokens>(agent).and_then(|m| m.0);
    let additional_params = world
        .get::<AdditionalParams>(agent)
        .and_then(|p| p.0.clone());
    let tool_choice = world
        .get::<ToolChoiceSpec>(agent)
        .and_then(|c| c.0.clone())
        .map(|choice| serde_json::to_value(choice).unwrap_or(serde_json::Value::Null));
    let output = world.get::<Output>(agent).cloned().unwrap_or_default();
    let max_turns = world
        .get::<DefaultMaxTurns>(agent)
        .and_then(|d| d.0)
        .unwrap_or(1);
    let mut context: Vec<(Order, serde_json::Value)> = Vec::new();
    if let Some(children) = world.get::<Children>(agent) {
        let links: Vec<Entity> = children.iter().collect();
        for child in links {
            if let (Some(Context(document)), Some(order)) =
                (world.get::<Context>(child), world.get::<Order>(child))
            {
                let document = *document;
                let order = *order;
                let id = world.get::<DocumentId>(document).map(|d| d.0.clone());
                let text = world.get::<DocumentText>(document).map(|d| d.0.clone());
                let props = world
                    .get::<DocumentProps>(document)
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                if let (Some(id), Some(text)) = (id, text) {
                    let mut json = serde_json::Map::new();
                    json.insert("id".to_owned(), serde_json::Value::String(id));
                    json.insert("text".to_owned(), serde_json::Value::String(text));
                    for (key, value) in props {
                        json.insert(key, serde_json::Value::String(value));
                    }
                    context.push((order, serde_json::Value::Object(json)));
                }
            }
        }
    }
    context.sort_by_key(|(order, _)| *order);
    let output_mode = match output.mode {
        OutputKind::Auto => "Auto",
        OutputKind::Native => "Native",
        OutputKind::Tool => "Tool",
        OutputKind::Prompted => "Prompted",
    };
    serde_json::json!({
        "preamble": preamble,
        "static_context": context.into_iter().map(|(_, document)| document).collect::<Vec<_>>(),
        "additional_params": additional_params,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "max_turns": max_turns,
        "max_invalid_tool_call_retries": 0,
        "output_schema": output.schema,
        "output_mode": output_mode,
        "output_tool_name": serde_json::Value::Null,
        "output_tool_description": serde_json::Value::Null,
        "augment_output_preamble": true,
        "unhandled_invalid_tool_call": "fail",
    })
}

/// The header's `run_spec` for `agent`.
pub fn spec_hash(world: &mut World, agent: Entity) -> Option<u64> {
    stable_hash(&spec_json(world, agent)).ok()
}

/// The agent's required effect row: its model, every route, every tool it
/// grants (retrievable or not), every index it retrieves from, and its
/// memory, by their bound keys.
pub fn required_row(world: &mut World, agent: Entity) -> EffectRow {
    let mut row = EffectRow::new();
    let model = world.get::<UsesModel>(agent).map(|uses| uses.0);
    if let Some(model) = model
        && let Some(bound) = world.get::<Bound>(model)
    {
        row.insert(bound.key.clone(), EffectFamily::Completion);
    }
    let links: Vec<Entity> = world
        .get::<Children>(agent)
        .map(|children| children.iter().collect())
        .unwrap_or_default();
    for link in links {
        let tool = world.get::<Grant>(link).map(|grant| grant.0);
        if let Some(tool) = tool
            && let Some(bound) = world.get::<Bound>(tool)
        {
            row.insert(bound.key.clone(), bound.descriptor.family.family());
        }
        let route = world.get::<Route>(link).map(|route| route.0);
        if let Some(route) = route
            && let Some(bound) = world.get::<Bound>(route)
        {
            row.insert(bound.key.clone(), EffectFamily::Completion);
        }
        let index = world.get::<Retrieves>(link).map(|retrieves| retrieves.0);
        if let Some(index) = index
            && let Some(bound) = world.get::<Bound>(index)
        {
            row.insert(bound.key.clone(), EffectFamily::Retrieve);
        }
    }
    let memory = world.get::<Remembers>(agent).map(|remembers| remembers.0);
    if let Some(memory) = memory
        && let Some(bound) = world.get::<Bound>(memory)
    {
        row.insert(bound.key.clone(), EffectFamily::Memory);
    }
    row
}

/// Stamp `recorder`'s header with `agent`'s identity: the spec hash, the
/// program's hook list as it declares it (`hooks` — the world has no hook
/// stack to name; a program with none passes an empty list), the required
/// row, and the bus policy the world runs under.
pub fn stamp_header(
    world: &mut World,
    agent: Entity,
    recorder: &EffectLogRecorder,
    bus: Option<rig_core::serve::ServingPolicy>,
    hooks: Vec<String>,
) {
    if let Some(hash) = spec_hash(world, agent) {
        recorder.set_run_spec(hash);
    }
    let required = required_row(world, agent);
    recorder.set_program(hooks, required, bus);
}

/// Stamp `run`'s program identity under its scope
/// ([`rig_effect_log::LogHeader::programs`]): the agent's required row and
/// policy hash, keyed by the run's `Scope`. A world running several
/// programs into one log names each this way.
pub fn stamp_run(world: &mut World, run: Entity, recorder: &EffectLogRecorder) {
    let Some(agent) = world.get::<RunOf>(run).map(|run_of| run_of.0) else {
        return;
    };
    let Some(scope) = world.get::<Scope>(run).map(|scope| scope.0.clone()) else {
        return;
    };
    let Some(policy) = spec_hash(world, agent) else {
        return;
    };
    let required = required_row(world, agent);
    recorder.set_program_identity(scope, rig_effect_log::ProgramIdentity { required, policy });
}

/// Whether `agent` is the program `log` was recorded by: refused by name
/// when the log's identity (a `programs` entry with this agent's policy,
/// else the header's `run_spec`) is not this agent's, when the required
/// row differs (every difference named), or when the log's handlers do
/// not serve the row.
pub fn check_replayable(
    world: &mut World,
    agent: Entity,
    log: &rig_effect_log::EffectLog,
) -> Result<(), rig_core::error::ErrorReport> {
    use rig_core::error::{ErrorKind, ErrorReport};
    let policy = spec_hash(world, agent)
        .ok_or_else(|| ErrorReport::new(ErrorKind::Internal, "the agent's policy does not hash"))?;
    let required = required_row(world, agent);
    let (recorded_policy, recorded_row) = if log.header.programs.is_empty() {
        (log.header.run_spec, log.header.required.clone())
    } else {
        match log
            .header
            .programs
            .values()
            .find(|identity| identity.policy == policy)
        {
            Some(identity) => (Some(identity.policy), identity.required.clone()),
            None => {
                return Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay refused: no program in the log has this agent's policy ({policy:#018x}); the log names {}",
                        log.header
                            .programs
                            .keys()
                            .map(String::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ));
            }
        }
    };
    if recorded_policy != Some(policy) {
        return Err(ErrorReport::new(
            ErrorKind::Internal,
            format!(
                "replay refused: the log was recorded under policy {:?}, this agent's is {policy:#018x}",
                recorded_policy.map(|hash| format!("{hash:#018x}"))
            ),
        ));
    }
    let differences = required.diff(&recorded_row);
    if !differences.is_empty() {
        return Err(ErrorReport::new(
            ErrorKind::Internal,
            format!(
                "replay refused: the required row differs from the log's: {}",
                differences
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        ));
    }
    if let Err(gap) = required.is_subset_of(&log.header.handlers) {
        return Err(ErrorReport::new(
            ErrorKind::HandlerUnavailable,
            format!("replay refused: the log's handlers do not serve the row: {gap}"),
        ));
    }
    Ok(())
}

/// The key an agent mints for its model: `<owner>/model:<label>`.
pub fn model_key(owner: &str, label: &str) -> HandlerKey {
    HandlerKey::from(format!("{owner}/model:{label}"))
}

/// The key an agent mints for a tool: `<owner>/tool:<name>#<n>`.
pub fn tool_key(owner: &str, name: &str, generation: u32) -> HandlerKey {
    HandlerKey::from(format!("{owner}/tool:{name}#{generation}"))
}
