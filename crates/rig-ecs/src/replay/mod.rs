//! The header from components: a run's program identity as the log names
//! it, computed from the agent entity — never from a spec struct.

use bevy_ecs::prelude::*;
use rig_core::effect::{EffectFamily, EffectRow, HandlerKey};
use rig_effect_log::{EffectLogRecorder, stable_hash};

use crate::{
    agent::{
        AdditionalParams, Context, DefaultMaxTurns, DocumentId, DocumentProps, DocumentText, Grant,
        MaxTokens, Order, Output, OutputKind, Preamble, Route, Temperature, ToolChoiceSpec,
        UsesModel,
    },
    bus::Bound,
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

/// The agent's required effect row: its model, every route, and every
/// tool it grants, by their bound keys.
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

/// The key an agent mints for its model: `<owner>/model:<label>`.
pub fn model_key(owner: &str, label: &str) -> HandlerKey {
    HandlerKey::from(format!("{owner}/model:{label}"))
}

/// The key an agent mints for a tool: `<owner>/tool:<name>#<n>`.
pub fn tool_key(owner: &str, name: &str, generation: u32) -> HandlerKey {
    HandlerKey::from(format!("{owner}/tool:{name}#{generation}"))
}
