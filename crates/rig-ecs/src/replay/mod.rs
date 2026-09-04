//! The header from components: a run's program identity as the log names
//! it, computed from the agent entity — never from a spec struct.

use bevy_ecs::prelude::*;
use rig_core::effect::{EffectFamily, EffectRow, HandlerKey};
use rig_effect_log::{EffectLogRecorder, stable_hash};

use crate::{
    agent::{
        AdditionalParams, Context, Conversation, DefaultMaxTurns, DocumentId, DocumentProps,
        DocumentText, Grant, InvalidCalls, MaxTokens, MaxTurns, Order, Output, OutputKind,
        PolicyVersion, Preamble, Remembers, Retrievable, Retrieval, Retrieves, Route, RunOf,
        Streamed, Temperature, ToolChoiceSpec, ToolPolicy, UsesModel,
    },
    bus::{Bound, Scope},
};

/// The JSON the header's `run_spec` hashes for `agent`: the shape
/// `CONTRACT.md` names, built from components, canonicalised by
/// [`stable_hash`]. The builder's identity: the default turn budget, no
/// retries, `fail`. Retained solely for builder/corpus header interoperability;
/// effective replay compatibility uses [`stamp_run`] instead.
fn builder_spec_json(world: &mut World, agent: Entity) -> serde_json::Value {
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

fn effective<T: Component>(world: &World, subject: Entity) -> Option<&T> {
    world.get::<T>(subject).or_else(|| {
        world
            .get::<RunOf>(subject)
            .and_then(|agent| world.get::<T>(agent.0))
    })
}

/// Effective policy from components, using the same run-over-agent precedence
/// as execution. This is not the legacy builder-only `LogHeader::run_spec`.
/// Custom systems and their ordering are represented by `PolicyVersion`.
/// Ambient tool inputs are not serialized or automatically fingerprinted.
pub fn spec_json(world: &mut World, subject: Entity) -> serde_json::Value {
    let agent = world.get::<RunOf>(subject).map_or(subject, |run| run.0);
    let mut spec = builder_spec_json(world, agent);
    if let Some(fields) = spec.as_object_mut() {
        fields.insert(
            "preamble".into(),
            serde_json::json!(effective::<Preamble>(world, subject).and_then(|v| v.0.as_ref())),
        );
        fields.insert(
            "temperature".into(),
            serde_json::json!(effective::<Temperature>(world, subject).and_then(|v| v.0)),
        );
        fields.insert(
            "max_tokens".into(),
            serde_json::json!(effective::<MaxTokens>(world, subject).and_then(|v| v.0)),
        );
        fields.insert(
            "additional_params".into(),
            serde_json::json!(
                effective::<AdditionalParams>(world, subject).and_then(|v| v.0.as_ref())
            ),
        );
        fields.insert(
            "tool_choice".into(),
            serde_json::json!(
                effective::<ToolChoiceSpec>(world, subject).and_then(|v| v.0.as_ref())
            ),
        );
        fields.insert(
            "max_turns".into(),
            serde_json::json!(effective::<MaxTurns>(world, subject).map_or(1, |v| v.0)),
        );
        let invalid = effective::<InvalidCalls>(world, subject)
            .copied()
            .unwrap_or_default();
        fields.insert(
            "max_invalid_tool_call_retries".into(),
            serde_json::json!(invalid.retries),
        );
        fields.insert(
            "unhandled_invalid_tool_call".into(),
            serde_json::json!(invalid.unhandled),
        );
        let output = effective::<Output>(world, subject)
            .cloned()
            .unwrap_or_default();
        fields.insert("output_mode".into(), serde_json::json!(output.mode));
        fields.insert("output_schema".into(), serde_json::json!(output.schema));
        fields.insert(
            "tool_concurrency".into(),
            serde_json::json!(
                effective::<ToolPolicy>(world, subject).map_or(1, |v| v.concurrency.max(1))
            ),
        );
        fields.insert(
            "policy_version".into(),
            serde_json::json!(effective::<PolicyVersion>(world, subject).map(|v| &v.0)),
        );
        fields.insert(
            "streamed".into(),
            serde_json::json!(world.get::<Streamed>(subject).is_some_and(|v| v.0)),
        );
        fields.insert(
            "conversation".into(),
            serde_json::json!(effective::<Conversation>(world, subject).map(|v| &v.0)),
        );
        fields.insert(
            "model".into(),
            serde_json::json!(
                effective::<UsesModel>(world, subject)
                    .and_then(|model| world.get::<Bound>(model.0))
                    .map(|bound| &bound.descriptor)
            ),
        );
        let mut links: Vec<_> = world
            .get::<Children>(agent)
            .into_iter()
            .flat_map(|children| children.iter())
            .filter_map(|link| world.get::<Order>(link).map(|order| (*order, link)))
            .collect();
        links.sort_by_key(|(order, _)| *order);
        let dependencies: Vec<_> = links.into_iter().filter_map(|(_, link)| {
            if let Some(Grant(tool)) = world.get::<Grant>(link) {
                Some(serde_json::json!({"tool": world.get::<Bound>(*tool).map(|b| &b.descriptor), "retrievable": world.get::<Retrievable>(link).is_some()}))
            } else if let Some(Retrieves(index)) = world.get::<Retrieves>(link) {
                Some(serde_json::json!({"index": world.get::<Bound>(*index).map(|b| &b.descriptor), "retrieval": world.get::<Retrieval>(link)}))
            } else { None }
        }).collect();
        fields.insert("dependencies".into(), serde_json::json!(dependencies));
    }
    spec
}

/// The effective policy hash for an agent or run, not a code fingerprint.
pub fn spec_hash(world: &mut World, agent: Entity) -> Option<u64> {
    stable_hash(&spec_json(world, agent)).ok()
}

/// The agent's required effect row: its model, every route, every tool it
/// grants (retrievable or not), every index it retrieves from, and its
/// memory, by their bound keys.
pub fn required_row(world: &mut World, agent: Entity) -> EffectRow {
    let subject = agent;
    let agent = world.get::<RunOf>(subject).map_or(subject, |run| run.0);
    let mut row = EffectRow::new();
    let model = effective::<UsesModel>(world, subject).map(|uses| uses.0);
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

/// Stamp `recorder`'s legacy builder header for corpus interoperability, not
/// a claim of effective run compatibility: the builder spec hash, the
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
    if let Ok(hash) = stable_hash(&builder_spec_json(world, agent)) {
        recorder.set_run_spec(hash);
    }
    let required = required_row(world, agent);
    recorder.set_program(hooks, required, bus);
}

/// Stamp `run`'s program identity under its scope
/// ([`rig_effect_log::LogHeader::programs`]): the effective run's required row
/// and policy hash, keyed by the run's `Scope`. A world running several
/// programs into one log names each this way.
pub fn stamp_run(world: &mut World, run: Entity, recorder: &EffectLogRecorder) {
    let Some(_) = world.get::<RunOf>(run) else {
        return;
    };
    let Some(scope) = world.get::<Scope>(run).map(|scope| scope.0.clone()) else {
        return;
    };
    let Some(policy) = spec_hash(world, run) else {
        return;
    };
    let required = required_row(world, run);
    recorder.set_program_identity(scope, rig_effect_log::ProgramIdentity { required, policy });
}

/// Check `run` against its explicitly named `Scope` in `log`, before dispatch.
/// Requires a nonempty `PolicyVersion`: arbitrary systems cannot be
/// automatically fingerprinted. A missing declaration or a builder-only log
/// is reported as unverified, not accepted as replay-compatible. Success
/// verifies the declared policy and supported configuration, not ambient
/// inputs, execution ordering beyond that declaration, or external state.
pub fn check_replayable(
    world: &mut World,
    run: Entity,
    log: &rig_effect_log::EffectLog,
) -> Result<(), rig_core::error::ErrorReport> {
    use rig_core::error::{ErrorKind, ErrorReport};
    rig_effect_log::EffectLogReplayer::check_header(log)?;
    if world.get::<RunOf>(run).is_none() {
        return Err(ErrorReport::new(
            ErrorKind::Request,
            "replay compatibility unverified: provide a run with an explicit Scope, not an agent",
        ));
    }
    let scope = world
        .get::<Scope>(run)
        .ok_or_else(|| {
            ErrorReport::new(
                ErrorKind::Request,
                "replay compatibility unverified: the run has no Scope",
            )
        })?
        .0
        .clone();
    let policy = spec_hash(world, run)
        .ok_or_else(|| ErrorReport::new(ErrorKind::Internal, "the agent's policy does not hash"))?;
    let required = required_row(world, run);
    let identity = log.header.programs.get(&scope).ok_or_else(|| ErrorReport::new(ErrorKind::Request,
        format!("replay compatibility unverified: the log has no program for scope `{scope}`; builder identity is insufficient")))?;
    let (recorded_policy, recorded_row) = (Some(identity.policy), &identity.required);
    if recorded_policy != Some(policy) {
        return Err(ErrorReport::new(
            ErrorKind::Internal,
            format!(
                "replay refused: the log was recorded under policy {:?}, this agent's is {policy:#018x}",
                recorded_policy.map(|hash| format!("{hash:#018x}"))
            ),
        ));
    }
    let differences = required.diff(recorded_row);
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
    if effective::<PolicyVersion>(world, run).is_none_or(|version| version.0.trim().is_empty()) {
        return Err(ErrorReport::new(
            ErrorKind::Request,
            "replay compatibility unverified: declare a nonempty PolicyVersion for custom systems, ordering and configuration",
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
