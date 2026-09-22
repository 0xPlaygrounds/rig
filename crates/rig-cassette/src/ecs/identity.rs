//! Program identity and replay compatibility derived from ECS components.
//!
//! ```
//! use bevy_ecs::prelude::*;
//! use rig_cassette::ecs::identity::required_row;
//! let mut world = World::new();
//! let agent = world.spawn_empty().id();
//! let required = required_row(&mut world, agent);
//! ```

use crate::effect_log::{EffectLogRecorder, stable_hash};
use bevy_ecs::prelude::*;
use rig_core::effect::{EffectFamily, EffectRow};

use rig_ecs::{
    agent::{
        AdditionalParams, Context, Conversation, DefaultMaxTurns, DocumentId, DocumentProps,
        DocumentText, Grant, InvalidCalls, MaxTokens, MaxTurns, Output, OutputKind,
        OutputToolConfig, PolicyVersion, Preamble, Remembers, Retrievable, Retrieval, Retrieves,
        Route, RunOf, StreamRequested, Temperature, ToolChoiceSpec, ToolPolicy, UsesModel,
    },
    bus::{Bound, Scope},
};

/// Return builder identity JSON from agent components, using the default turn
/// budget, zero retries, and failure for invalid calls.
/// Effective replay identity is stamped separately by [`stamp_run`].
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
    let output_tool = world
        .get::<OutputToolConfig>(agent)
        .cloned()
        .unwrap_or_default();
    let max_turns = world
        .get::<DefaultMaxTurns>(agent)
        .and_then(|d| d.0)
        .unwrap_or(1);
    let mut context: Vec<serde_json::Value> = Vec::new();
    if let Some(children) = world.get::<Children>(agent) {
        let links: Vec<Entity> = children.iter().collect();
        for child in links {
            if let Some(Context(document)) = world.get::<Context>(child) {
                let document = *document;
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
                    context.push(serde_json::Value::Object(json));
                }
            }
        }
    }
    let output_mode = match output.mode {
        OutputKind::Auto => "Auto",
        OutputKind::Native => "Native",
        OutputKind::Tool => "Tool",
        OutputKind::Prompted => "Prompted",
    };
    serde_json::json!({
        "preamble": preamble,
        "static_context": context,
        "additional_params": additional_params,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "max_turns": max_turns,
        "max_invalid_tool_call_retries": 0,
        "output_schema": output.schema,
        "output_mode": output_mode,
        "output_tool_name": output_tool.name,
        "output_tool_description": output_tool.description,
        "augment_output_preamble": output_tool.augment_preamble,
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

/// Return effective policy JSON with run components taking precedence over agent
/// components. Custom systems and ordering require a declared `PolicyVersion`;
/// ambient tool inputs are not serialized or fingerprinted.
pub fn spec_json(world: &mut World, subject: Entity) -> serde_json::Value {
    let agent = world.get::<RunOf>(subject).map_or(subject, |run| run.0);
    let access = effective::<rig_ecs::agent::ToolAccess>(world, subject).cloned();
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
        fields.insert(
            "provider_retries".into(),
            serde_json::json!(
                effective::<rig_ecs::agent::ProviderRetries>(world, subject)
                    .map_or(rig_ecs::agent::DEFAULT_PROVIDER_RETRIES, |v| v.0)
            ),
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
        let output_tool = effective::<OutputToolConfig>(world, subject)
            .cloned()
            .unwrap_or_default();
        fields.insert(
            "output_tool_name".into(),
            serde_json::json!(output_tool.name),
        );
        fields.insert(
            "output_tool_description".into(),
            serde_json::json!(output_tool.description),
        );
        fields.insert(
            "augment_output_preamble".into(),
            serde_json::json!(output_tool.augment_preamble),
        );
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
            serde_json::json!(world.get::<StreamRequested>(subject).is_some_and(|v| v.0)),
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
        let links: Vec<Entity> = world
            .get::<Children>(agent)
            .into_iter()
            .flat_map(|children| children.iter())
            .collect();
        let dependencies: Vec<_> = links.into_iter().filter_map(|link| {
            if let Some(Grant(tool)) = world.get::<Grant>(link) {
                Some(serde_json::json!({"tool": world.get::<Bound>(*tool).map(|b| &b.descriptor), "retrievable": world.get::<Retrievable>(link).is_some()}))
            } else if let Some(Retrieves(index)) = world.get::<Retrieves>(link) {
                Some(serde_json::json!({"index": world.get::<Bound>(*index).map(|b| &b.descriptor), "retrieval": world.get::<Retrieval>(link)}))
            } else { None }
        }).collect();
        fields.insert("dependencies".into(), serde_json::json!(dependencies));
        if let Some(access) =
            access.filter(|access| access != &rig_ecs::agent::ToolAccess::default())
        {
            let executable_dependencies: Vec<_> = access
                .executable
                .iter()
                .flat_map(|map| map.values())
                .map(|key| {
                    let descriptor = world
                        .query::<&Bound>()
                        .iter(world)
                        .find(|bound| &bound.key == key)
                        .map(|bound| bound.descriptor.clone());
                    serde_json::json!({"key": key, "descriptor": descriptor})
                })
                .collect();
            fields.insert("tool_access".into(), serde_json::json!(access));
            fields.insert(
                "executable_dependencies".into(),
                serde_json::json!(executable_dependencies),
            );
        }
    }
    spec
}

/// Return the effective policy hash for an agent or run, or `None` if
/// serialization fails. This is not a code fingerprint.
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
    if let Some(executable) = effective::<rig_ecs::agent::ToolAccess>(world, subject)
        .and_then(|access| access.executable.as_ref())
    {
        for key in executable.values() {
            row.insert(key.clone(), EffectFamily::Tool);
        }
    }
    // Persisted turn snapshots can still dispatch bindings from before a run
    // policy change. They remain dependencies of a resumed run.
    for (parent, access) in world
        .query_filtered::<(&ChildOf, &rig_ecs::agent::ToolAccess), With<rig_ecs::agent::Turn>>()
        .iter(world)
    {
        let run = parent.parent();
        if run == subject
            || (subject == agent
                && world
                    .get::<RunOf>(run)
                    .is_some_and(|owner| owner.0 == agent))
        {
            for key in access
                .executable
                .iter()
                .flat_map(|bindings| bindings.values())
            {
                row.insert(key.clone(), EffectFamily::Tool);
            }
        }
    }
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

/// Stamp the effective required row and policy hash under the run's `Scope` in
/// [`crate::effect_log::LogHeader::programs`]. Returns an error if `run` lacks
/// `RunOf` or `Scope`, or its policy cannot be hashed.
pub fn stamp_run(
    world: &mut World,
    run: Entity,
    recorder: &EffectLogRecorder,
) -> Result<(), rig_core::error::ErrorReport> {
    use rig_core::error::{ErrorKind, ErrorReport};
    if world.get::<RunOf>(run).is_none() {
        return Err(ErrorReport::new(
            ErrorKind::Request,
            "cannot stamp program identity: provide a run with an explicit Scope, not an agent",
        ));
    }
    let scope = world
        .get::<Scope>(run)
        .ok_or_else(|| {
            ErrorReport::new(
                ErrorKind::Request,
                "cannot stamp program identity: the run has no Scope",
            )
        })?
        .0
        .clone();
    let policy = spec_hash(world, run)
        .ok_or_else(|| ErrorReport::new(ErrorKind::Internal, "the agent's policy does not hash"))?;
    let required = required_row(world, run);
    recorder.set_program_identity(
        scope,
        crate::effect_log::ProgramIdentity { required, policy },
    );
    Ok(())
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
    log: &crate::effect_log::EffectLog,
) -> Result<(), rig_core::error::ErrorReport> {
    use rig_core::error::{ErrorKind, ErrorReport};
    crate::effect_log::EffectLogReplayer::check_header(log)?;
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
