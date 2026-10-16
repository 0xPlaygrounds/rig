//! Plugin corpus (held-out): H31: Repair broken tool-call arguments (spec
//! `H31-tool-arg-repair.md`).
//!
//! Cheap models break tool-call JSON. A model's reply is reviewed before
//! it is committed: each call whose arguments are no JSON object is read
//! leniently (trailing commas, single quotes, unquoted keys, raw newlines
//! and tabs in strings, a missing final bracket, Python's literals, an
//! object sent as a string), and each value is fitted to the tool's schema
//! where that is unambiguous. The repaired reply is what the conversation
//! keeps and what its calls run with. With `store = "original"` the reply
//! is kept as the model sent it and each call is repaired as it is
//! admitted instead. A call beyond repair gets the usual error plus a hint
//! of the arguments the tool takes. `/repairs` counts the repairs per model
//! and kind; the before and after of each go to the debug log.

mod repair;

use std::collections::BTreeMap;

use bevy_log::debug;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::error::ErrorKind;
use rig_harness::rig_core::message::ToolFunction;
use rig_harness::rig_core::tool::ToolErrorKind;
use serde::Deserialize;
use serde_json::{Map, Value};

/// The `[tool_arg_repair]` table of the user's settings.
#[derive(Resource, Deserialize, Default, Debug, Clone, Copy)]
#[serde(default, deny_unknown_fields)]
pub struct ArgRepairSettings {
    /// What the conversation keeps of a repaired call.
    pub store: Store,
}

/// What the conversation keeps of a repaired call.
#[derive(Deserialize, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Store {
    /// The repaired arguments, so the provider reads valid JSON next time.
    #[default]
    Repaired,
    /// The model's own text; only the call runs repaired.
    Original,
}

/// How many repairs of each kind each model's calls needed, by the model's
/// reference.
#[derive(Resource, Default, Debug)]
pub struct Repairs(pub BTreeMap<String, BTreeMap<String, u32>>);

/// The plugin's stage systems, for others to order against: a Prepare
/// system that edits tool arguments goes `.after(ArgRepairSystems::Call)`.
/// `Call` runs before the coding tools give a call its agent's directory.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgRepairSystems {
    /// Review: repairs the calls of a model's reply before it is committed.
    Reply,
    /// Prepare: repairs a tool call as it is admitted.
    Call,
    /// Review: adds the hint to a call refused for its arguments.
    Hint,
}

#[derive(Default)]
pub struct ToolArgRepairPlugin;

impl Plugin for ToolArgRepairPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ArgRepairSettings>("tool_arg_repair")
            .init_resource::<Repairs>()
            .configure_sets(AgentPass, ArgRepairSystems::Reply.in_set(Stage::Review))
            .configure_sets(
                AgentPass,
                ArgRepairSystems::Call.in_set(PrepareSystems::Input),
            )
            .configure_sets(AgentPass, ArgRepairSystems::Hint.in_set(Stage::Review))
            .add_systems(
                AgentPass,
                (
                    repair_replies.in_set(ArgRepairSystems::Reply),
                    repair_calls.in_set(ArgRepairSystems::Call),
                    hint.in_set(ArgRepairSystems::Hint),
                ),
            )
            .add_command(
                "repairs",
                "How many tool calls were repaired, per model and kind",
                list,
            );
    }
}

/// Repairs the calls of each turn's reply in place before it is committed,
/// so the conversation keeps the repaired arguments, unless the settings
/// keep the model's text. A repaired reply needs no repair, so the next
/// pass leaves it alone.
fn repair_replies(
    mut replies: Query<
        (&mut Outcome, Option<&SentTo>),
        (ReadyToSettle, ModelCalls, With<TurnCall>),
    >,
    tools: Query<&ToolDef>,
    settings: Res<ArgRepairSettings>,
    mut repairs: ResMut<Repairs>,
) {
    if settings.store == Store::Original {
        return;
    }
    for (mut outcome, sent) in &mut replies {
        let needs = outcome.completion().is_some_and(|response| {
            response
                .tool_calls()
                .any(|call| repaired(&call.function, &tools).is_some())
        });
        if !needs {
            continue;
        }
        let model = sent.map_or_else(|| "unknown".to_owned(), |sent| sent.0.spec.reference());
        let Ok(Answered::Completion(response)) = &mut outcome.0 else {
            continue;
        };
        for part in &mut std::sync::Arc::make_mut(response).choice {
            if let AssistantContent::ToolCall(call) = part {
                apply(&mut call.function, &tools, &model, &mut repairs);
            }
        }
    }
}

/// Repairs each tool call as it is admitted: one whose reply kept the
/// model's text, or one no reply asked for.
fn repair_calls(
    mut calls: Query<(&mut ToolRequest, &Effect), (Added<Admitted>, ToolCalls)>,
    replies: Query<(&Effect, &SentTo)>,
    tools: Query<&ToolDef>,
    mut repairs: ResMut<Repairs>,
) {
    for (mut request, effect) in &mut calls {
        if repaired(&request.call.function, &tools).is_none() {
            continue;
        }
        let model = replies
            .iter()
            .find(|(reply, _)| Some(reply.id()) == effect.parent)
            .map_or_else(|| "unknown".to_owned(), |(_, sent)| sent.0.spec.reference());
        apply(&mut request.call.function, &tools, &model, &mut repairs);
    }
}

/// On a refused call that has the hint.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Hinted;

/// A call refused for its arguments gets the hint of what they must be,
/// once.
fn hint(
    mut calls: Query<
        (Entity, &ToolRequest, &mut Outcome),
        (ReadyToSettle, ToolCalls, Without<Hinted>),
    >,
    tools: Query<&ToolDef>,
    mut commands: Commands,
) {
    for (call, request, mut outcome) in &mut calls {
        let refused = outcome.0.as_ref().err().is_some_and(|report| {
            matches!(report.kind, ErrorKind::Tool(ToolErrorKind::InvalidArgs))
        });
        let Some(schema) = schema(&request.call.function, &tools).filter(|_| refused) else {
            continue;
        };
        if let Err(report) = &mut outcome.0 {
            let like = repair::summary(schema);
            report.message = format!(
                "{} Hint: arguments must be a JSON object like: {like}",
                report.message
            );
            commands.entity(call).insert(Hinted);
        }
    }
}

/// `/repairs`: the repairs per model and kind.
fn list(In(args): In<CommandArgs>, repairs: Res<Repairs>, mut notices: MessageWriter<Notice>) {
    if repairs.0.is_empty() {
        notices.write(Notice::info(args.agent, "No tool call needed a repair."));
        return;
    }
    let lines: Vec<String> = repairs
        .0
        .iter()
        .map(|(model, kinds)| {
            let kinds: Vec<String> = kinds
                .iter()
                .map(|(kind, count)| format!("{kind} {count}"))
                .collect();
            format!("{model}: {}", kinds.join(", "))
        })
        .collect();
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// The parameters of the tool `function` calls.
fn schema<'a>(function: &ToolFunction, tools: &'a Query<&ToolDef>) -> Option<&'a Value> {
    tools
        .iter()
        .find(|def| def.0.name == function.name)
        .map(|def| &def.0.parameters)
}

/// The arguments `function` should have and the kinds of repair that took,
/// or `None` when they are fine as they are or beyond repair. Arguments
/// that are a JSON object fitting the tool's schema are never touched.
fn repaired(
    function: &ToolFunction,
    tools: &Query<&ToolDef>,
) -> Option<(Map<String, Value>, repair::Kinds)> {
    let schema = schema(function, tools).unwrap_or(&Value::Null);
    let (mut arguments, mut kinds) = match &function.invalid_arguments {
        Some(raw) => repair::parse_broken(raw)?,
        None => (function.arguments.clone(), repair::Kinds::new()),
    };
    repair::coerce(&mut arguments, schema, &mut kinds);
    (!kinds.is_empty()).then_some((arguments, kinds))
}

/// Repairs `function`, made by `model`, counting it and logging the before
/// and after.
fn apply(function: &mut ToolFunction, tools: &Query<&ToolDef>, model: &str, repairs: &mut Repairs) {
    let Some((arguments, kinds)) = repaired(function, tools) else {
        return;
    };
    let before = function.raw_arguments();
    function.arguments = arguments;
    function.invalid_arguments = None;
    let after = function.arguments_value();
    debug!(
        tool = function.name.as_str(),
        model,
        ?kinds,
        before,
        %after,
        "repaired tool-call arguments"
    );
    let counts = repairs.0.entry(model.to_owned()).or_default();
    for kind in kinds {
        *counts.entry(kind.to_owned()).or_default() += 1;
    }
}

#[cfg(test)]
mod tests;
