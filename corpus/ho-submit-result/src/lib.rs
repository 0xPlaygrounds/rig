//! Plugin corpus (held-out): H07: Structured result tool that ends the turn
//! (spec `H07-submit-result.md`).
//!
//! `--schema=FILE` (or `/schema FILE`) offers the top-level agent the
//! `submit_result` tool, whose parameters are the JSON schema in `FILE`; a
//! schema whose root is no object is a startup error. A valid call is the
//! run's result ([`SubmitResult`]) and ends the turn once the other calls of
//! its reply settled, with no further model request: `--print` writes it
//! alone, on one line. Invalid arguments get the schema's complaints as the
//! tool's result, and the third invalid call fails the run (exit code 2).
//! A turn that ends in text gets one reminder; the second such end fails
//! the run (exit code 3). In the terminal view the result is shown as a
//! JSON block and the tool goes after one success, unless `--schema-keep`.

use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ToolErrorKind;
use rig_print::PrintSection;
use serde_json::Value;

/// The tool's name.
pub const TOOL: &str = "submit_result";

/// The option naming the schema file.
pub const SCHEMA: &str = "schema";

/// The option that keeps the tool after its first valid call.
pub const KEEP: &str = "schema-keep";

/// Invalid calls of one turn that fail the run.
pub const MAX_INVALID: u32 = 3;

/// The exit code after [`MAX_INVALID`] invalid calls.
pub const INVALID_EXIT: u8 = 2;

/// The exit code after a second turn end without a result.
pub const NO_RESULT_EXIT: u8 = 3;

/// What a turn that ended in text is told, once.
pub const REMINDER: &str =
    "You ended your turn without a result: call submit_result with it, as its schema says.";

/// Why a turn whose result came in ends.
pub const SUBMITTED: &str = "submit_result was called";

const DESCRIPTION: &str =
    "Submit the task's result, as its parameters' schema says. A valid call ends the turn.";

const RULES: &[&str] = &[
    "When the task is done, call `submit_result` with the result instead of answering in text; a \
     valid call ends your turn.",
];

/// The run's result: the arguments of the last valid `submit_result` call.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SubmitResult(pub Option<Value>);

impl SubmitResult {
    /// What `--print` writes: the result as one line of JSON.
    pub fn printed(&self) -> Option<String> {
        self.0.as_ref().map(Value::to_string)
    }
}

/// The plugin's Review systems, for other reviewers to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubmitSystems;

/// Offers `submit_result` and ends the turns it answers.
#[derive(Default)]
pub struct SubmitResultPlugin;

impl Plugin for SubmitResultPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(
            Flag::new(
                SCHEMA,
                "Offer `submit_result`, whose result is the run's answer",
            )
            .value("FILE"),
        )
        .add_flag(Flag::new(
            KEEP,
            "Keep `submit_result` after its first valid call",
        ));
        let world = app.world_mut();
        let taken = world
            .query::<&ToolDef>()
            .iter(world)
            .any(|def| def.0.name.as_str() == TOOL);
        if taken {
            // Another plugin's tool of the name stays.
            return;
        }
        let options = ToolOptions {
            rules: RULES,
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_open_tool(TOOL, DESCRIPTION, options, submit);
        let world = app.world_mut();
        let tool = world
            .query::<(Entity, &ToolDef)>()
            .iter(world)
            .find_map(|(entity, def)| (def.0.name.as_str() == TOOL).then_some(entity));
        let Some(tool) = tool else {
            return;
        };
        // Off until a schema comes.
        world.entity_mut(tool).insert(Disabled);
        world.spawn((ResultSection, PrintSection::new(TOOL, "")));
        let invoked = world.get_resource::<Invoked>().cloned().unwrap_or_default();
        app.insert_resource(SubmitTool {
            entity: tool,
            keep: invoked.flag(KEEP),
            interactive: invoked.interactive(),
        })
        .init_resource::<SubmitResult>()
        .register_required_components::<TurnOf, Submitting>()
        .add_command(
            "schema",
            "Offer submit_result with this JSON schema: /schema FILE",
            schema,
        )
        .configure_sets(AgentPass, SubmitSystems.in_set(Stage::Review))
        .add_systems(
            AgentPass,
            (judge, end_submitted, remind).chain().in_set(SubmitSystems),
        );
        if let Some(path) = invoked.flag_value(SCHEMA) {
            app.insert_resource(StartupSchema(path.to_owned()))
                .add_systems(Startup, load_startup_schema);
        }
    }
}

/// The tool's entity and how the run treats it.
#[derive(Resource)]
struct SubmitTool {
    entity: Entity,
    /// `--schema-keep`.
    keep: bool,
    /// Someone sits at the terminal: failures end the turn, not the run.
    interactive: bool,
}

/// `--schema`'s file.
#[derive(Resource)]
struct StartupSchema(String);

/// The `--print` section that holds the result.
#[derive(Component)]
struct ResultSection;

/// How a turn fares with `submit_result`, on every turn.
#[derive(Component, Default)]
struct Submitting {
    invalid: u32,
    reminded: bool,
    submitted: bool,
}

/// On a `submit_result` call already counted.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Judged;

/// The schema in `path`, relative to the project, if its root is an
/// object.
fn read_schema(root: Option<&Path>, path: &str) -> Result<Value, String> {
    let path = root.map_or_else(|| PathBuf::from(path), |root| root.join(path));
    let text = std::fs::read_to_string(&path)
        .map_err(|failure| format!("cannot read {}: {failure}", path.display()))?;
    let schema: Value = serde_json::from_str(&text)
        .map_err(|failure| format!("{} is no JSON: {failure}", path.display()))?;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(format!(
            "the root of the schema in {} must be an object (\"type\": \"object\")",
            path.display()
        ));
    }
    Ok(schema)
}

/// Gives the tool `schema` as its parameters and turns it on.
fn offer(commands: &mut Commands, tool: Entity, schema: Value) {
    commands
        .entity(tool)
        .entry::<ToolDef>()
        .and_modify(move |mut def| def.0.parameters = schema)
        .entity()
        .remove::<Disabled>();
}

/// `--schema=FILE`: offers the tool, or fails the start.
fn load_startup_schema(
    file: Res<StartupSchema>,
    root: Option<Res<ProjectRoot>>,
    tool: Res<SubmitTool>,
    mut commands: Commands,
) {
    match read_schema(root.as_deref().map(|root| root.0.as_path()), &file.0) {
        Ok(schema) => offer(&mut commands, tool.entity, schema),
        Err(why) => {
            eprintln!("rig: --{SCHEMA}: {why}");
            commands.trigger(Exit(ExitReason::Failed(2)));
        }
    }
}

/// `/schema FILE`.
fn schema(
    In(args): In<CommandArgs>,
    root: Option<Res<ProjectRoot>>,
    tool: Res<SubmitTool>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        notices.write(Notice::error(
            args.agent,
            "/schema needs a file: /schema FILE",
        ));
        return;
    }
    match read_schema(root.as_deref().map(|root| root.0.as_path()), &args.args) {
        Ok(schema) => {
            offer(&mut commands, tool.entity, schema);
            let text = format!("`{TOOL}` takes the schema in {}.", args.args);
            notices.write(Notice::info(args.agent, text));
        }
        Err(why) => {
            notices.write(Notice::error(args.agent, format!("/schema: {why}")));
        }
    }
}

/// Answers a `submit_result` call: accepted when its arguments fit the
/// tool's schema, else the schema's complaints. Only a top-level agent
/// submits the run's result.
fn submit(
    called: On<ToolCalled<Value>>,
    tools: Query<&ToolDef>,
    spawned: Query<(), With<SpawnedBy>>,
    mut commands: Commands,
) {
    let result = if spawned.contains(called.agent) {
        ToolResult::failed(ToolExecutionError::refused(
            "Only the main agent submits the result.",
        ))
    } else {
        let schema = tools.get(called.entity).map(|def| &def.0.parameters);
        let mut errors = Vec::new();
        if let Ok(schema) = schema {
            violations(schema, &called.args, "$", &mut errors);
        }
        if errors.is_empty() {
            ToolResult::success("Result accepted; the turn ends.".into())
        } else {
            let why = format!(
                "The result does not fit the schema:\n- {}\nFix it and call {TOOL} again.",
                errors.join("\n- ")
            );
            ToolResult::failed(ToolExecutionError::new(ToolErrorKind::InvalidArgs, why))
        }
    };
    commands
        .entity(called.call)
        .insert_if_new(Outcome::from(result));
}

/// Counts each `submit_result` outcome once: a success is the run's
/// result, and the turn ends; the third failure of a turn fails it, and a
/// run nobody sits at.
fn judge(
    calls: Query<
        (Entity, &ToolRequest, &Outcome, &ForAgent, Option<&CallOf>),
        (Unreviewed, Without<Judged>),
    >,
    mut turns: Query<&mut Submitting>,
    tool: Res<SubmitTool>,
    mut result: ResMut<SubmitResult>,
    mut sections: Query<&mut PrintSection, With<ResultSection>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, request, outcome, &ForAgent(agent), of) in &calls {
        if request.call.function.name.as_str() != TOOL {
            continue;
        }
        commands.entity(call).insert(Judged);
        let turn = of.map(|of| of.0);
        let state = turn.and_then(|turn| turns.get_mut(turn).ok());
        let answered = outcome.tool_result();
        if answered.is_success() {
            let value = Value::Object(request.call.function.arguments.clone());
            for mut section in &mut sections {
                section.text = value.to_string();
            }
            if tool.interactive {
                let pretty = serde_json::to_string_pretty(&value).unwrap_or_default();
                notices.write(Notice::info(agent, format!("```json\n{pretty}\n```")));
            }
            if !tool.keep {
                commands.entity(tool.entity).insert(Disabled);
            }
            result.0 = Some(value);
            if let Some(mut state) = state {
                state.submitted = true;
            }
            continue;
        }
        let (Some(turn), Some(mut state)) = (turn, state) else {
            continue;
        };
        state.invalid += 1;
        if state.invalid < MAX_INVALID {
            continue;
        }
        let why = format!("{MAX_INVALID} invalid {TOOL} calls");
        let errors = answered.output().render();
        notices.write(Notice::error(agent, format!("{why}; the last: {errors}")));
        commands.entity(turn).insert(TurnOutcome::Failed(why));
        if !tool.interactive {
            commands.trigger(Exit(ExitReason::Failed(INVALID_EXIT)));
        }
    }
}

/// Ends each turn whose result came in once every other call of its reply
/// is answered and unheld, so they keep their results and the turn calls
/// the model no more.
fn end_submitted(
    turns: Query<(Entity, &Submitting), (With<TurnOf>, Without<TurnOutcome>)>,
    calls: Query<(&CallOf, Has<Outcome>, Has<Holds>), (With<ToolRequest>, Without<Settled>)>,
    mut commands: Commands,
) {
    for (turn, _) in turns.iter().filter(|(_, state)| state.submitted) {
        let open = calls
            .iter()
            .any(|(of, answered, held)| of.0 == turn && (!answered || held));
        if !open {
            let stopped = TurnOutcome::Stopped(Some(SUBMITTED.to_owned()));
            commands.entity(turn).insert(stopped);
        }
    }
}

/// A top-level agent's turn that would end in text while the tool is
/// offered gets one reminder; the second such end fails it, and a run
/// nobody sits at.
fn remind(
    mut turns: Query<(&TurnOf, &TurnOutcome, &mut Submitting), Added<TurnOutcome>>,
    top: Query<(), (With<Agent>, Without<SpawnedBy>)>,
    offered: Query<(), With<ToolDef>>,
    tool: Res<SubmitTool>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if !offered.contains(tool.entity) {
        return;
    }
    for (&TurnOf(agent), outcome, mut state) in &mut turns {
        if !matches!(outcome, TurnOutcome::Answered(_)) || state.submitted || !top.contains(agent) {
            continue;
        }
        if !state.reminded {
            state.reminded = true;
            let reminder = Deliver::new(agent, REMINDER, DeliveryMode::Queue);
            commands.trigger(reminder.with_origin(Origin::plugin("submit-result")));
            continue;
        }
        let why = format!("The turn ended twice without calling {TOOL}.");
        notices.write(Notice::error(agent, why));
        if !tool.interactive {
            commands.trigger(Exit(ExitReason::Failed(NO_RESULT_EXIT)));
        }
    }
}

/// Collects what in `value`, at the path `at`, does not fit `schema`. The
/// keywords read: `type`, `enum`, `const`, `properties`, `required`,
/// `additionalProperties`, `items`, `minItems`, `maxItems`, `minLength`,
/// `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`,
/// `exclusiveMaximum`, `allOf`, `anyOf` and `oneOf`; others are ignored.
fn violations(schema: &Value, value: &Value, at: &str, errors: &mut Vec<String>) {
    let Some(schema) = schema.as_object() else {
        return;
    };
    let types: Vec<&str> = match schema.get("type") {
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    if !types.is_empty() && !types.iter().any(|name| is_type(value, name)) {
        errors.push(format!(
            "{at}: expected {}, got {value}",
            types.join(" or ")
        ));
        return;
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        let options = Value::Array(options.clone());
        errors.push(format!("{at}: {value} is not one of {options}"));
    }
    if let Some(expected) = schema.get("const")
        && expected != value
    {
        errors.push(format!("{at}: must be {expected}"));
    }
    let number = |key: &str| schema.get(key).and_then(Value::as_f64);
    let count = |key: &str| schema.get(key).and_then(Value::as_u64);
    let mut out_of = |what: &str, len: usize, min: &str, max: &str| {
        let len = len as u64;
        if count(min).is_some_and(|min| len < min) || count(max).is_some_and(|max| len > max) {
            let (min, max) = (count(min), count(max));
            errors.push(format!("{at}: {len} {what}, not within {min:?}..={max:?}"));
        }
    };
    match value {
        Value::Object(fields) => {
            let required = schema.get("required").and_then(Value::as_array);
            for name in required.into_iter().flatten().filter_map(Value::as_str) {
                if !fields.contains_key(name) {
                    errors.push(format!("{at}: `{name}` is required"));
                }
            }
            let properties = schema.get("properties").and_then(Value::as_object);
            for (name, field) in fields {
                let path = format!("{at}.{name}");
                match (
                    properties.and_then(|properties| properties.get(name)),
                    schema.get("additionalProperties"),
                ) {
                    (Some(field_schema), _) | (None, Some(field_schema @ Value::Object(_))) => {
                        violations(field_schema, field, &path, errors);
                    }
                    (None, Some(Value::Bool(false))) => {
                        errors.push(format!("{path}: no such property"));
                    }
                    _ => {}
                }
            }
        }
        Value::Array(items) => {
            out_of("items", items.len(), "minItems", "maxItems");
            if let Some(item_schema) = schema.get("items") {
                for (index, item) in items.iter().enumerate() {
                    violations(item_schema, item, &format!("{at}[{index}]"), errors);
                }
            }
        }
        Value::String(text) => out_of("characters", text.chars().count(), "minLength", "maxLength"),
        Value::Number(n) => {
            let n = n.as_f64().unwrap_or_default();
            let low = number("minimum").is_some_and(|min| n < min)
                || number("exclusiveMinimum").is_some_and(|min| n <= min);
            let high = number("maximum").is_some_and(|max| n > max)
                || number("exclusiveMaximum").is_some_and(|max| n >= max);
            if low || high {
                errors.push(format!("{at}: {n} is out of range"));
            }
        }
        _ => {}
    }
    let fits = |sub: &Value| {
        let mut found = Vec::new();
        violations(sub, value, at, &mut found);
        found
    };
    let subs = |key: &str| {
        schema
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
    };
    for sub in subs("allOf") {
        errors.extend(fits(sub));
    }
    if schema.contains_key("anyOf") && !subs("anyOf").any(|sub| fits(sub).is_empty()) {
        errors.push(format!("{at}: fits none of anyOf"));
    }
    let one = subs("oneOf").filter(|sub| fits(sub).is_empty()).count();
    if schema.contains_key("oneOf") && one != 1 {
        errors.push(format!("{at}: fits {one} of oneOf, not exactly one"));
    }
}

/// Whether `value` is of the JSON Schema type `name`.
fn is_type(value: &Value, name: &str) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

#[cfg(test)]
mod tests;
