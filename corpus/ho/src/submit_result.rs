//! Plugin corpus (held-out): H07: Structured result tool that ends the turn
//! (spec `H07-submit-result.md`).
//!
//! `--schema FILE` (or `/schema FILE`) offers the top-level agent the
//! `submit_result` tool, whose parameters are the JSON schema in `FILE`; a
//! schema whose root is no object refuses the start. A valid call is the
//! run's result ([`SubmitResult`]): once the other calls of its reply
//! settled, a Gate system answers the turn's next model call with the JSON
//! itself, so nothing more is sent and the turn ends answered with it, which
//! `--print` prints alone, on one line. The stack refuses arguments that do
//! not fit the schema with its complaints, and the third such call fails the run
//! (exit code 2). A turn that ends in text gets one reminder; the second
//! such end fails the run (exit code 3). In the terminal view the answer is
//! a JSON block and the tool goes after one success, unless `--schema-keep`.

use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Usage};
use rig_harness::rig_core::effect;
use rig_harness::rig_core::message::{AssistantContent, Origin as Made, Text};
use serde_json::Value;

/// The tool's name.
pub const TOOL: &str = "submit_result";

/// The option naming the schema file.
pub const SCHEMA: &str = "schema";

/// The option that keeps the tool after its first valid call.
pub const KEEP: &str = "schema-keep";

/// Invalid calls of one turn that fail the run.
pub const MAX_INVALID: u32 = 3;

/// The exit code after [`MAX_INVALID`] invalid calls, and of a refused
/// start.
pub const INVALID_EXIT: u8 = 2;

/// The exit code after a second turn end without a result.
pub const NO_RESULT_EXIT: u8 = 3;

/// What a turn that ended in text is told, once.
pub const REMINDER: &str =
    "You ended your turn without a result: call submit_result with it, as its schema says.";

const DESCRIPTION: &str = "Submit the task's result, as its parameters' schema says. When the \
     task is done, call it with the result instead of answering in text; a valid call ends your \
     turn.";

/// The run's result: the arguments of the last valid `submit_result` call.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SubmitResult(pub Option<Value>);

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubmitSystems {
    /// Review: counts each `submit_result` outcome, and reminds or fails a
    /// turn that would end without a result.
    Review,
    /// Gate: answers a submitted turn's next model call with the result.
    Gate,
}

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
        let options = ToolOptions {
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
        .configure_sets(
            AgentPass,
            (
                SubmitSystems::Review.in_set(Stage::Review),
                SubmitSystems::Gate.in_set(Stage::Gate),
            ),
        )
        .add_systems(
            AgentPass,
            (
                (judge, remind).chain().in_set(SubmitSystems::Review),
                end_with_result.in_set(SubmitSystems::Gate),
            ),
        );
        if invoked.flag(SCHEMA) {
            let path = invoked.flag_value(SCHEMA).map(str::to_owned);
            app.insert_resource(StartupSchema(path))
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

/// `--schema`'s file; none when it came without one.
#[derive(Resource)]
struct StartupSchema(Option<String>);

/// How a turn fares with `submit_result`, on every turn.
#[derive(Component, Default)]
struct Submitting {
    invalid: u32,
    reminded: bool,
    /// The turn's answer, once a valid result came in.
    answer: Option<String>,
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
        .and_modify(move |mut def| std::sync::Arc::make_mut(&mut def.0).parameters = schema)
        .entity()
        .remove::<Disabled>();
}

/// `--schema=FILE`: offers the tool, or refuses the start.
fn load_startup_schema(
    file: Res<StartupSchema>,
    root: Option<Res<ProjectRoot>>,
    tool: Res<SubmitTool>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let schema = match &file.0 {
        Some(path) => read_schema(root.as_deref().map(|root| root.0.as_path()), path),
        None => Err("needs a file: --schema FILE".to_owned()),
    };
    match schema {
        Ok(schema) => offer(&mut commands, tool.entity, schema),
        Err(why) => {
            notices.write(Notice::error(None, format!("--{SCHEMA}: {why}")));
            commands.trigger(Exit(ExitReason::Failed(INVALID_EXIT)));
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

/// Answers a `submit_result` call, whose arguments the stack already
/// checked against the tool's schema (a call that does not fit is refused
/// with the schema's complaints before it gets here): accepted, unless a
/// spawned agent made it, as only a top-level agent submits the run's
/// result.
fn submit(
    called: On<ToolCalled<Value>>,
    spawned: Query<(), With<SpawnedBy>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let result = if spawned.contains(called.agent) {
        ToolResult::failed(ToolExecutionError::refused(
            "Only the main agent submits the result.",
        ))
    } else {
        ToolResult::success("Result accepted; the turn ends.".into())
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("ho-submit-result"), result);
    }
}

/// Counts each `submit_result` outcome once: a success is the run's result
/// and the turn's answer; the third failure of a turn fails it, and a run
/// nobody sits at.
fn judge(
    calls: Query<
        (Entity, &ToolRequest, &Outcome, &CallOf),
        (ReadyToSettle, ToolCalls, Without<Judged>),
    >,
    mut turns: Query<&mut Submitting>,
    tool: Res<SubmitTool>,
    mut result: ResMut<SubmitResult>,
    mut commands: Commands,
) {
    for (call, request, outcome, &CallOf(turn)) in &calls {
        if request.call.function.name.as_str() != TOOL {
            continue;
        }
        commands.entity(call).insert(Judged);
        let Ok(mut state) = turns.get_mut(turn) else {
            continue;
        };
        let answered = outcome.tool_result();
        if answered.is_success() {
            let value = Value::Object(request.call.function.arguments.clone());
            let answer = match tool.interactive {
                true => serde_json::to_string_pretty(&value)
                    .map(|pretty| format!("```json\n{pretty}\n```")),
                false => serde_json::to_string(&value),
            };
            state.answer = answer.ok();
            result.0 = Some(value);
            if !tool.keep {
                commands.entity(tool.entity).insert(Disabled);
            }
            continue;
        }
        state.invalid += 1;
        if state.invalid < MAX_INVALID {
            continue;
        }
        let errors = answered.output().render();
        let why = format!("{MAX_INVALID} invalid {TOOL} calls; the last: {errors}");
        commands.entity(turn).insert(TurnOutcome::Failed(why));
        if !tool.interactive {
            commands.trigger(Exit(ExitReason::Failed(INVALID_EXIT)));
        }
    }
}

/// A top-level agent's turn that would end in text while the tool is
/// offered gets one reminder; the second such end fails the run, when
/// nobody sits at it.
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
        let answered = matches!(outcome, TurnOutcome::Answered(_));
        if !answered || state.answer.is_some() || !top.contains(agent) {
            continue;
        }
        if !state.reminded {
            state.reminded = true;
            let reminder = Deliver::new(agent, REMINDER, DeliveryMode::Queue);
            commands.trigger(reminder.with_author(Author::plugin("submit-result")));
            continue;
        }
        let why = format!("The turn ended twice without calling {TOOL}.");
        notices.write(Notice::error(agent, why));
        if !tool.interactive {
            commands.trigger(Exit(ExitReason::Failed(NO_RESULT_EXIT)));
        }
    }
}

/// Answers a submitted turn's next model call, which the kernel asks for
/// once every other call of the reply settled, with the result: nothing is
/// sent and the turn ends answered with it.
fn end_with_result(
    mut calls: Query<(&CallOf, &mut Decision), (ReadyToStart, ModelCalls, With<TurnCall>)>,
    turns: Query<&Submitting>,
) {
    for (&CallOf(turn), mut decision) in &mut calls {
        let Ok(Submitting {
            answer: Some(answer),
            ..
        }) = turns.get(turn)
        else {
            continue;
        };
        let choice = vec![AssistantContent::Text(Text::new(answer.clone()))];
        let made = Made::new("plugin", "submit-result", TOOL);
        let reply = CompletionResponse::new(choice, Usage::default(), made, Value::Null);
        decision.decide(
            Decider::plugin("ho-submit-result"),
            Outcome(Ok(effect::Answered::Completion(reply.into()))),
        );
    }
}

#[cfg(test)]
mod tests;
