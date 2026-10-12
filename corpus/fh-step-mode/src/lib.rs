//! Step-through mode: `/step` makes every tool call of the focused agent
//! wait for the user, one at a time, as in a debugger; `/step all` does it
//! for every agent, subagents spawned later included. The status line
//! shows `STEP` while it is on.
//!
//! Each call about to run is held by a question ([`Ask`]) that shows the
//! tool and its arguments and offers Run, Skip, Edit arguments, Run all
//! remaining in this reply, and Leave step mode. The calls of one reply are
//! stepped in the order the model asked for them: a call whose earlier
//! sibling has not settled waits behind a hold of its own, so nothing runs
//! beside the call being stepped. Editing puts the arguments as JSON in the
//! terminal view's input (or `$EDITOR`); the user's next line is checked
//! against the tool's parameters and, if it fits, becomes the call, both
//! the one that runs and the one the saved conversation says the model
//! made. Closing the question stops the turn, as an interrupt does.
//!
//! Step mode is never saved: a resumed or reloaded session starts without
//! it. It needs an interactive front.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::transcript::arguments_refusal;
use rig_tui::{
    RunInTerminal, Side, StatusItem, StatusItems, StatusSystems, TerminalExit, Tone, TuiInput,
};
use serde_json::{Map, Value};

/// What the model reads for a call the user skipped.
pub const SKIPPED: &str = "Skipped by the user.";

/// Why `/step` is refused in a run nobody sits at.
pub const NEEDS_FRONT: &str = "step mode needs an interactive front";

/// The note on a call whose arguments the user edited.
pub const EDITED: &str = "(arguments edited by the user)";

/// The status line's `STEP`, before the agent's status; never dropped.
const STEP_ITEM: StatusItem = StatusItem::at(Side::Left, 45, u8::MAX);

/// Whether every agent steps, those spawned later included (`/step all`).
#[derive(Resource, Default, Debug)]
pub struct StepAll(pub bool);

/// On an agent whose tool calls are stepped (`/step`).
#[derive(Component, Default, Debug)]
pub struct Stepping;

/// The program arguments are edited in, from `$EDITOR`; `None` edits them
/// in the terminal view's input. Inserted from the environment unless the
/// app inserted one before adding the plugin.
#[derive(Resource, Clone, Debug, Default)]
pub struct StepEditor(pub Option<String>);

/// The plugin's stage systems, in Gate before [`CommandSystems`], since it
/// takes the user's line with edited arguments over. A gate that decides
/// tool calls goes `.before` it, so a refused call is never stepped.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepSystems;

/// `/step`, the questions that hold each call, and the `STEP` item.
#[derive(Default)]
pub struct StepModePlugin;

impl Plugin for StepModePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<StepEditor>() {
            let editor = std::env::var("EDITOR").ok();
            app.insert_resource(StepEditor(editor.filter(|it| !it.trim().is_empty())));
        }
        app.init_resource::<StepAll>()
            .add_command(
                "step",
                "Step through the agent's tool calls; /step all for every agent",
                step_command,
            )
            .configure_sets(
                AgentPass,
                StepSystems.in_set(Stage::Gate).before(CommandSystems),
            )
            .add_systems(
                AgentPass,
                (
                    release.in_set(Stage::Review),
                    (take_edits, answered, step).chain().in_set(StepSystems),
                ),
            )
            .add_systems(Update, (edited, release).chain().before(AgentLoop))
            .add_systems(PostUpdate, show_step.in_set(StatusSystems))
            .add_observer(apply_args);
    }
}

/// What holds a call for step mode, on the hold.
#[derive(Component, Clone, Copy, Debug)]
struct StepHold {
    /// The call held.
    call: Entity,
    /// Its agent.
    agent: Entity,
    /// Why.
    kind: HoldKind,
}

/// Why step mode holds a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HoldKind {
    /// An earlier call of its reply has not settled.
    Queued,
    /// The question, on the [`Ask`] itself.
    Asking,
    /// The user edits its arguments.
    Editing,
}

/// On a call the user let through: not stepped again.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Stepped;

/// On a hold to let go of once the stages before Gate run again, so every
/// gate sees the call as it now is.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Released;

/// On a turn: the calls of this reply run without stepping.
#[derive(Component)]
struct RunsFreely(EffectId);

/// `$EDITOR` editing the arguments of the call `hold` holds, in `path`.
#[derive(Component)]
struct InEditor {
    hold: Entity,
    path: PathBuf,
    /// What it started from, for the input when the editor fails.
    text: String,
}

/// The arguments the user typed for the call `hold` holds.
#[derive(Event, Clone, Debug)]
struct ArgsTyped {
    hold: Entity,
    text: String,
}

/// The question's choices: id and label.
const CHOICES: [(&str, &str); 5] = [
    ("run", "Run"),
    ("skip", "Skip"),
    ("edit", "Edit arguments"),
    ("rest", "Run all remaining in this reply"),
    ("leave", "Leave step mode"),
];

/// Which agents step.
#[derive(bevy_ecs::system::SystemParam)]
struct Steppers<'w, 's> {
    all: Res<'w, StepAll>,
    flagged: Query<'w, 's, (), With<Stepping>>,
}

impl Steppers<'_, '_> {
    fn on(&self, agent: Entity) -> bool {
        self.all.0 || self.flagged.contains(agent)
    }
}

/// The tool calls of each reply, in the order the model asked for them.
#[derive(bevy_ecs::system::SystemParam)]
struct Replies<'w, 's> {
    turns: Query<'w, 's, &'static Calls>,
    calls:
        Query<'w, 's, (&'static Effect, Has<Settled>), (With<ToolRequest>, Without<SupersededBy>)>,
    free: Query<'w, 's, &'static RunsFreely>,
}

/// Where a call stands in its reply.
struct Place {
    /// Its position, from 1.
    nth: usize,
    /// How many calls the reply asked for.
    of: usize,
    /// Whether an earlier call of the reply has not settled.
    behind: bool,
    /// Whether the user let the rest of the reply run.
    free: bool,
}

impl Replies<'_, '_> {
    fn place(&self, call: Entity, turn: Option<Entity>) -> Place {
        let parent = self
            .calls
            .get(call)
            .ok()
            .and_then(|(effect, _)| effect.parent);
        let siblings: Vec<(Entity, bool)> = turn
            .and_then(|turn| self.turns.get(turn).ok())
            .into_iter()
            .flat_map(Calls::iter)
            .filter_map(|sibling| {
                let (effect, settled) = self.calls.get(sibling).ok()?;
                (effect.parent == parent).then_some((sibling, settled))
            })
            .collect();
        let at = siblings.iter().position(|(sibling, _)| *sibling == call);
        let earlier = siblings.get(..at.unwrap_or_default()).unwrap_or_default();
        let free = turn
            .and_then(|turn| self.free.get(turn).ok())
            .is_some_and(|free| Some(free.0) == parent);
        Place {
            nth: at.map_or(1, |at| at + 1),
            of: siblings.len().max(1),
            behind: earlier.iter().any(|(_, settled)| !settled),
            free,
        }
    }
}

/// Where edited arguments are typed: `$EDITOR`, else the input.
#[derive(bevy_ecs::system::SystemParam)]
struct Editors<'w, 's> {
    editor: Res<'w, StepEditor>,
    input: Option<ResMut<'w, TuiInput>>,
    paths: Option<Res<'w, SessionPaths>>,
    commands: Commands<'w, 's>,
}

impl Editors<'_, '_> {
    /// Opens `text`, the arguments of the call `hold` holds, for editing.
    fn open(&mut self, hold: Entity, text: &str) {
        let mut words = self.editor.0.iter().flat_map(|it| it.split_whitespace());
        if let Some(program) = words.next() {
            let dir = self
                .paths
                .as_ref()
                .map_or_else(std::env::temp_dir, |paths| paths.0.path().to_path_buf());
            let path = dir.join(format!("step-arguments-{}.json", hold.to_bits()));
            if std::fs::write(&path, text).is_ok() {
                let mut command = Command::new(program);
                command.args(words).arg(&path);
                let text = text.to_owned();
                let run = InEditor { hold, path, text };
                self.commands.spawn((RunInTerminal(command), run));
                return;
            }
        }
        if let Some(input) = &mut self.input {
            input.set(text);
        }
    }
}

/// `/step` and `/step all`.
fn step_command(
    In(args): In<CommandArgs>,
    invoked: Option<Res<Invoked>>,
    mut all: ResMut<StepAll>,
    flagged: Query<Entity, With<Stepping>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if !invoked.is_some_and(|invoked| invoked.interactive()) {
        notices.write(Notice::error(agent, NEEDS_FRONT));
        return;
    }
    let off = |commands: &mut Commands| {
        for stepping in &flagged {
            commands.entity(stepping).remove::<Stepping>();
        }
    };
    match args.args.as_str() {
        "all" if all.0 => {
            all.0 = false;
            off(&mut commands);
            notices.write(Notice::info(agent, "Step mode off for every agent."));
        }
        "all" => {
            all.0 = true;
            notices.write(Notice::info(agent, "Step mode on for every agent."));
        }
        "" if all.0 || flagged.contains(agent) => {
            if all.0 {
                all.0 = false;
                off(&mut commands);
            }
            commands.entity(agent).remove::<Stepping>();
            notices.write(Notice::info(agent, "Step mode off."));
        }
        "" => {
            commands.entity(agent).insert(Stepping);
            notices.write(Notice::info(
                agent,
                "Step mode on: each tool call waits for you.",
            ));
        }
        _ => {
            notices.write(Notice::error(agent, "/step takes nothing, or `all`"));
        }
    }
}

/// Holds each call about to run of an agent that steps: behind the
/// earlier calls of its reply that have not settled, else with the
/// question.
fn step(
    calls: Query<
        (Entity, &ForAgent, Option<&CallOf>, &ToolRequest, &Decision),
        (Undecided, ToolCalls, Without<Stepped>),
    >,
    replies: Replies,
    steppers: Steppers,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent), turn, request, decision) in &calls {
        if decision.is_decided() || !steppers.on(agent) || !answerers.can_answer(Some(agent)) {
            continue;
        }
        let place = replies.place(call, turn.map(|turn| turn.0));
        if place.free {
            continue;
        }
        if place.behind {
            let hold = StepHold {
                call,
                agent,
                kind: HoldKind::Queued,
            };
            let why = Name::new("waiting for the step before it");
            commands.spawn((why, HoldOf(call), hold));
            continue;
        }
        let function = &request.call.function;
        let arguments = Value::Object(function.arguments.clone());
        let title = format!(
            "Step {} of {}: {}",
            place.nth,
            place.of,
            function.name.as_str()
        );
        let body = serde_json::to_string_pretty(&arguments).unwrap_or_default();
        let ask = CHOICES
            .iter()
            .fold(Ask::new(title).body(body), |ask, (id, label)| {
                ask.choice(*id, *label)
            });
        let hold = StepHold {
            call,
            agent,
            kind: HoldKind::Asking,
        };
        commands.spawn((ask, ForAgent(agent), HoldOf(call), hold));
    }
}

/// Carries out the user's answer to a step's question; the question lets
/// go of the call.
fn answered(
    asks: Query<(Entity, &Outcome, &StepHold), (With<Ask>, Added<Outcome>)>,
    mut calls: Query<(&ToolRequest, &Effect, Option<&CallOf>, &mut Decision), ToolCalls>,
    mut all: ResMut<StepAll>,
    mut editors: Editors,
) {
    for (ask, outcome, &StepHold { call, agent, .. }) in &asks {
        editors.commands.entity(ask).remove::<(HoldOf, StepHold)>();
        let Ok((request, effect, turn, mut decision)) = calls.get_mut(call) else {
            continue;
        };
        let choice = match Answer::of(outcome) {
            Some(Answer::Choice(choice)) => choice,
            Some(Answer::Text(_)) => continue,
            // Closed: the user stops the turn, as an interrupt does.
            None => {
                editors.commands.trigger(Interrupt::new(agent));
                continue;
            }
        };
        match choice.as_str() {
            "skip" => {
                decision.decide(Outcome::from(ToolResult::success(SKIPPED.into())));
            }
            "edit" => {
                let hold = StepHold {
                    call,
                    agent,
                    kind: HoldKind::Editing,
                };
                let why = Name::new("editing its arguments");
                let hold = editors.commands.spawn((why, HoldOf(call), hold)).id();
                let arguments = Value::Object(request.call.function.arguments.clone());
                let text = serde_json::to_string_pretty(&arguments).unwrap_or_default();
                editors.open(hold, &text);
            }
            "rest" => {
                editors.commands.entity(call).insert(Stepped);
                if let (Some(&CallOf(turn)), Some(reply)) = (turn, effect.parent) {
                    editors.commands.entity(turn).insert(RunsFreely(reply));
                }
            }
            "leave" => {
                all.0 = false;
                editors.commands.entity(agent).remove::<Stepping>();
            }
            _ => {
                editors.commands.entity(call).insert(Stepped);
            }
        }
    }
}

/// The user's next line to an agent whose call's arguments are being
/// edited is the arguments, not a message; commands still run.
fn take_edits(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
    holds: Query<(Entity, &StepHold), Without<Released>>,
    runs: Query<&InEditor>,
    mut commands: Commands,
) {
    for (incoming, &ForAgent(agent), mut decision) in &mut typed {
        let typed_by_user = incoming.origin.kind == OriginKind::User;
        if !typed_by_user || incoming.command().is_some() || decision.is_decided() {
            continue;
        }
        let editing = holds.iter().find(|(hold, step)| {
            step.agent == agent
                && step.kind == HoldKind::Editing
                && !runs.iter().any(|run| run.hold == *hold)
        });
        if let Some((hold, _)) = editing {
            decision.decide(Outcome::custom("step: edited arguments"));
            let text = incoming.text.clone();
            commands.trigger(ArgsTyped { hold, text });
        }
    }
}

/// What `$EDITOR` left: the arguments typed, or, when it failed, the
/// arguments in the input instead.
fn edited(
    done: Query<(Entity, &TerminalExit, &InEditor), Added<TerminalExit>>,
    holds: Query<&StepHold>,
    mut input: Option<ResMut<TuiInput>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (run, exit, editing) in &done {
        commands.entity(run).despawn();
        let text = std::fs::read_to_string(&editing.path);
        std::fs::remove_file(&editing.path).ok();
        match (exit.success(), text) {
            (true, Ok(text)) => {
                commands.trigger(ArgsTyped {
                    hold: editing.hold,
                    text,
                });
            }
            _ => {
                let agent = holds.get(editing.hold).ok().map(|hold| hold.agent);
                let why = "The editor failed: edit the arguments in the input and send them.";
                notices.write(Notice::error(agent, why));
                if let Some(input) = &mut input {
                    input.set(editing.text.as_str());
                }
            }
        }
    }
}

/// Makes the arguments the user typed the call's, if they fit its tool:
/// the call runs with them, and the conversation says the model asked for
/// them. Otherwise says why and opens them for editing again.
fn apply_args(
    typed: On<ArgsTyped>,
    holds: Query<&StepHold>,
    mut calls: Query<&mut ToolRequest>,
    tools: Query<&ToolDef>,
    mut agents: Query<&mut Conversation>,
    mut commit: Commit,
    mut editors: Editors,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&StepHold { call, agent, .. }) = holds.get(typed.hold) else {
        return;
    };
    let Ok(mut request) = calls.get_mut(call) else {
        return;
    };
    let name = request.call.function.name.as_str().to_owned();
    let tool = tools.iter().find(|def| def.0.name.as_str() == name);
    let parameters = tool.map(|def| &def.0.parameters);
    let arguments = match fitting(&typed.text, parameters, &request.call) {
        Ok(arguments) => arguments,
        Err(why) => {
            let why = format!("The arguments do not fit `{name}`: {why}. Edit them again.");
            notices.write(Notice::error(agent, why));
            editors.open(typed.hold, &typed.text);
            return;
        }
    };
    request.call.function.arguments = arguments;
    request.call.function.invalid_arguments = None;
    if let Ok(mut conversation) = agents.get_mut(agent) {
        record(agent, &mut conversation, &request.call, &mut commit);
    }
    notices.write(Notice::info(agent, format!("`{name}` {EDITED}")));
    editors.commands.entity(call).insert(Stepped);
    editors.commands.entity(typed.hold).insert(Released);
}

/// Puts `call`, edited, in place of the call of that id in the model's
/// message that asked for it, so the saved conversation has it.
fn record(agent: Entity, conversation: &mut Conversation, call: &ToolCall, commit: &mut Commit) {
    let asked = |message: &Message| {
        matches!(message, Message::Assistant(reply)
            if reply.tool_calls().any(|asked| asked.id == call.id))
    };
    let Some(at) = conversation.messages().iter().rposition(asked) else {
        return;
    };
    let Some(mut entry) = conversation.entry(at) else {
        return;
    };
    if let Message::Assistant(reply) = &mut entry.message {
        for content in &mut reply.content {
            if let AssistantContent::ToolCall(asked) = content
                && asked.id == call.id
            {
                asked.function = call.function.clone();
            }
        }
    }
    commit.splice(agent, conversation, at..=at, [entry]);
}

/// The arguments `text` holds, if they are a JSON object that fits the
/// tool's `parameters`: no argument it does not declare, every one it
/// requires, and each of the type it says.
fn fitting(
    text: &str,
    parameters: Option<&Value>,
    call: &ToolCall,
) -> Result<Map<String, Value>, String> {
    let value: Value =
        serde_json::from_str(text.trim()).map_err(|error| format!("not JSON ({error})"))?;
    let Value::Object(arguments) = value else {
        return Err("not a JSON object".to_owned());
    };
    let Some(schema) = parameters else {
        return Ok(arguments);
    };
    let mut candidate = call.clone();
    candidate.function.arguments = arguments.clone();
    candidate.function.invalid_arguments = None;
    if let Some(why) = arguments_refusal(schema, &candidate) {
        return Err(why);
    }
    let required = schema.get("required").and_then(Value::as_array);
    let missing = required
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|name| !arguments.contains_key(*name));
    if let Some(missing) = missing {
        return Err(format!("`{missing}` is required"));
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    let wrong = arguments.iter().find_map(|(name, value)| {
        let types = properties?.get(name)?.get("type")?;
        let fits = match types {
            Value::String(kind) => is(kind, value),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|it| is(it, value)),
            _ => true,
        };
        (!fits).then(|| format!("`{name}` must be of type {types}"))
    });
    wrong.map_or(Ok(arguments), Err)
}

/// Whether `value` is of the JSON Schema type `kind`.
fn is(kind: &str, value: &Value) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Lets go of the holds step mode no longer needs: those of agents that
/// stopped stepping, a queued call once the calls before it settled or the
/// rest of its reply runs freely, and edited calls.
fn release(
    holds: Query<(Entity, &StepHold, Has<Released>)>,
    calls: Query<Option<&CallOf>>,
    replies: Replies,
    steppers: Steppers,
    mut commands: Commands,
) {
    for (hold, step, released) in &holds {
        let queued = || {
            let turn = calls.get(step.call).ok().flatten().map(|turn| turn.0);
            let place = replies.place(step.call, turn);
            place.free || !place.behind
        };
        let done = match step.kind {
            HoldKind::Queued => queued(),
            HoldKind::Asking | HoldKind::Editing => false,
        };
        if released || done || !steppers.on(step.agent) {
            commands.entity(hold).try_despawn();
        }
    }
}

/// `STEP` on the status line of each agent that steps.
fn show_step(
    steppers: Steppers,
    mut agents: Query<(Entity, &mut StatusItems), With<Agent>>,
    mut shown: Local<HashSet<Entity>>,
) {
    for (agent, mut items) in &mut agents {
        let on = steppers.on(agent);
        if on == shown.contains(&agent) {
            continue;
        }
        if on {
            shown.insert(agent);
            items.show(STEP_ITEM.says("STEP", Tone::Yellow));
        } else {
            shown.remove(&agent);
            items.show(STEP_ITEM);
        }
    }
}

#[cfg(test)]
mod tests;
