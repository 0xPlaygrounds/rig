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
//! beside the call being stepped. Editing asks for the arguments as JSON
//! to edit in the input (and in `$VISUAL` or `$EDITOR`, if set); they are
//! validated against the tool's schema and, if they fit, become the call,
//! both the one that runs and the one the saved conversation says the
//! model made, with a note in the transcript. Closing the question stops
//! the turn, as an interrupt does.
//!
//! Step mode is never saved: a resumed or reloaded session starts without
//! it. It needs a front that answers questions.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::tool::validate::validate;
use rig_tui::{RunInTerminal, Side, StatusItem, StatusItems, StatusSystems, TerminalExit, Tone};
use serde_json::{Map, Value};

/// What the model reads for a call the user skipped.
pub const SKIPPED: &str = "Skipped by the user.";

/// Why `/step` is refused in a run nobody answers questions in.
pub const NEEDS_FRONT: &str = "step mode needs an interactive front";

/// The note on a call whose arguments the user edited.
pub const EDITED: &str = "(arguments edited by the user)";

/// The origin of the transcript's notes.
const ORIGIN: &str = "fh-step-mode";

/// The status line's `STEP`, before the agent's status; never dropped.
const STEP_ITEM: StatusItem = StatusItem::new("fh-step-mode", Side::Left, 45, u8::MAX);

/// Whether every agent steps, those spawned later included (`/step all`).
#[derive(Resource, Default, Debug)]
pub struct StepAll(pub bool);

/// On an agent whose tool calls are stepped (`/step`).
#[derive(Component, Default, Debug)]
pub struct Stepping;

/// The editor from `$VISUAL` or `$EDITOR`; `None` edits arguments
/// in the input only. Inserted from the environment unless the app
/// inserted one before adding the plugin.
#[derive(Resource, Clone, Debug, Default)]
pub struct StepEditor(pub Option<String>);

/// The plugin's stage systems. A gate that decides tool calls goes
/// `.before(StepSystems::Gate)`, so a refused call is never stepped.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepSystems {
    /// In [`Stage::Gate`]: holds each call of a stepping agent.
    Gate,
    /// In [`ReviewSystems::Edit`]: carries out the answers and lets go of the
    /// calls, so the next pass's gates see an edited call as it now is.
    Review,
}

/// `/step`, the questions that hold each call, and the `STEP` item.
#[derive(Default)]
pub struct StepModePlugin;

impl Plugin for StepModePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<StepEditor>() {
            app.insert_resource(StepEditor(corpus_shared::editor::line()));
        }
        app.init_resource::<StepAll>()
            .add_command(
                "step",
                "Step through the agent's tool calls; /step all for every agent",
                step_command,
            )
            .configure_sets(AgentPass, StepSystems::Gate.in_set(Stage::Gate))
            .configure_sets(AgentPass, StepSystems::Review.in_set(ReviewSystems::Edit))
            .add_systems(
                AgentPass,
                (
                    step.in_set(StepSystems::Gate),
                    (answered, release).chain().in_set(StepSystems::Review),
                ),
            )
            .add_systems(Update, edited.before(AgentLoop))
            .add_systems(PostUpdate, show_step.in_set(StatusSystems::Show));
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
    /// The step's question, on the [`Ask`] itself.
    Asking,
    /// The question for its edited arguments, on the [`Ask`] itself.
    Editing,
}

/// On a call the user let through: not stepped again.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Stepped;

/// On a turn: the calls of this reply run without stepping.
#[derive(Component)]
struct RunsFreely(EffectId);

/// The editor answering the question `ask` with the file at `path`.
#[derive(Component)]
struct InEditor {
    ask: Entity,
    path: PathBuf,
}

/// A choice of the step question.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StepChoice {
    Run,
    Skip,
    Edit,
    Rest,
    Leave,
}

impl StepChoice {
    /// Every choice, in the order the question offers them.
    const ALL: [Self; 5] = [Self::Run, Self::Skip, Self::Edit, Self::Rest, Self::Leave];

    /// Its id in the question's answer.
    fn id(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Skip => "skip",
            Self::Edit => "edit",
            Self::Rest => "rest",
            Self::Leave => "leave",
        }
    }

    /// Its label in the question.
    fn label(self) -> &'static str {
        match self {
            Self::Run => "Run",
            Self::Skip => "Skip",
            Self::Edit => "Edit arguments",
            Self::Rest => "Run all remaining in this reply",
            Self::Leave => "Leave step mode",
        }
    }

    /// The choice whose id is `id`.
    fn of(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|choice| choice.id() == id)
    }
}

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
    calls: Query<
        'w,
        's,
        (&'static Effect, Option<&'static CallOf>, Has<Settled>),
        (With<ToolRequest>, Without<SupersededBy>),
    >,
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
    fn place(&self, call: Entity) -> Place {
        let (parent, turn) = self
            .calls
            .get(call)
            .ok()
            .map_or((None, None), |(effect, turn, _)| {
                (effect.parent, turn.map(|turn| turn.0))
            });
        let siblings: Vec<(Entity, bool)> = turn
            .and_then(|turn| self.turns.get(turn).ok())
            .into_iter()
            .flat_map(Calls::iter)
            .filter_map(|sibling| {
                let (effect, _, settled) = self.calls.get(sibling).ok()?;
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

/// Asks for a call's edited arguments: in the input, and in the
/// [`StepEditor`], if there is one.
#[derive(bevy_ecs::system::SystemParam)]
struct Edits<'w, 's> {
    editor: Res<'w, StepEditor>,
    scratch: Option<Res<'w, Scratch>>,
    commands: Commands<'w, 's>,
}

impl Edits<'_, '_> {
    /// Holds the call with a question for its arguments, starting from
    /// `text`, saying `why` the last ones did not fit, if they did not.
    fn ask(&mut self, hold: StepHold, tool: &str, text: &str, why: &str) {
        let ask = Ask::new(format!("Edit the arguments of `{tool}`"))
            .body(why)
            .initial(text)
            .multiline();
        let hold = StepHold {
            kind: HoldKind::Editing,
            ..hold
        };
        let ask = (self.commands)
            .spawn((ask, ForAgent(hold.agent), HoldOf(hold.call), hold))
            .id();
        let mut words = self.editor.0.iter().flat_map(|it| it.split_whitespace());
        let dir = self.scratch.as_ref().and_then(|scratch| scratch.0.as_ref());
        let (Some(program), Some(dir)) = (words.next(), dir) else {
            return;
        };
        let path = dir.join(format!("step-arguments-{}.json", ask.to_bits()));
        if std::fs::write(&path, text).is_ok() {
            let mut command = Command::new(program);
            command.args(words).arg(&path);
            (self.commands).spawn((RunInTerminal(command), InEditor { ask, path }));
        }
    }
}

/// `/step` and `/step all`.
fn step_command(
    In(args): In<CommandArgs>,
    answerers: Answerers,
    mut all: ResMut<StepAll>,
    flagged: Query<Entity, With<Stepping>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::error(agent, NEEDS_FRONT));
        return;
    }
    let off = |commands: &mut Commands| {
        for stepping in &flagged {
            commands.entity(stepping).remove::<Stepping>();
        }
    };
    let said = match args.args.as_str() {
        "all" if all.0 => {
            all.0 = false;
            off(&mut commands);
            "Step mode off for every agent."
        }
        "all" => {
            all.0 = true;
            "Step mode on for every agent."
        }
        "" if all.0 || flagged.contains(agent) => {
            if all.0 {
                all.0 = false;
                off(&mut commands);
            }
            commands.entity(agent).remove::<Stepping>();
            "Step mode off."
        }
        "" => {
            commands.entity(agent).insert(Stepping);
            "Step mode on: each tool call waits for you."
        }
        _ => {
            notices.write(Notice::error(agent, "/step takes nothing, or `all`"));
            return;
        }
    };
    notices.write(Notice::info(agent, said));
}

/// Holds each call about to run of an agent that steps: behind the
/// earlier calls of its reply that have not settled, else with the
/// question.
fn step(
    calls: Query<
        (Entity, &ForAgent, &ToolRequest, &Decision),
        (ReadyToStart, ToolCalls, Without<Stepped>),
    >,
    replies: Replies,
    steppers: Steppers,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent), request, decision) in &calls {
        if decision.is_decided() || !steppers.on(agent) || !answerers.can_answer(Some(agent)) {
            continue;
        }
        let place = replies.place(call);
        if place.free {
            continue;
        }
        let kind = if place.behind {
            HoldKind::Queued
        } else {
            HoldKind::Asking
        };
        let hold = (HoldOf(call), StepHold { call, agent, kind });
        if place.behind {
            commands.spawn((Name::new("waiting for the step before it"), hold));
            continue;
        }
        let function = &request.call.function;
        let title = format!("Step {} of {}: {}", place.nth, place.of, function.name);
        let ask = StepChoice::ALL.into_iter().fold(
            Ask::new(title).body(pretty(&function.arguments)),
            |ask, choice| ask.choice(choice.id(), choice.label()),
        );
        commands.spawn((ask, ForAgent(agent), hold));
    }
}

/// Carries out the user's answers to the step's questions; each question
/// lets go of its call.
fn answered(
    asks: Query<(Entity, &Outcome, &StepHold), (With<Ask>, Added<Outcome>)>,
    mut calls: Query<(&mut ToolRequest, &Effect, Option<&CallOf>, &mut Decision), ToolCalls>,
    tools: Query<&ToolDef>,
    mut agents: Query<&mut Conversation>,
    mut all: ResMut<StepAll>,
    mut edits: Edits,
) {
    for (ask, outcome, &hold) in &asks {
        let StepHold { call, agent, kind } = hold;
        edits.commands.entity(ask).remove::<(HoldOf, StepHold)>();
        let Ok((mut request, effect, turn, mut decision)) = calls.get_mut(call) else {
            continue;
        };
        let tool = request.call.function.name.as_str().to_owned();
        let choice = match (kind, Answer::of(outcome)) {
            (
                HoldKind::Editing,
                Some(Answer {
                    text: Some(text), ..
                }),
            ) => {
                let schema = tools.iter().find(|def| def.0.name.as_str() == tool);
                match fitting(&text, schema.map(|def| &def.0.parameters)) {
                    Ok(arguments) => {
                        request.call.function.arguments = arguments;
                        request.call.function.invalid_arguments = None;
                        if let Ok(mut conversation) = agents.get_mut(agent) {
                            let after = record(&mut conversation, &request.call);
                            let title = format!("`{tool}` {EDITED}");
                            let text = pretty(&request.call.function.arguments);
                            let note = Note::new(title, text).aside(Author::plugin(ORIGIN), after);
                            conversation.add_aside(note);
                        }
                        edits.commands.entity(call).insert(Stepped);
                    }
                    Err(why) => {
                        let why = format!("These do not fit `{tool}`: {why}.");
                        edits.ask(hold, &tool, &text, &why);
                    }
                }
                continue;
            }
            // Closed: the step's question comes back.
            (HoldKind::Editing, _) => continue,
            (_, Some(answer)) => match answer.choices.into_iter().next() {
                Some(choice) => choice,
                None => continue,
            },
            // Closed: the user stops the turn, as an interrupt does.
            (_, None) => {
                edits.commands.trigger(StopTurn::new(agent));
                continue;
            }
        };
        match StepChoice::of(&choice) {
            Some(StepChoice::Skip) => {
                decision.decide(
                    Decider::plugin(ORIGIN),
                    Outcome::from(ToolResult::success(SKIPPED.into())),
                );
            }
            Some(StepChoice::Edit) => {
                let text = pretty(&request.call.function.arguments);
                edits.ask(hold, &tool, &text, "");
            }
            Some(StepChoice::Rest) => {
                edits.commands.entity(call).insert(Stepped);
                if let (Some(&CallOf(turn)), Some(reply)) = (turn, effect.parent) {
                    edits.commands.entity(turn).insert(RunsFreely(reply));
                }
            }
            Some(StepChoice::Leave) => {
                all.0 = false;
                edits.commands.entity(agent).remove::<Stepping>();
            }
            Some(StepChoice::Run) | None => {
                edits.commands.entity(call).insert(Stepped);
            }
        }
    }
}

/// Answers the question the editor edited the arguments for with what it
/// left, once it exited well; otherwise the question waits in the input.
fn edited(
    done: Query<(Entity, &TerminalExit, &InEditor), Added<TerminalExit>>,
    holds: Query<&StepHold>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (run, exit, editing) in &done {
        commands.entity(run).despawn();
        let text = std::fs::read_to_string(&editing.path);
        std::fs::remove_file(&editing.path).ok();
        match (exit.success(), text) {
            (true, Ok(text)) => {
                commands.trigger(Pick::person(editing.ask, Some(Chosen::text(text))));
            }
            _ => {
                let agent = holds.get(editing.ask).ok().map(|hold| hold.agent);
                let why = "The editor failed: edit the arguments in the input.";
                notices.write(Notice::error(agent, why));
            }
        }
    }
}

/// `arguments` as pretty JSON.
fn pretty(arguments: &Map<String, Value>) -> String {
    serde_json::to_string_pretty(arguments).unwrap_or_default()
}

/// The arguments `text` holds, if it is a JSON object that fits the tool's
/// `schema`; otherwise why not.
fn fitting(text: &str, schema: Option<&Value>) -> Result<Map<String, Value>, String> {
    let value: Value =
        serde_json::from_str(text.trim()).map_err(|error| format!("not JSON ({error})"))?;
    if let Some(Err(problems)) = schema.map(|schema| validate(schema, &value)) {
        let problems: Vec<String> = problems.iter().map(ToString::to_string).collect();
        return Err(problems.join("; "));
    }
    match value {
        Value::Object(arguments) => Ok(arguments),
        _ => Err("not a JSON object".to_owned()),
    }
}

/// Puts `call`, edited, in place of the call of that id in the model's
/// message that asked for it, so the saved conversation has it; returns
/// that message's id.
fn record(conversation: &mut Conversation, call: &ToolCall) -> Option<MessageId> {
    let asked = |message: &Message| {
        matches!(message, Message::Assistant(reply)
            if reply.tool_calls().any(|asked| asked.id == call.id))
    };
    let at = conversation
        .messages()
        .iter()
        .rposition(|message| asked(message))?;
    let mut entry = conversation.entry(at)?;
    if let Message::Assistant(reply) = std::sync::Arc::make_mut(&mut entry.message) {
        for content in &mut reply.content {
            if let AssistantContent::ToolCall(asked) = content
                && asked.id == call.id
            {
                asked.function = call.function.clone();
            }
        }
    }
    conversation.splice(at..=at, [entry], Author::plugin(ORIGIN));
    conversation.id(at)
}

/// Lets go of the queued holds step mode no longer needs: those of agents
/// that stopped stepping (the questions too, so their calls run), and a
/// queued call once the calls before it settled or the rest of its reply
/// runs freely.
fn release(
    holds: Query<(Entity, &StepHold)>,
    replies: Replies,
    steppers: Steppers,
    mut commands: Commands,
) {
    for (hold, step) in &holds {
        let queued = step.kind == HoldKind::Queued && {
            let place = replies.place(step.call);
            place.free || !place.behind
        };
        if queued || !steppers.on(step.agent) {
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
