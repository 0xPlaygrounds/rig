//! Second opinions from other models: the `consult` tool and `/consult`.
//!
//! The `[consult]` settings table ([`ConsultConfig`]) names two or three
//! consultant models, a step limit per consultant and a timeout. A
//! `consult` call (`question`, optional `paths`) starts one read-only helper
//! agent per model, all at once, through `rig_basics::subagents`' [`SpawnSubagent`]:
//! each gets the question, the listed files and a short summary of the
//! asking agent's task, and only the asking agent's tools that only read.
//! The call's result lists each answer under its model, with the steps it
//! took and what it cost; a consultant that fails, times out or hits the
//! step limit shows as `model: failed (why)` while the others still answer.
//!
//! The consultants' requests are effects under the `consult` call, so the
//! session's effect log keeps their work under it. With the default `tui`
//! feature the call's render system draws each consultant's model and tool
//! calls nested under it ([`ToolCallView::children`](rig_tui::ToolCallView::children)),
//! live and after a resume. Their model calls are ordinary model calls, so
//! they count toward the session's spending per model (`rig_basics::telemetry`),
//! which also gives each consultant's cost.
//! Interrupting the asking agent, or the call ending any other way, retires
//! every consultant still at work, briefed or not.
//!
//! `/consult <question>` asks the same for the user: the answers show in a
//! picker, whose **Send to agent** adds them to the agent's conversation as
//! a note.

use std::path::PathBuf;
use std::sync::Arc;

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_basics::telemetry::usage::AgentUsage;
use rig_core::completion::UsageTotals;
use rig_core::effect::EffectId;
use rig_harness::prelude::*;
use rig_tools::workspace::Workspace;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool's name.
pub const CONSULT: &str = "consult";

/// The most consultants one consult asks.
pub const MAX_CONSULTANTS: usize = 3;

const DESCRIPTION: &str = "Ask two or three other models the same question and get their \
    answers side by side. Each consultant is a read-only agent on its own model: it sees the \
    question, the files in `paths` and a short summary of your task, and may read and search \
    the project, but changes nothing. Returns each model's answer, or why it failed. Use it \
    for a second opinion when you are unsure about a design choice or a tricky bug: ask one \
    self-contained question and list the files that matter in `paths`. Weigh the answers \
    yourself; they may disagree.";

const CONSULTANT_PROMPT: &str = "You are a consultant: another agent asks for your opinion on \
    one question. You may read and search the project with your tools, and change nothing. \
    Answer in a few short paragraphs: your recommendation first, then why, naming the files \
    and lines it rests on. Nobody answers questions while you work; say what you assumed.";

/// The `[consult]` settings table.
///
/// ```toml
/// [consult]
/// models = ["openai/gpt-5.1", "anthropic/claude-sonnet-4-5"]
/// max_steps = 8
/// timeout = 120
/// ```
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsultConfig {
    /// The consultants' catalog models (`vendor/model`), two or three.
    pub models: Vec<String>,
    /// The model calls each consultant may make; 8 when not set.
    pub max_steps: u32,
    /// The seconds a consult waits for its consultants; 120 when not set.
    pub timeout: u64,
}

impl Default for ConsultConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            max_steps: 8,
            timeout: 120,
        }
    }
}

/// The plugin's stage systems: the Gate system that counts consultants'
/// steps and refuses the model calls past their limit or their consult,
/// and the Review system that keeps a consultant's failed model call from
/// being retried.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConsultSystems;

/// Adds the `consult` tool and `/consult`; see the [crate](self) docs.
#[derive(Default)]
pub struct ConsultPlugin;

impl Plugin for ConsultPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.add_setting::<ConsultConfig>("consult")
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_open_tool(CONSULT, DESCRIPTION, options, on_consult)
            .add_command(
                "consult",
                "Ask the consultant models of [consult] a question",
                consult_command,
            )
            .configure_sets(AgentPass, ConsultSystems.in_set(Stage::Gate))
            .add_systems(
                AgentPass,
                (
                    count_steps.in_set(ConsultSystems),
                    no_retries.in_set(ReviewSystems::Edit),
                ),
            )
            .add_observer(follow_successor::<SlotRequest>)
            .add_observer(start_consultants)
            .add_observer(link_agent)
            .add_observer(request_settled)
            .add_observer(turn_settled)
            .add_observer(timed_out)
            .add_observer(finish)
            .add_observer(stop_consultants)
            .add_observer(cancel_on_interrupt);
        #[cfg(feature = "tui")]
        app.add_systems(
            rig_tui::TranscriptRender,
            render::draw.in_set(rig_tui::RenderSystems::Render),
        );
    }
}

/// One consultant of a [`Consultation`].
#[derive(Clone, Debug, Default)]
pub struct Slot {
    /// Its catalog model.
    pub model: String,
    /// Its agent, once spawned.
    pub agent: Option<Entity>,
    /// Whether its agent is still at work: spawned, and its turn not
    /// ended (a once-helper retires itself when it ends).
    pub running: bool,
    /// The model calls it made.
    pub steps: u32,
    /// What its model calls used and cost.
    pub spent: UsageTotals,
    /// Its answer, or why it has none; `None` while it works.
    pub result: Option<Result<String, String>>,
}

/// A consult under way, on an entity of its own: a child of its `consult`
/// call, so the call's end ends it. Despawning it stops every consultant
/// still at work.
#[derive(Component, Debug)]
pub struct Consultation {
    /// The agent that asked.
    pub asker: Entity,
    /// The question.
    pub question: String,
    /// The `consult` call it answers; `None` for `/consult`.
    pub call: Option<Entity>,
    /// The consultants.
    pub slots: Vec<Slot>,
    /// The read-only tools the consultants get.
    tools: Vec<String>,
    /// The call's effect, which the consultants' requests are recorded
    /// under.
    effect: Option<EffectId>,
}

impl Consultation {
    /// Ends `slot` with `result` and what its agent spent, unless it ended
    /// already; whether it did.
    fn end(
        &mut self,
        slot: usize,
        result: Result<String, String>,
        spending: &Query<&AgentUsage>,
    ) -> bool {
        let Some(slot) = self
            .slots
            .get_mut(slot)
            .filter(|slot| slot.result.is_none())
        else {
            return false;
        };
        let spent = slot.agent.and_then(|agent| spending.get(agent).ok());
        slot.spent = spent.map(|spent| spent.total).unwrap_or_default();
        slot.result = Some(result);
        true
    }

    /// The answers side by side.
    fn report(&self) -> String {
        let mut text = format!("Second opinions on: {}\n", self.question);
        for slot in &self.slots {
            let steps = match slot.steps {
                1 => "1 step".to_owned(),
                steps => format!("{steps} steps"),
            };
            let spent = slot.spent.cost_or_tokens();
            match &slot.result {
                Some(Ok(answer)) => text.push_str(&format!(
                    "\n## {} ({steps}, {spent})\n{}\n",
                    slot.model,
                    answer.trim()
                )),
                Some(Err(why)) => text.push_str(&format!(
                    "\n## {}: failed ({why}) ({steps}, {spent})\n",
                    slot.model
                )),
                None => text.push_str(&format!("\n## {}: failed (no answer)\n", slot.model)),
            }
        }
        text
    }
}

/// On a consultant's [`SpawnSubagent`] request: its consult and slot.
#[derive(Component, Clone, Copy, Debug)]
struct SlotRequest {
    consultation: Entity,
    slot: usize,
}

/// On a consultant agent while its consult runs: its consult and slot.
#[derive(Component, Clone, Copy, Debug)]
struct SlotOf {
    consultation: Entity,
    slot: usize,
}

/// On a consultation: the consultants' first message, the question, the
/// files and the summary, once the files were read.
#[derive(Component, Debug)]
struct Briefing(String);

impl From<String> for Briefing {
    fn from(text: String) -> Self {
        Self(text)
    }
}

/// On a counted model call of a consultant.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Counted;

/// Asks a consultation whether every consultant ended.
#[derive(EntityEvent, Clone, Copy, Debug)]
struct CheckDone {
    entity: Entity,
}

/// The arguments of a `consult` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ConsultArgs {
    /// The question, self-contained: the consultants know nothing else of
    /// your work but a short summary of your task.
    question: String,
    /// Files the consultants get in full up front, relative to your
    /// working directory.
    #[serde(default)]
    paths: Vec<String>,
}

/// What a consult is asked from: the asking agent's model, tools,
/// conversation and directory.
type Asker<'a> = (&'a ToolAccess, &'a Conversation, Option<&'a WorkDir>);

/// What starting a consult reads.
#[derive(bevy_ecs::system::SystemParam)]
struct Start<'w, 's> {
    askers: Query<'w, 's, Asker<'static>>,
    tools: Query<'w, 's, (&'static ToolDef, &'static Footprint)>,
    config: Res<'w, ConsultConfig>,
    root: Res<'w, ProjectRoot>,
    workspace: Res<'w, ProjectWorkspace>,
    wake: Res<'w, Wake>,
}

/// Where a consult goes once it is started.
struct Asked {
    asker: Entity,
    question: String,
    paths: Vec<String>,
    call: Option<(Entity, EffectId)>,
}

/// Starts a consult: reads its files off the main thread, and arms its
/// timeout. Returns why it cannot start.
fn start(start: &Start, asked: Asked, commands: &mut Commands) -> Result<(), String> {
    let config = &start.config;
    if config.models.is_empty() || config.models.len() > MAX_CONSULTANTS {
        return Err(format!(
            "consult needs one to {MAX_CONSULTANTS} models in `models` of [consult] in the \
             settings; it has {}",
            config.models.len()
        ));
    }
    if asked.question.trim().is_empty() {
        return Err("consult needs a question".to_owned());
    }
    let (access, conversation, work_dir) = start
        .askers
        .get(asked.asker)
        .map_err(|_| "only an agent can consult".to_owned())?;
    let tools: Vec<String> = start
        .tools
        .iter()
        .filter(|(def, footprint)| footprint.0.reads_only() && access.keeps(def.0.name.as_str()))
        .map(|(def, _)| def.0.name.as_str().to_owned())
        .collect();
    let base = work_dir.map_or(&start.root.0, |dir| &dir.0).clone();
    let head = format!(
        "Question: {}\n\nWhat the asking agent works on: {}",
        asked.question.trim(),
        summary(conversation)
    );
    let slots = config.models.iter().map(|model| Slot {
        model: model.clone(),
        ..Slot::default()
    });
    let consultation = Consultation {
        asker: asked.asker,
        question: asked.question.trim().to_owned(),
        call: asked.call.map(|(call, _)| call),
        slots: slots.collect(),
        tools,
        effect: asked.call.map(|(_, effect)| effect),
    };
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let read = briefing(start.workspace.0.clone(), base, asked.paths, head);
    let mut entity = commands.spawn((
        Name::new("consult"),
        consultation,
        WakeAt::after(Duration::from_secs(config.timeout)),
        Running::spawn_into::<Briefing, String>(pool, &start.wake, read),
    ));
    if let Some((call, _)) = asked.call {
        entity.insert(ChildOf(call));
    }
    Ok(())
}

/// What the asking agent works on: the first and the latest thing the user
/// typed to it.
fn summary(conversation: &Conversation) -> String {
    let short = |text: String| {
        let text = text.trim().to_owned();
        match text.char_indices().nth(400) {
            Some((at, _)) => format!("{}…", text.get(..at).unwrap_or_default()),
            None => text,
        }
    };
    let first = first_typed(conversation);
    let latest = (0..conversation.messages().len())
        .rev()
        .find_map(|at| typed(conversation, at));
    match (first, latest) {
        (Some(first), Some(latest)) if first != latest => {
            format!("{}\nIts latest request: {}", short(first), short(latest))
        }
        (Some(first), _) => short(first),
        _ => "(nothing typed yet)".to_owned(),
    }
}

/// The consultants' brief: `head`, then each of `paths` read through the
/// workspace from `base`, or why it could not be.
async fn briefing(
    workspace: Arc<dyn Workspace>,
    base: PathBuf,
    paths: Vec<String>,
    head: String,
) -> String {
    let mut text = head;
    for path in paths {
        let (workspace, base, named) = (workspace.clone(), base.clone(), path.clone());
        let read = blocking(move || {
            let place = workspace.place(Some(&base), true, &named)?;
            workspace.read_text(&place)
        })
        .await;
        match read {
            Ok(content) => text.push_str(&format!("\n\n=== {path} ===\n{content}")),
            Err(error) => text.push_str(&format!("\n\n=== {path} ===\n(not read: {error})")),
        }
    }
    text
}

/// A `consult` call: starts its consult, or refuses it.
fn on_consult(
    called: On<ToolCalled<ConsultArgs>>,
    starting: Start,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let asked = Asked {
        asker: called.agent,
        question: called.args.question.clone(),
        paths: called.args.paths.clone(),
        call: Some((called.call, called.effect)),
    };
    if let Err(why) = start(&starting, asked, &mut commands)
        && let Ok(mut decision) = calls.get_mut(called.call)
    {
        let refused = ToolResult::failed(ToolExecutionError::other(why));
        decision.decide(Decider::plugin(CONSULT), Outcome::from(refused));
    }
}

/// `/consult <question>`.
fn consult_command(
    In(args): In<CommandArgs>,
    starting: Start,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let asked = Asked {
        asker: args.agent,
        question: args.args.clone(),
        paths: Vec::new(),
        call: None,
    };
    match start(&starting, asked, &mut commands) {
        Ok(()) => notices.write(Notice::info(args.agent, "Consulting…")),
        Err(why) => notices.write(Notice::error(args.agent, format!("/consult: {why}"))),
    };
}

/// Asks for each consultant once the brief is ready.
fn start_consultants(
    briefed: On<Add<Briefing>>,
    consultations: Query<(&Consultation, &Briefing)>,
    me: Option<Res<PluginEntity<ConsultPlugin>>>,
    mut commands: Commands,
) {
    let consultation = briefed.entity;
    let Ok((consult, Briefing(brief))) = consultations.get(consultation) else {
        return;
    };
    let (tools, effect) = (consult.tools.clone(), consult.effect);
    let asker = consult.asker;
    for (slot, seat) in consult.slots.iter().enumerate() {
        let request = SpawnSubagent {
            model: Some(seat.model.clone()),
            tools: Some(tools.clone()),
            ..SpawnSubagent::helper(format!("consult {}", seat.model))
                .system_prompt(CONSULTANT_PROMPT)
                .brief(brief.clone())
                .once()
        };
        let mut spawned = commands.spawn((
            request,
            ForAgent(asker),
            Effect::with_parent_id(effect),
            SlotRequest { consultation, slot },
            ChildOf(consultation),
        ));
        if let Some(me) = &me {
            spawned.insert(ProvidedBy(me.entity()));
        }
    }
}

/// Marks each consultant agent as it is spawned.
fn link_agent(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &SlotRequest)>,
    mut consultations: Query<&mut Consultation>,
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(agent), &SlotRequest { consultation, slot })) =
        requests.get(spawned.entity)
    else {
        return;
    };
    let Ok(mut consult) = consultations.get_mut(consultation) else {
        return;
    };
    let Some(seat) = consult.slots.get_mut(slot) else {
        return;
    };
    seat.agent = Some(agent);
    seat.running = true;
    commands.entity(agent).insert(SlotOf { consultation, slot });
}

/// A consultant whose request was refused has failed.
fn request_settled(
    settled: On<Add<Settled>>,
    requests: Query<(&SlotRequest, &Outcome, Has<SpawnedAgent>)>,
    mut consultations: Query<&mut Consultation>,
    spending: Query<&AgentUsage>,
    mut commands: Commands,
) {
    let Ok((&SlotRequest { consultation, slot }, outcome, spawned)) = requests.get(settled.entity)
    else {
        return;
    };
    let why = match &outcome.0 {
        Err(report) => report.message.clone(),
        Ok(_) if !spawned => "no agent was spawned".to_owned(),
        Ok(_) => return,
    };
    if let Ok(mut consult) = consultations.get_mut(consultation)
        && consult.end(slot, Err(why), &spending)
    {
        commands.trigger(CheckDone {
            entity: consultation,
        });
    }
}

/// Counts each model call of a consultant as it is about to start, and
/// refuses the one past the step limit.
fn count_steps(
    mut calls: Query<
        (Entity, &ForAgent, &mut Decision),
        (ReadyToStart, ModelCalls, With<TurnCall>, Without<Counted>),
    >,
    agents: Query<&SlotOf>,
    mut consultations: Query<&mut Consultation>,
    spending: Query<&AgentUsage>,
    config: Res<ConsultConfig>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent), mut decision) in &mut calls {
        let Ok(&SlotOf { consultation, slot }) = agents.get(agent) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        commands.entity(call).insert(Counted);
        let Ok(mut consult) = consultations.get_mut(consultation) else {
            continue;
        };
        let Some(seat) = consult.slots.get_mut(slot) else {
            continue;
        };
        if seat.steps < config.max_steps {
            seat.steps += 1;
            continue;
        }
        consult.end(slot, Err("step limit".to_owned()), &spending);
        commands.trigger(CheckDone {
            entity: consultation,
        });
        let refused = Outcome::refused("the consultant reached its step limit");
        decision.decide(Decider::plugin(CONSULT), refused);
    }
}

/// A consultant's failed model call is not tried again: the consult has a
/// deadline, and the other consultants answer meanwhile.
fn no_retries(
    mut failed: Query<(&ForAgent, &mut RetryVerdict), (ReadyToSettle, ModelCalls)>,
    consultants: Query<(), With<SlotOf>>,
) {
    for (&ForAgent(agent), mut verdict) in &mut failed {
        if consultants.contains(agent) && !matches!(verdict.0, Verdict::Final) {
            verdict.0 = Verdict::Final;
        }
    }
}

/// A consultant's turn ended: its answer, or why it has none.
fn turn_settled(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), With<Turn>>,
    agents: Query<&SlotOf>,
    mut consultations: Query<&mut Consultation>,
    spending: Query<&AgentUsage>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(&SlotOf { consultation, slot }) = agents.get(agent) else {
        return;
    };
    let Ok(mut consult) = consultations.get_mut(consultation) else {
        return;
    };
    if let Some(seat) = consult.slots.get_mut(slot) {
        seat.running = false;
    }
    let result = match outcome {
        TurnOutcome::Answered(message) => message
            .final_answer()
            .ok_or_else(|| "it gave no answer".to_owned()),
        TurnOutcome::Failed(why) => Err(why.clone()),
        TurnOutcome::Stopped(why) => Err(why.clone().unwrap_or_else(|| "stopped".to_owned())),
    };
    consult.end(slot, result, &spending);
    commands.trigger(CheckDone {
        entity: consultation,
    });
}

/// The consult's time is up: every consultant still at work failed.
fn timed_out(
    due: On<Add<Due>>,
    mut consultations: Query<&mut Consultation>,
    spending: Query<&AgentUsage>,
    mut commands: Commands,
) {
    let Ok(mut consult) = consultations.get_mut(due.entity) else {
        return;
    };
    for slot in 0..consult.slots.len() {
        consult.end(slot, Err("timeout".to_owned()), &spending);
    }
    commands.trigger(CheckDone { entity: due.entity });
}

/// Once every consultant ended: answers the call with the answers side by
/// side, or shows them to the user, and ends the consult.
fn finish(
    check: On<CheckDone>,
    consultations: Query<&Consultation>,
    mut calls: Query<&mut Decision, ToolCalls>,
    answerers: Answerers,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(consult) = consultations.get(check.entity) else {
        return;
    };
    if consult.slots.iter().any(|slot| slot.result.is_none()) {
        return;
    }
    let report = consult.report();
    match consult.call {
        Some(call) => {
            if let Ok(mut decision) = calls.get_mut(call) {
                let answered = ToolResult::success(report.into());
                decision.decide(Decider::plugin(CONSULT), Outcome::from(answered));
            }
        }
        None if answerers.can_answer(Some(consult.asker)) => {
            show(&mut commands, consult.asker, report);
        }
        None => {
            notices.write(Notice::info(consult.asker, report));
        }
    }
    commands.entity(check.entity).try_despawn();
}

/// Shows `/consult`'s answers to the user, with **Send to agent**.
fn show(commands: &mut Commands, asker: Entity, report: String) {
    let picker = Picker::new("Second opinions")
        .for_agent(asker)
        .body(report.clone())
        .item("Send to agent", "")
        .item("Close", "");
    commands
        .spawn(picker)
        .observe(move |picked: On<Picked>, mut commands: Commands| {
            if picked.chosen == Some(Chosen::item(0)) {
                let note = Deliver::new(asker, report.clone(), DeliveryMode::Inform);
                commands.trigger(note.with_author(Author::user().titled("consult")));
            }
        });
}

/// A consult that ends retires every consultant still at work, briefed or
/// not, with its work; its consultants' requests not answered yet go with
/// it, as its children.
fn stop_consultants(
    removed: On<Remove<Consultation>>,
    consultations: Query<&Consultation>,
    mut commands: Commands,
) {
    let Ok(consult) = consultations.get(removed.entity) else {
        return;
    };
    let working = consult.slots.iter().filter(|slot| slot.running);
    for agent in working.filter_map(|slot| slot.agent) {
        commands.trigger(Retire { entity: agent });
    }
}

/// Interrupting an agent ends its consults.
fn cancel_on_interrupt(
    interrupt: On<StopTurn>,
    consultations: Query<(Entity, &Consultation)>,
    mut commands: Commands,
) {
    for (entity, consult) in &consultations {
        if consult.asker == interrupt.entity {
            commands.entity(entity).try_despawn();
        }
    }
}

#[cfg(feature = "tui")]
mod render;

#[cfg(test)]
mod tests;
