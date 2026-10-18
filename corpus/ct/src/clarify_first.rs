//! CT-15: clarify first.
//!
//! `/clarify <task>` starts an interview before the agent works on a task:
//! a one-job helper subagent ([`SpawnSubagent::helper`], on the model the
//! `[clarify]` table names, else the agent's) that may use only the tools
//! that read, plus [`CLARIFY_ASK`]. Each `clarify_ask` call is one question
//! for the user, an [`Ask`] that holds the call, with the interviewer's
//! suggested answer first, its options, and "just start"; the user types an
//! answer, takes the suggestion or an option, or says "just start", which
//! answers every later question with its suggestion unasked. The
//! interviewer's final answer is the brief, to which the plugin adds each
//! question and how it was answered. The user then starts with the brief,
//! edits it (a typed answer is the edited brief), or cancels; "just start"
//! skips that review.
//!
//! On start, the agent gets the task and the brief as one message from the
//! user. The interviewer is an agent of its own, so nothing of its
//! conversation reaches the agent's. A run nobody answers questions in (no
//! [`Answerer`], or an unattended session such as `rig -p`'s) asks
//! nothing: every question takes its suggestion and the brief says so. With `mode = "auto"`, a message the
//! user sends that is longer than `auto_min_chars` or says "implement" or
//! "build" is interviewed on first; other messages go straight through.
//!
//! ```toml
//! [clarify]
//! mode = "auto"
//! model = "deepseek/deepseek-flash"
//! auto_min_chars = 400
//! ```

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool the interviewer asks the user with.
pub const CLARIFY_ASK: &str = "clarify_ask";

/// The interviewer's title, which names it.
pub const TITLE: &str = "clarify";

/// The most questions an interview asks.
pub const MAX_QUESTIONS: usize = 5;

/// Who decides what this plugin decides.
const DECIDER: &str = "clarify";

/// What `clarify_ask` says it does.
const ASK_DESCRIPTION: &str = "Ask the user one clarifying question about the task and wait for \
    the answer, which is the result. Always give `default`, the answer you would assume, and \
    `options` when the answer is one of a few.";

/// What the interviewer is told about its job.
const INTERVIEWER_PROMPT: &str = "You interview the user about a task before another agent \
    starts it. You may read the project to ask better questions; change nothing. Ask between \
    one and five short questions that matter for doing the task right, one `clarify_ask` call \
    at a time, each with the answer you would assume as `default`. Then answer with a short \
    brief and nothing else, under these headings: Goal, Constraints, Decisions (from the \
    answers), Open points.";

/// What the interviewer is told when nobody can answer.
const NOBODY_ANSWERS: &str = "Nobody can answer questions in this run: do not call \
    `clarify_ask`; write the brief from your defaults and mark them as assumptions.";

/// The `[clarify]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClarifySettings {
    /// When the user's messages are interviewed on without `/clarify`.
    pub mode: ClarifyMode,
    /// The interviewer's model, as `vendor/model`; the agent's when not
    /// set.
    pub model: Option<String>,
    /// In [`ClarifyMode::Auto`], a message longer than this many
    /// characters is interviewed on. 400 when not set.
    pub auto_min_chars: usize,
}

impl Default for ClarifySettings {
    fn default() -> Self {
        Self {
            mode: ClarifyMode::Off,
            model: None,
            auto_min_chars: 400,
        }
    }
}

/// When the user's messages are interviewed on without `/clarify`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClarifyMode {
    /// Only with `/clarify`.
    #[default]
    Off,
    /// Also a message that is long or says "implement" or "build".
    Auto,
}

impl ClarifySettings {
    /// Whether `text`, a message the user sent, is interviewed on first.
    pub fn wants_interview(&self, text: &str) -> bool {
        let asks_for_work = text.split(|c: char| !c.is_alphanumeric()).any(|word| {
            word.eq_ignore_ascii_case("implement") || word.eq_ignore_ascii_case("build")
        });
        self.mode == ClarifyMode::Auto
            && (text.chars().count() > self.auto_min_chars || asks_for_work)
    }
}

/// How a question was answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answered {
    /// The user's own answer, typed or picked.
    User(String),
    /// The suggested answer, which the user took.
    Default(String),
    /// The suggested answer, taken unasked: after "just start", in a run
    /// nobody answers, or past [`MAX_QUESTIONS`].
    Unasked(String),
    /// The user said "just start" here, so it took its suggestion.
    JustStart(String),
}

impl Answered {
    /// The answer, as the brief lists it.
    fn line(&self) -> String {
        match self {
            Self::User(text) => text.clone(),
            Self::Default(text) => format!("{text} (the suggested default)"),
            Self::Unasked(text) => format!("{text} (default, not asked)"),
            Self::JustStart(text) => format!("{text} (default; the user said just start)"),
        }
    }

    /// What the interviewer reads as the call's result.
    fn result(&self) -> String {
        match self {
            Self::User(text) => format!("The user answered: {text}"),
            Self::Default(text) => format!("The user took your suggestion: {text}"),
            Self::Unasked(text) => format!(
                "Not asked; take your suggestion ({text}). Ask nothing more and write the brief \
                 now."
            ),
            Self::JustStart(text) => format!(
                "The user said just start: take your suggestion ({text}), ask nothing more and \
                 write the brief now, with your defaults for what is still open."
            ),
        }
    }
}

/// An interview on a task, on an entity of its own until the agent gets
/// the task or the user cancels.
#[derive(Component, Clone, Debug)]
pub struct Interview {
    /// The agent the task is for.
    pub agent: Entity,
    /// The task as the user wrote it.
    pub task: String,
    /// How the task goes to the agent.
    pub mode: DeliveryMode,
    /// Whether nobody answers questions about the agent.
    pub headless: bool,
    /// Whether the user said "just start".
    pub just_start: bool,
    /// Each question and how it was answered, in order.
    pub answers: Vec<(String, Answered)>,
    /// The brief, once the interviewer wrote it.
    pub brief: Option<String>,
}

impl Interview {
    /// The brief with the questions' answers and how they were given.
    fn full_brief(&self, written: &str) -> String {
        let mut brief = written.trim().to_owned();
        if !self.answers.is_empty() {
            brief.push_str("\n\nAnswers from the interview:");
            for (question, answer) in &self.answers {
                brief.push_str(&format!("\n- {question} -> {}", answer.line()));
            }
        }
        if self.headless {
            brief.push_str(
                "\n\nNote: no one could answer questions in this run, so the questions were \
                 skipped and their defaults used.",
            );
        } else if self.just_start {
            brief.push_str(
                "\n\nNote: the user said just start; open questions took their defaults.",
            );
        }
        brief
    }

    /// The message the agent gets: the task and the brief.
    fn message(&self, brief: &str) -> String {
        format!(
            "{}\n\nBrief from a clarifying interview with the user:\n{brief}",
            self.task.trim()
        )
    }
}

/// On the interviewer agent: its interview.
#[derive(Component, Clone, Copy, Debug)]
pub struct Interviewer(pub Entity);

/// On the interviewer's spawn request: its interview.
#[derive(Component, Clone, Copy, Debug)]
struct InterviewerOf(Entity);

/// On a question for one `clarify_ask` call: its interview, the question
/// and its suggested answer, and the options offered by choice id.
#[derive(Component, Clone, Debug)]
struct Question {
    interview: Entity,
    text: String,
    default: String,
    options: Vec<String>,
}

/// On the question that shows the brief: its interview.
#[derive(Component, Clone, Copy, Debug)]
struct BriefReview(Entity);

/// On the message that carries a task and its brief, so auto mode lets it
/// through.
#[derive(Component, Clone, Copy, Debug)]
pub struct Clarified;

/// The arguments of `clarify_ask`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AskArgs {
    /// The question, in a line.
    question: String,
    /// The answer you would assume, which the user may take.
    default: String,
    /// Other answers to pick from, if any.
    #[serde(default)]
    options: Vec<String>,
}

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClarifySystems {
    /// Prepare (`PrepareSystems::Edit`): offers `clarify_ask` only to
    /// interviewers.
    Offer,
    /// Gate, after `AskSystems`: in auto mode, takes over a user's
    /// message that wants an interview.
    Auto,
    /// Review: reads each answered question.
    Answers,
}

/// `/clarify`, the interviewer, its questions, the brief and auto mode.
#[derive(Default)]
pub struct ClarifyFirstPlugin;

impl Plugin for ClarifyFirstPlugin {
    fn build(&self, app: &mut App) {
        // The interviewer is a helper subagent: without the plugin nothing
        // starts it and `/clarify` would wait forever.
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        let options = ToolOptions {
            // One question at a time, in the order asked.
            footprint: Some(Touches::Writes(Scope::Anything)),
        };
        app.add_setting::<ClarifySettings>("clarify")
            .add_open_tool(CLARIFY_ASK, ASK_DESCRIPTION, options, ask)
            .add_command(
                "clarify",
                "Answer a few questions before the agent starts: /clarify <task>",
                clarify,
            )
            .add_observer(interviewer_spawned)
            .add_observer(brief_written)
            .configure_sets(
                AgentPass,
                (
                    ClarifySystems::Offer.in_set(PrepareSystems::Edit),
                    ClarifySystems::Auto.in_set(Stage::Gate).after(AskSystems),
                    ClarifySystems::Answers.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    offer_to_interviewers.in_set(ClarifySystems::Offer),
                    auto_interview.in_set(ClarifySystems::Auto),
                    answered.in_set(ClarifySystems::Answers),
                ),
            );
    }
}

/// What starting an interview reads.
#[derive(bevy_ecs::system::SystemParam)]
struct Starter<'w, 's> {
    answerers: Answerers<'w, 's>,
    unattended: Option<Res<'w, rig_harness::Unattended>>,
    settings: Res<'w, ClarifySettings>,
    agents: Query<'w, 's, &'static ToolAccess>,
    tools: Query<'w, 's, (&'static ToolDef, &'static Footprint)>,
    commands: Commands<'w, 's>,
}

impl Starter<'_, '_> {
    /// Starts an interview on `task` for `agent`.
    fn start(&mut self, agent: Entity, task: &str, mode: DeliveryMode) {
        // The questions are optional: an unattended session skips them.
        let unattended = self.unattended.as_deref().is_some_and(|unattended| unattended.0);
        let headless = unattended || !self.answerers.can_answer(Some(agent));
        let access = self.agents.get(agent).ok();
        let mut tools: Vec<String> = self
            .tools
            .iter()
            .filter(|(def, footprint)| {
                let name = def.0.name.as_str();
                footprint.0.reads_only()
                    && name != CLARIFY_ASK
                    && access.is_none_or(|access| access.allows(name))
            })
            .map(|(def, _)| def.0.name.to_string())
            .collect();
        tools.push(CLARIFY_ASK.to_owned());
        let interview = Interview {
            agent,
            task: task.trim().to_owned(),
            mode,
            headless,
            just_start: false,
            answers: Vec::new(),
            brief: None,
        };
        let mut brief = format!("Task: {}", interview.task);
        if headless {
            brief.push_str("\n\n");
            brief.push_str(NOBODY_ANSWERS);
        }
        let interview = self
            .commands
            .spawn((Name::new("clarify interview"), interview))
            .id();
        let mut request = SpawnSubagent::helper(TITLE)
            .system_prompt(INTERVIEWER_PROMPT)
            .brief(brief)
            .once();
        request.tools = Some(tools);
        request.model.clone_from(&self.settings.model);
        self.commands
            .spawn((request, ForAgent(agent), InterviewerOf(interview)));
    }
}

/// `/clarify <task>`.
fn clarify(In(args): In<CommandArgs>, mut starter: Starter, mut notices: MessageWriter<Notice>) {
    if args.args.trim().is_empty() {
        notices.write(Notice::error(
            args.agent,
            "/clarify needs a task: /clarify <task>",
        ));
        return;
    }
    starter.start(args.agent, &args.args, args.mode);
}

/// In auto mode, a message the user sends an agent nothing spawned that
/// wants an interview goes to one first.
fn auto_interview(
    mut typed: Query<
        (&Incoming, &ForAgent, &mut Decision),
        (
            ReadyToStart,
            IncomingMessages,
            Without<Clarified>,
            Without<Restored>,
        ),
    >,
    roots: Query<(), (With<Agent>, Without<SpawnedBy>)>,
    mut starter: Starter,
) {
    if starter.settings.mode != ClarifyMode::Auto {
        return;
    }
    for (incoming, &ForAgent(agent), mut decision) in &mut typed {
        let by_user = incoming.author.kind == AuthorKind::User && incoming.command().is_none();
        if decision.is_decided()
            || !by_user
            || !roots.contains(agent)
            || !starter.settings.wants_interview(incoming.text())
        {
            continue;
        }
        decision.decide(
            Decider::plugin(DECIDER),
            Outcome::custom("clarifying first"),
        );
        starter.start(agent, incoming.text(), incoming.mode);
    }
}

/// Marks the interviewer once it is spawned.
fn interviewer_spawned(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &InterviewerOf)>,
    mut commands: Commands,
) {
    if let Ok((&SpawnedAgent(agent), &InterviewerOf(interview))) = requests.get(spawned.entity) {
        commands.entity(agent).insert(Interviewer(interview));
    }
}

/// Offers `clarify_ask` only to interviewers.
fn offer_to_interviewers(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    interviewers: Query<(), With<Interviewer>>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if !interviewers.contains(agent) {
            call.request_mut()
                .tools
                .retain(|tool| tool.name.as_str() != CLARIFY_ASK);
        }
    }
}

/// A `clarify_ask` call: asks the user, holding the call, or takes the
/// suggestion at once when the question is not to be asked.
fn ask(
    called: On<ToolCalled<AskArgs>>,
    interviewers: Query<&Interviewer>,
    mut interviews: Query<&mut Interview>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let call = called.call;
    let mut decide = |outcome: Outcome| {
        if let Ok(mut decision) = calls.get_mut(call) {
            decision.decide(Decider::plugin(DECIDER), outcome);
        }
    };
    let Some((interview, mut state)) = interviewers
        .get(called.agent)
        .ok()
        .and_then(|&Interviewer(interview)| Some((interview, interviews.get_mut(interview).ok()?)))
    else {
        decide(Outcome::refused(
            "Only a clarifying interviewer asks with `clarify_ask`.",
        ));
        return;
    };
    let args = &called.args;
    let (question, default) = (
        args.question.trim().to_owned(),
        args.default.trim().to_owned(),
    );
    if state.headless || state.just_start || state.answers.len() >= MAX_QUESTIONS {
        let answer = Answered::Unasked(default);
        decide(ToolResult::success(answer.result().into()).into());
        state.answers.push((question, answer));
        return;
    }
    let options: Vec<String> = args
        .options
        .iter()
        .map(|option| option.trim().to_owned())
        .filter(|option| !option.is_empty() && *option != default)
        .collect();
    let mut asked = Ask::new(question.clone())
        .body(format!("Suggested: {default}"))
        .choice("default", format!("Use the suggestion: {default}"));
    for (at, option) in options.iter().enumerate() {
        asked = asked.choice(format!("option-{at}"), option.clone());
    }
    let asked = asked.choice("start", "Just start").free_text();
    commands.spawn((
        asked,
        Question {
            interview,
            text: question,
            default,
            options,
        },
        ForAgent(state.agent),
        HoldOf(call),
        Effect::with_parent_id(Some(called.effect)),
    ));
}

/// What a question's answer means.
fn read_answer(question: &Question, outcome: &Outcome) -> Answered {
    let Some(answer) = Answer::of(outcome) else {
        return Answered::Default(question.default.clone());
    };
    let typed = answer
        .text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    if answer.is("start") || typed.is_some_and(|text| text.eq_ignore_ascii_case("just start")) {
        return Answered::JustStart(question.default.clone());
    }
    let picked = answer.choices.iter().find_map(|id| {
        let at: usize = id.strip_prefix("option-")?.parse().ok()?;
        question.options.get(at).cloned()
    });
    match (picked, typed) {
        (Some(option), Some(note)) => Answered::User(format!("{option} ({note})")),
        (Some(option), None) => Answered::User(option),
        (None, Some(text)) if !answer.is("default") => Answered::User(text.to_owned()),
        _ => Answered::Default(question.default.clone()),
    }
}

/// Gives each answered question's call its answer and lets go of it, and
/// starts or cancels each reviewed brief.
fn answered(
    questions: Query<(Entity, &Outcome, &Question), (PluginEffects, With<Ask>, Added<Outcome>)>,
    reviews: Query<(&Outcome, &BriefReview), (PluginEffects, With<Ask>, Added<Outcome>)>,
    mut interviews: Query<&mut Interview>,
    mut calls: Query<&mut Decision, ToolCalls>,
    holds: Query<&HoldOf>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (asked, outcome, question) in &questions {
        let answer = read_answer(question, outcome);
        if let Ok(mut interview) = interviews.get_mut(question.interview) {
            interview.just_start |= matches!(answer, Answered::JustStart(_));
            interview
                .answers
                .push((question.text.clone(), answer.clone()));
        }
        if let Ok(&HoldOf(call)) = holds.get(asked)
            && let Ok(mut decision) = calls.get_mut(call)
        {
            decision.decide(
                Decider::plugin(DECIDER),
                ToolResult::success(answer.result().into()),
            );
        }
        commands.entity(asked).remove::<HoldOf>();
    }
    for (outcome, &BriefReview(interview)) in &reviews {
        let Ok(state) = interviews.get(interview) else {
            continue;
        };
        let answer = Answer::of(outcome).unwrap_or_else(|| Answer::choice("cancel"));
        let edited = answer
            .text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty());
        if answer.is("cancel") || (answer.choices.is_empty() && edited.is_none()) {
            notices.write(Notice::info(
                state.agent,
                "Clarify cancelled; the task was not sent.",
            ));
        } else if let Some(brief) = edited.map(str::to_owned).or_else(|| state.brief.clone()) {
            send(&mut commands, state, &brief);
        }
        commands.entity(interview).despawn();
    }
}

/// Sends the agent the task with `brief`, as one message from the user.
fn send(commands: &mut Commands, interview: &Interview, brief: &str) {
    let author = Author::user().titled(TITLE);
    let message = Incoming::new(interview.message(brief), interview.mode).with_author(author);
    commands.spawn((message, ForAgent(interview.agent), Clarified));
}

/// The interviewer's turn ended: its answer is the brief, which the user
/// reviews, or which goes to the agent at once after "just start" or in a
/// run nobody answers.
fn brief_written(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), With<Turn>>,
    interviewers: Query<&Interviewer>,
    mut interviews: Query<&mut Interview>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(&Interviewer(interview)) = interviewers.get(agent) else {
        return;
    };
    let Ok(mut state) = interviews.get_mut(interview) else {
        return;
    };
    let written = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => {
            let text = format!("The clarifying interview failed ({why}); the task was not sent.");
            notices.write(Notice::error(state.agent, text));
            commands.entity(interview).despawn();
            return;
        }
        TurnOutcome::Stopped(_) => {
            let text = "The clarifying interview was stopped; the task was not sent.";
            notices.write(Notice::info(state.agent, text));
            commands.entity(interview).despawn();
            return;
        }
    };
    let brief = state.full_brief(&written);
    if state.headless || state.just_start {
        send(&mut commands, &state, &brief);
        commands.entity(interview).despawn();
        return;
    }
    let review = Ask::new("Start with this brief?")
        .body(brief.clone())
        .initial(brief.clone())
        .multiline()
        .choice("start", "Start")
        .choice("cancel", "Cancel");
    commands.spawn((review, BriefReview(interview), ForAgent(state.agent)));
    state.brief = Some(brief);
}

#[cfg(test)]
mod tests;
