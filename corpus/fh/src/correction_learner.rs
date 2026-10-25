//! Plugin corpus (fresh held-out): FH-11: learn project rules from
//! corrections.
//!
//! When the user's message reaches an agent that answered before, a model
//! call of the plugin's own (on `[correction-learner] model`, else the
//! agent's model) reads that message, the agent's previous reply and its
//! tool calls in short form, and the rules already learned or dismissed,
//! and answers one rule of at most 200 characters, or `NONE`. The call is
//! no turn's: the agent's turns never wait for it, and its usage counts
//! under the purpose `correction-learner`.
//!
//! A new rule is a [`Proposal`]: where someone sits at a front, an inline
//! [`Ask`] beside the work, which waits until it is answered Save, Dismiss,
//! or with an edited rule, or until a newer proposal replaces it; in a run
//! nobody answers, an effect recorded as logged and nothing more. Saving
//! appends the rule under `## Learned rules` in the project's instruction
//! file, which rig's prompt has every agent read for itself, and tells the
//! corrected agent's model in a note. Dismissed rules are kept in the project's
//! `.rig/dismissed-rules.txt` and never proposed again. `/rules` lists the
//! learned rules, and `/rules rm N` removes one. Every file is read and
//! written through the project's workspace, off the main thread when it is
//! not this machine.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::agents_md;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::Message;
use rig_tools::workspace::{Stat, Workspace};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The settings' table, the classifier calls' purpose and the logged
/// proposals' effect kind.
pub const NAME: &str = "correction-learner";

/// The heading the learned rules are listed under.
pub const HEADING: &str = "## Learned rules";

/// The longest rule kept.
pub const RULE_CHARS: usize = 200;

/// How much of each reply text and tool call the classifier reads.
const SHORT: usize = 300;

/// What the classifier is told.
const PREAMBLE: &str = "You watch a user working with a coding agent. Given the agent's \
    previous reply and tool calls and the user's next message, decide whether the user \
    corrects the agent in a way that is a lasting rule for this project, such as which tool, \
    command or convention to use, or which files never to touch. If so, answer with that rule \
    alone: one short imperative sentence of at most 200 characters, such as `Use pnpm, not \
    npm.` Answer NONE when the message is no such correction, or when the rule is already \
    among the learned or dismissed rules you are shown.";

/// Adds the classifier, the proposals and `/rules`.
#[derive(Default)]
pub struct CorrectionLearnerPlugin;

impl Plugin for CorrectionLearnerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_setting::<CorrectionLearnerConfig>(NAME)
            .add_observer(classify)
            .add_observer(ask_classifier)
            .add_observer(follow_successor::<Classifying>)
            .add_observer(check)
            .add_observer(propose)
            .add_observer(answered)
            .add_observer(said)
            .add_command(
                "rules",
                "Learned project rules: /rules lists them, /rules rm N removes one",
                rules,
            );
    }
}

/// The `[correction-learner]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorrectionLearnerConfig {
    /// The catalog model (`vendor/model`) that classifies; none uses the
    /// agent's own.
    pub model: Option<String>,
}

/// On a classifier call, and on whichever call takes it over: the agent
/// and the index of the turn whose user message it reads.
#[derive(Component, Clone, Copy, Debug)]
pub struct Classifying {
    /// The agent corrected.
    pub agent: Entity,
    /// The turn the message went with ([`Turn::index`]).
    pub turn: u64,
}

/// On a proposal: the rule, and the agent and turn it came from. Shown, it
/// is an inline [`Ask`]; in a run nobody answers, a logged effect.
#[derive(Component, Clone, Debug)]
pub struct Proposal {
    /// The rule.
    pub rule: String,
    /// The agent corrected.
    pub agent: Entity,
    /// The turn whose user message it came from ([`Turn::index`]).
    pub turn: u64,
}

/// What a piece of file work ends as: its result, or why it failed.
type Worked<T> = Done<Result<T, String>>;

/// The rules learned and dismissed, as read.
#[derive(Clone, Debug, Default)]
struct Known {
    learned: Vec<String>,
    dismissed: Vec<String>,
}

impl Known {
    /// Whether `rule` is among them, ignoring case and punctuation.
    fn has(&self, rule: &str) -> bool {
        let rule = normalized(rule);
        (self.learned.iter().chain(&self.dismissed)).any(|known| normalized(known) == rule)
    }
}

/// On file work that reads the rules to ask the classifier with: what it
/// asks about.
#[derive(Component)]
struct Reading {
    classifying: Classifying,
    previous: String,
    message: String,
}

/// On file work that reads the rules to check a classifier's rule against.
#[derive(Component)]
struct Checking(Proposal);

/// On file work whose result is said to `agent`; with `tell`, to its model
/// too, in a note.
#[derive(Component)]
struct Saying {
    agent: Entity,
    tell: bool,
}

/// The project's instruction file and the plugin's file of dismissed
/// rules, in the project's workspace.
struct Files {
    workspace: Arc<dyn Workspace>,
    root: PathBuf,
}

impl Files {
    fn of(workspace: &ProjectWorkspace, root: &ProjectRoot) -> Self {
        Self {
            workspace: workspace.0.clone(),
            root: root.0.clone(),
        }
    }

    /// The project's `AGENTS.md`.
    fn instructions(&self) -> PathBuf {
        self.root.join(agents_md::FILE_NAME)
    }

    fn dismissed_file(&self) -> PathBuf {
        self.root.join(".rig").join("dismissed-rules.txt")
    }

    /// The text of `path`, empty when it does not exist.
    fn text(&self, path: &Path) -> Result<String, String> {
        match self.workspace.stat(path) {
            Ok(Stat::Missing) => Ok(String::new()),
            _ => (self.workspace.read_text(path)).map_err(|why| why.to_string()),
        }
    }

    fn write(&self, path: &Path, text: &str) -> Result<(), String> {
        let wrote = self.workspace.write(path, text.as_bytes());
        wrote.map_err(|why| format!("cannot write {}: {why}", path.display()))
    }

    fn known(&self) -> Known {
        let instructions = self.text(&self.instructions()).unwrap_or_default();
        let dismissed = self.text(&self.dismissed_file()).unwrap_or_default();
        Known {
            learned: learned(&instructions),
            dismissed: dismissed.lines().map(str::to_owned).collect(),
        }
    }

    /// Appends `rule` under [`HEADING`], made with the file if missing.
    fn save(&self, rule: &str) -> Result<String, String> {
        let path = self.instructions();
        self.write(&path, &with_rule(&self.text(&path)?, rule))?;
        Ok(format!("Saved to {}: {rule}", path.display()))
    }

    /// Remembers `rule` as dismissed.
    fn dismiss(&self, rule: &str) -> Result<String, String> {
        let path = self.dismissed_file();
        let mut text = self.text(&path)?;
        text.push_str(rule);
        text.push('\n');
        self.write(&path, &text)?;
        Ok(format!("Dismissed: {rule}"))
    }

    /// The learned rules, numbered.
    fn list(&self) -> Result<String, String> {
        let path = self.instructions();
        let learned = learned(&self.text(&path)?);
        let lines = learned
            .iter()
            .enumerate()
            .map(|(at, rule)| format!("{}. {rule}", at + 1));
        Ok(match learned.is_empty() {
            true => format!("No learned rules in {}.", path.display()),
            false => format!(
                "Learned rules in {}:\n{}",
                path.display(),
                lines.collect::<Vec<_>>().join("\n")
            ),
        })
    }

    /// Removes learned rule `number`, counted from 1.
    fn remove(&self, number: usize) -> Result<String, String> {
        let path = self.instructions();
        let Some((kept, removed)) = without_rule(&self.text(&path)?, number) else {
            return Err(format!("There is no learned rule {number}."));
        };
        self.write(&path, &kept)?;
        Ok(format!(
            "Removed rule {number} from {}: {removed}",
            path.display()
        ))
    }

    /// Runs `job` on the files: at once on this machine, else off the main
    /// thread. It ends as a [`Worked`] on `entity`.
    fn work<T: Send + Sync + 'static>(
        self,
        mut entity: EntityCommands,
        wake: &Wake,
        job: impl FnOnce(&Self) -> Result<T, String> + Send + 'static,
    ) {
        if self.workspace.host().is_none() {
            entity.insert(Done(job(&self)));
            return;
        }
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let run = blocking(move || Ok(job(&self)));
        let run = async move { run.await.unwrap_or_else(|why| Err(why.to_string())) };
        entity.insert(Running::spawn(pool, wake, run));
    }
}

/// Reads the known rules for each message the user typed that went into
/// a turn of an agent that answered before, to ask the classifier.
fn classify(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&Incoming, &ForAgent, &DeliveredIn)>,
    turns: Query<&Turn>,
    conversations: Query<&Conversation>,
    (workspace, root, wake): (Res<ProjectWorkspace>, Res<ProjectRoot>, Res<Wake>),
    mut commands: Commands,
) {
    let Ok((message, &ForAgent(agent), &DeliveredIn(turn))) = messages.get(delivered.entity) else {
        return;
    };
    if message.author.kind != AuthorKind::User {
        return;
    }
    let (Ok(turn), Ok(conversation)) = (turns.get(turn), conversations.get(agent)) else {
        return;
    };
    let Some(previous) = previous_reply(conversation) else {
        return;
    };
    let reading = Reading {
        classifying: Classifying {
            agent,
            turn: turn.index,
        },
        previous,
        message: message.text().trim().to_owned(),
    };
    Files::of(&workspace, &root).work(commands.spawn(reading), &wake, |files| Ok(files.known()));
}

/// Asks the classifier, once the known rules are read.
fn ask_classifier(
    read: On<Insert<Worked<Known>>>,
    readings: Query<(&Reading, &Worked<Known>)>,
    (config, models): (Res<CorrectionLearnerConfig>, Res<Models>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((reading, Done(known))) = readings.get(read.entity) else {
        return;
    };
    commands.entity(read.entity).despawn();
    let known = known.clone().unwrap_or_default();
    let listed = |rules: &[String]| match rules.is_empty() {
        true => "(none)".to_owned(),
        false => (rules.iter().map(|rule| format!("- {rule}")))
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let prompt = format!(
        "Learned rules:\n{}\n\nDismissed rules:\n{}\n\nThe agent's previous reply:\n{}\n\n\
         The user's message:\n{}",
        listed(&known.learned),
        listed(&known.dismissed),
        reading.previous,
        reading.message,
    );
    let agent = reading.classifying.agent;
    let mut request =
        ModelRequest::new(CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE));
    if let Some(reference) = config.model.as_deref() {
        match models.connect(reference) {
            Ok(connection) => request = request.on(connection),
            Err(why) => {
                let text = format!(
                    "{NAME}: cannot use {reference}, so the agent's model classifies: {why}"
                );
                notices.write(Notice::warn(agent, text));
            }
        }
    }
    commands.spawn((
        Name::new("correction check"),
        reading.classifying,
        request,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
    ));
}

/// The agent's messages between the user's last two typed messages, in
/// short form: its texts and tool calls; `None` when it has none.
fn previous_reply(conversation: &Conversation) -> Option<String> {
    let messages = conversation.messages();
    let typed_at = |at: &usize| typed(conversation, *at).is_some();
    let current = (0..messages.len()).rev().find(typed_at)?;
    let start = (0..current).rev().find(typed_at).map_or(0, |at| at + 1);
    let mut lines = Vec::new();
    for message in messages.get(start..current).unwrap_or_default() {
        let Message::Assistant(reply) = &**message else {
            continue;
        };
        for item in reply.content.iter() {
            match item {
                AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                    lines.push(rig_tools::shorten(text.text.trim(), SHORT));
                }
                AssistantContent::ToolCall(call) => {
                    let args = serde_json::to_string(&call.function.arguments).unwrap_or_default();
                    let call = format!("called {} {args}", call.function.name.as_str());
                    lines.push(rig_tools::shorten(&call, SHORT));
                }
                _ => {}
            }
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// Reads the known rules again for each rule a classifier answered, to
/// check it is new.
fn check(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &Classifying)>,
    (workspace, root, wake): (Res<ProjectWorkspace>, Res<ProjectRoot>, Res<Wake>),
    mut commands: Commands,
) {
    let Ok((outcome, &Classifying { agent, turn })) = calls.get(settled.entity) else {
        return;
    };
    let Some(rule) = outcome.completion().and_then(|reply| rule(&reply.text())) else {
        return;
    };
    let checking = Checking(Proposal { rule, agent, turn });
    Files::of(&workspace, &root).work(commands.spawn(checking), &wake, |files| Ok(files.known()));
}

/// Proposes a classifier's rule that is neither known nor proposed
/// already: inline beside the work, in place of the one waiting, or, when
/// nobody can answer, only logged.
fn propose(
    read: On<Insert<Worked<Known>>>,
    checks: Query<(&Checking, &Worked<Known>)>,
    mut waiting: Query<(&Proposal, &mut Decision), (With<Ask>, Without<Outcome>)>,
    answerers: Answerers,
    mut commands: Commands,
) {
    let Ok((Checking(proposal), Done(known))) = checks.get(read.entity) else {
        return;
    };
    commands.entity(read.entity).despawn();
    let rule = &proposal.rule;
    let shown = (waiting.iter()).any(|(shown, _)| normalized(&shown.rule) == normalized(rule));
    if shown || known.as_ref().is_ok_and(|known| known.has(rule)) {
        return;
    }
    let by = Decider::plugin("fh-correction-learner");
    if !answerers.can_answer(Some(proposal.agent)) {
        let kind = EffectKind::Custom {
            kind: NAME.into(),
            payload: serde_json::json!({ "rule": rule, "turn": proposal.turn }),
        };
        let mut logged = Decision::default();
        logged.decide(by, Outcome::custom("logged"));
        let request = EffectRequest {
            kind,
            handler: None,
        };
        commands.spawn((Name::new("rule proposal"), request, logged));
        return;
    }
    for (_, mut decision) in &mut waiting {
        decision.decide(by.clone(), Outcome::refused("a newer rule replaced it"));
    }
    let ask = Ask::new("Remember for this project?")
        .body(rule.clone())
        .choice("save", "Save")
        .choice("dismiss", "Dismiss")
        .initial(rule.clone())
        .inline();
    commands.spawn((ask, proposal.clone(), ForAgent(proposal.agent)));
}

/// Saves or dismisses a proposal as the user answered it: Save, Dismiss,
/// or an edited rule to save. Closed or replaced, it is left.
fn answered(
    added: On<Add<Outcome>>,
    proposals: Query<(&Proposal, &Outcome), With<Ask>>,
    (workspace, root, wake): (Res<ProjectWorkspace>, Res<ProjectRoot>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((proposal, outcome)) = proposals.get(added.entity) else {
        return;
    };
    let agent = proposal.agent;
    let files = Files::of(&workspace, &root);
    let saying = |tell| Saying { agent, tell };
    let rule = match Answer::of(outcome) {
        Some(answer) if answer.is("save") => proposal.rule.clone(),
        Some(answer) if answer.is("dismiss") => {
            let rule = proposal.rule.clone();
            files.work(commands.spawn(saying(false)), &wake, move |files| {
                files.dismiss(&rule)
            });
            return;
        }
        Some(Answer {
            text: Some(text), ..
        }) => text.split_whitespace().collect::<Vec<_>>().join(" "),
        _ => return,
    };
    if rule.is_empty() || rule.chars().count() > RULE_CHARS {
        let why = format!("A rule takes 1 to {RULE_CHARS} characters; not saved.");
        notices.write(Notice::error(agent, why));
        return;
    }
    files.work(commands.spawn(saying(true)), &wake, move |files| {
        files.save(&rule)
    });
}

/// Says how file work went and, after a change to the instruction file,
/// tells the agent's model in a note.
fn said(
    done: On<Insert<Worked<String>>>,
    works: Query<(&Saying, &Worked<String>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((saying, Done(result))) = works.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    match result {
        Ok(text) => {
            if saying.tell {
                let note = Deliver::new(saying.agent, text.clone(), DeliveryMode::Inform);
                commands.trigger(note.with_author(Author::plugin(NAME)));
            }
            notices.write(Notice::info(saying.agent, text.clone()));
        }
        Err(why) => {
            notices.write(Notice::error(saying.agent, why.clone()));
        }
    }
}

/// `/rules`: lists the learned rules; `/rules rm N` removes one.
fn rules(
    In(args): In<CommandArgs>,
    (workspace, root, wake): (Res<ProjectWorkspace>, Res<ProjectRoot>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let files = Files::of(&workspace, &root);
    let agent = args.agent;
    let saying = |tell| Saying { agent, tell };
    let (verb, rest) = args.args.split_once(' ').unwrap_or((&args.args, ""));
    match (verb, rest.trim().parse::<usize>()) {
        ("", _) => {
            files.work(commands.spawn(saying(false)), &wake, Files::list);
        }
        ("rm", Ok(number)) => {
            files.work(commands.spawn(saying(true)), &wake, move |files| {
                files.remove(number)
            });
        }
        ("rm", Err(_)) => {
            notices.write(Notice::error(
                agent,
                "/rules rm needs the number /rules lists.",
            ));
        }
        _ => {
            notices.write(Notice::error(agent, "/rules takes nothing, or rm N."));
        }
    }
}

/// The rule a classifier answered: its first line, without list marks or
/// quotes; `None` for `NONE`, nothing, or one too long.
fn rule(answer: &str) -> Option<String> {
    let line = answer
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    let line = line.trim_start_matches(['-', '*', ' ']);
    let rule = line.trim_matches(['`', '"', '\'', ' ']).to_owned();
    let none = rule.trim_end_matches('.').eq_ignore_ascii_case("none");
    (!none && !rule.is_empty() && rule.chars().count() <= RULE_CHARS).then_some(rule)
}

/// `rule` lowercased, with only its letters, digits and single spaces.
fn normalized(rule: &str) -> String {
    let kept: String = rule
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The lines of the [`HEADING`] section of `text`: from the heading to the
/// next heading.
fn section(lines: &[&str]) -> Option<std::ops::Range<usize>> {
    let heading = lines.iter().position(|line| line.trim() == HEADING)?;
    let after = heading + 1;
    let end = lines
        .iter()
        .skip(after)
        .position(|line| line.starts_with('#'))
        .map_or(lines.len(), |at| after + at);
    Some(heading..end)
}

/// The bullets of the [`HEADING`] section of `text`.
fn learned(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let Some(range) = section(&lines) else {
        return Vec::new();
    };
    let section = lines.get(range).unwrap_or_default();
    section
        .iter()
        .filter_map(|line| line.strip_prefix("- "))
        .map(|rule| rule.trim().to_owned())
        .collect()
}

/// `text` with `rule` as the last bullet of its [`HEADING`] section, the
/// section added at the end when missing.
fn with_rule(text: &str, rule: &str) -> String {
    let bullet = format!(
        "- {}",
        rule.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    let mut lines: Vec<&str> = text.lines().collect();
    match section(&lines) {
        Some(range) => {
            let section = lines.get(range.clone()).unwrap_or_default();
            let last = section.iter().rposition(|line| line.starts_with("- "));
            let at = last.map_or(range.start + 1, |last| range.start + last + 1);
            if last.is_none() {
                lines.insert(at, "");
            }
            lines.insert(at + usize::from(last.is_none()), &bullet);
        }
        None => {
            while lines.last().is_some_and(|line| line.trim().is_empty()) {
                lines.pop();
            }
            if !lines.is_empty() {
                lines.push("");
            }
            lines.extend([HEADING, "", &bullet]);
        }
    }
    lines.join("\n") + "\n"
}

/// `text` without the `number`th bullet of its [`HEADING`] section,
/// counted from 1, and that rule.
fn without_rule(text: &str, number: usize) -> Option<(String, String)> {
    let mut lines: Vec<&str> = text.lines().collect();
    let range = section(&lines)?;
    let section = lines.get(range.clone()).unwrap_or_default();
    let bullets = section
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("- "));
    let (at, line) = bullets
        .map(|(at, line)| (range.start + at, *line))
        .nth(number.checked_sub(1)?)?;
    let removed = line.trim_start_matches("- ").trim().to_owned();
    lines.remove(at);
    Some((lines.join("\n") + "\n", removed))
}
