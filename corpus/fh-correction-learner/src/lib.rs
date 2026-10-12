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
//! A new rule is a [`Proposal`], an open effect recorded in the session's
//! effect log with how it was answered. Where someone sits at a front, it
//! waits, shown under the transcript with the `tui` feature, until it is
//! saved (`/rules save`, Alt+S), saved as edited (`/rules save <text>`;
//! Alt+E puts that line in the input), dismissed (`/rules dismiss`, Alt+X)
//! or replaced by a newer one. In a run nobody answers, it is only logged.
//! Saving appends the rule under `## Learned rules` in the project's
//! instruction file and re-reads the project's prompt section, so every
//! agent's next request has it. Dismissed rules are kept in the project's
//! `.rig/dismissed-rules.txt` and never proposed again. `/rules` lists the
//! learned rules, and `/rules rm N` removes one.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::Message;
use rig_tools::context::{Instructions, instruction_file};
use serde::Deserialize;

#[cfg(feature = "tui")]
mod tui;

#[cfg(test)]
mod tests;

/// The settings' table, the classifier calls' purpose and the proposals'
/// effect kind.
pub const NAME: &str = "correction-learner";

/// The heading the learned rules are listed under.
pub const HEADING: &str = "## Learned rules";

/// The longest rule kept.
pub const RULE_CHARS: usize = 200;

/// How much of each reply text and tool call the classifier reads.
const SHORT: usize = 300;

/// The tag of rig-basics' prompt section of the instruction files.
const PROJECT_SECTION: &str = "project_instructions";

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
            .init_resource::<RigHome>()
            .add_setting::<CorrectionLearnerConfig>(NAME)
            .add_observer(classify)
            .add_observer(propose)
            .add_observer(follow_successor::<Classifying>)
            .add_command(
                "rules",
                "Learned project rules: /rules, /rules rm N, /rules save [text], /rules dismiss",
                rules,
            );
        #[cfg(feature = "tui")]
        tui::add(app);
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

/// On an open proposal effect a front shows: the rule, and the agent and
/// turn it came from. It settles saved, dismissed or replaced.
#[derive(Component, Clone, Debug)]
pub struct Proposal {
    /// The rule.
    pub rule: String,
    /// The agent corrected.
    pub agent: Entity,
    /// The turn whose user message it came from ([`Turn::index`]).
    pub turn: u64,
}

/// The project's instruction file and the plugin's file of dismissed
/// rules.
struct Files {
    instructions: PathBuf,
    dismissed: PathBuf,
}

impl Files {
    /// The project's: its `AGENTS.md`, or the `CLAUDE.md` it has instead.
    fn of(root: &Path) -> Self {
        Self {
            instructions: instruction_file(root).unwrap_or_else(|| root.join("AGENTS.md")),
            dismissed: root.join(".rig").join("dismissed-rules.txt"),
        }
    }

    /// The rules under [`HEADING`].
    fn learned(&self) -> Vec<String> {
        learned(&fs::read_to_string(&self.instructions).unwrap_or_default())
    }

    /// The rules dismissed, one per line.
    fn dismissed(&self) -> Vec<String> {
        let text = fs::read_to_string(&self.dismissed).unwrap_or_default();
        text.lines().map(str::to_owned).collect()
    }

    /// Appends `rule` under [`HEADING`], made with the file if missing.
    fn save(&self, rule: &str) -> io::Result<()> {
        let text = fs::read_to_string(&self.instructions).unwrap_or_default();
        fs::write(&self.instructions, with_rule(&text, rule))
    }

    /// Removes learned rule `number`, counted from 1; whether there was one.
    fn remove(&self, number: usize) -> io::Result<Option<String>> {
        let text = fs::read_to_string(&self.instructions)?;
        let Some((kept, removed)) = without_rule(&text, number) else {
            return Ok(None);
        };
        fs::write(&self.instructions, kept)?;
        Ok(Some(removed))
    }

    /// Remembers `rule` as dismissed.
    fn dismiss(&self, rule: &str) -> io::Result<()> {
        if let Some(dir) = self.dismissed.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut lines = self.dismissed();
        lines.push(rule.to_owned());
        fs::write(&self.dismissed, lines.join("\n") + "\n")
    }

    /// Whether `rule` is learned or dismissed already, ignoring case and
    /// punctuation.
    fn knows(&self, rule: &str) -> bool {
        let rule = normalized(rule);
        self.learned()
            .iter()
            .chain(&self.dismissed())
            .any(|known| normalized(known) == rule)
    }
}

/// Asks the classifier about each message the user typed that went into a
/// turn of an agent that answered before.
fn classify(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&Incoming, &ForAgent, &DeliveredIn)>,
    turns: Query<&Turn>,
    conversations: Query<&Conversation>,
    (config, models, root): (Res<CorrectionLearnerConfig>, Res<Models>, Res<ProjectRoot>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((message, &ForAgent(agent), &DeliveredIn(turn))) = messages.get(delivered.entity) else {
        return;
    };
    if message.origin.kind != OriginKind::User {
        return;
    }
    let (Ok(turn), Ok(conversation)) = (turns.get(turn), conversations.get(agent)) else {
        return;
    };
    let Some(previous) = previous_reply(conversation) else {
        return;
    };
    let files = Files::of(&root.0);
    let listed = |rules: Vec<String>| match rules.is_empty() {
        true => "(none)".to_owned(),
        false => rules
            .iter()
            .map(|rule| format!("- {rule}"))
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let prompt = format!(
        "Learned rules:\n{}\n\nDismissed rules:\n{}\n\nThe agent's previous reply:\n{previous}\n\n\
         The user's message:\n{}",
        listed(files.learned()),
        listed(files.dismissed()),
        message.text.trim(),
    );
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
        Classifying {
            agent,
            turn: turn.index,
        },
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
        let Message::Assistant(reply) = message else {
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

/// Turns each classifier answer that is a new rule into a proposal: shown
/// in place of the one waiting, or, when nobody can answer, only logged.
fn propose(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &Classifying)>,
    mut waiting: Query<(&Proposal, &mut Decision), Without<Outcome>>,
    answerers: Answerers,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let Ok((outcome, &Classifying { agent, turn })) = calls.get(settled.entity) else {
        return;
    };
    let Some(rule) = outcome.completion().and_then(|reply| rule(&reply.text())) else {
        return;
    };
    let shown = waiting
        .iter()
        .any(|(shown, _)| normalized(&shown.rule) == normalized(&rule));
    if shown || Files::of(&root.0).knows(&rule) {
        return;
    }
    let kind = EffectKind::Custom {
        kind: NAME.into(),
        payload: serde_json::json!({ "rule": rule, "turn": turn }),
    };
    let request = EffectRequest {
        kind,
        handler: None,
    };
    if !answerers.can_answer(Some(agent)) {
        let mut logged = Decision::default();
        logged.decide(Outcome::custom("logged"));
        commands.spawn((Name::new("rule proposal"), request, logged));
        return;
    }
    for (_, mut decision) in &mut waiting {
        decision.decide(Outcome::custom("replaced"));
    }
    let proposal = Proposal { rule, agent, turn };
    commands.spawn((Name::new("rule proposal"), request, proposal));
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

/// Whether `line` starts a section.
fn is_heading(line: &str) -> bool {
    line.starts_with('#')
}

/// The lines of the [`HEADING`] section of `text`: from the heading to the
/// next heading.
fn section(lines: &[&str]) -> Option<std::ops::Range<usize>> {
    let heading = lines.iter().position(|line| line.trim() == HEADING)?;
    let after = heading + 1;
    let end = lines
        .iter()
        .skip(after)
        .position(|line| is_heading(line))
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

/// Sets rig-basics' prompt section of the instruction files from them
/// now, so the running agents' next requests have what changed.
fn reread(world: &mut World) {
    let home = RigHome::of(world);
    let Some(root) = world
        .get_resource::<ProjectRoot>()
        .map(|root| root.0.clone())
    else {
        return;
    };
    let text = Instructions::discover([home.root().to_path_buf()], &root).prompt_text();
    let mut sections = world.query_filtered::<&mut PromptSection, Without<SectionOf>>();
    for mut section in sections.iter_mut(world) {
        if section.tag == PROJECT_SECTION && section.text != text {
            section.text.clone_from(&text);
        }
    }
}

/// `/rules`: lists the learned rules; `/rules rm N` removes one,
/// `/rules save [text]` saves the proposal (as `text`, when given) and
/// `/rules dismiss` dismisses it.
fn rules(
    In(args): In<CommandArgs>,
    mut waiting: Query<(&Proposal, &mut Decision), Without<Outcome>>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let files = Files::of(&root.0);
    let (verb, rest) = args.args.split_once(' ').unwrap_or((&args.args, ""));
    let rest = rest.trim();
    let mut open = waiting
        .iter_mut()
        .find(|(_, decision)| !decision.is_decided());
    let path = files.instructions.display();
    let said = match verb {
        "" => {
            let learned = files.learned();
            let lines = learned
                .iter()
                .enumerate()
                .map(|(at, rule)| format!("{}. {rule}", at + 1));
            match learned.is_empty() {
                true => Ok(format!("No learned rules in {path}.")),
                false => Ok(format!(
                    "Learned rules in {path}:\n{}",
                    lines.collect::<Vec<_>>().join("\n")
                )),
            }
        }
        "rm" => match rest.parse::<usize>() {
            Ok(number) => match files.remove(number) {
                Ok(Some(rule)) => Ok(format!("Removed rule {number} from {path}: {rule}")),
                Ok(None) => Err(format!("There is no learned rule {number}.")),
                Err(why) => Err(format!("Cannot change {path}: {why}")),
            },
            Err(_) => Err("/rules rm needs the number /rules lists.".to_owned()),
        },
        "save" => {
            let proposed = open.as_ref().map(|(proposal, _)| proposal.rule.clone());
            let typed = (!rest.is_empty()).then(|| rest.to_owned());
            match typed.or(proposed) {
                None => Err("No rule is proposed; /rules save <text> saves one.".to_owned()),
                Some(rule) if rule.chars().count() > RULE_CHARS => {
                    Err(format!("A rule takes at most {RULE_CHARS} characters."))
                }
                Some(rule) => match files.save(&rule) {
                    Ok(()) => {
                        if let Some((_, decision)) = &mut open {
                            decision.decide(Outcome::custom(serde_json::json!({ "saved": rule })));
                        }
                        Ok(format!("Saved to {path}: {rule}"))
                    }
                    Err(why) => Err(format!("Cannot write {path}: {why}")),
                },
            }
        }
        "dismiss" => match &mut open {
            Some((proposal, decision)) => match files.dismiss(&proposal.rule) {
                Ok(()) => {
                    decision.decide(Outcome::custom("dismissed"));
                    Ok(format!("Dismissed: {}", proposal.rule))
                }
                Err(why) => Err(format!("Cannot remember the dismissal: {why}")),
            },
            None => Err("No rule is proposed.".to_owned()),
        },
        _ => Err("/rules takes nothing, rm N, save [text] or dismiss.".to_owned()),
    };
    match said {
        Ok(text) => {
            if matches!(verb, "rm" | "save") {
                commands.queue(reread);
            }
            notices.write(Notice::info(args.agent, text));
        }
        Err(text) => {
            notices.write(Notice::error(args.agent, text));
        }
    }
}
