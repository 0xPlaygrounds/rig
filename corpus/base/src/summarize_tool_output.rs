//! Big tool outputs are condensed by a cheap model before the agent's model
//! sees them. A Review system takes each tool call's result over
//! [`SummarizeOutputs::threshold_tokens`]: it saves the whole output to a
//! file under [`SummarizeOutputs::dir`], which the model can `read`, and
//! holds the call with a summary call of the turn ([`Summarizing`], a
//! [`ModelRequest`] on the configured cheap model, named `summarizing
//! output`). Once the summary is in, another Review system replaces the
//! call's result with it and lets the call go. When there is no cheap
//! model, or it fails or takes longer than
//! [`SummarizeOutputs::timeout_secs`], the model gets the output's head and
//! tail with a note instead. The summary's usage counts with the agent's
//! under the purpose `summarizer`. The default `shell` keeps whole
//! outputs while the plugin is loaded ([`ShellOutput`]), so the summary and
//! the saved file cover all of a command's output, not its last lines.

use std::path::{Path, PathBuf};

use rig_basics::coding_tools::ShellOutput;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::rig_tools::Read;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use serde::Deserialize;

/// Summarizes big tool outputs with a cheap model.
#[derive(Default)]
pub struct SummarizeToolOutputPlugin;

impl Plugin for SummarizeToolOutputPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .get_resource_or_init::<ShellOutput>()
            .set(None);
        app.add_setting::<SummarizeOutputs>("summarize_outputs")
            .init_resource::<ProjectRoot>()
            .add_observer(follow_successor::<Summarizing>)
            .configure_sets(AgentPass, SummarizeSystems.in_set(Stage::Review))
            .add_systems(
                AgentPass,
                (take_summaries, summarize).chain().in_set(SummarizeSystems),
            );
    }
}

/// The Review systems that replace big tool results with their summary or
/// a cut. A reviewer that must see a call's final output goes `.after` it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SummarizeSystems;

/// The `[summarize_outputs]` settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SummarizeOutputs {
    /// Outputs estimated above this many tokens are summarized.
    pub threshold_tokens: usize,
    /// The catalog model (`vendor/model`) that summarizes; without one,
    /// big outputs are cut to their head and tail.
    pub model: Option<String>,
    /// Tools whose outputs are never summarized.
    pub skip: Vec<String>,
    /// Seconds the summary may take before the output is cut instead.
    pub timeout_secs: u64,
    /// Where whole outputs are saved, as `call-<n>.txt`, under the project.
    pub dir: PathBuf,
}

impl Default for SummarizeOutputs {
    fn default() -> Self {
        Self {
            threshold_tokens: 8_000,
            model: None,
            skip: Vec::new(),
            timeout_secs: 30,
            dir: PathBuf::from(".rig/outputs"),
        }
    }
}

/// On a tool call whose output this plugin took: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Summarized;

/// The summary call of a big output, which holds the tool call.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// The tool call.
    pub call: Entity,
    /// What the tool returned.
    pub result: ToolResult,
    /// What the model is told about it.
    pub about: About,
}

/// A big output, as the model is told about it.
#[derive(Clone, Debug)]
pub struct About {
    /// The tool's name.
    pub tool: String,
    /// The output's estimated tokens.
    pub tokens: usize,
    /// The file holding it whole, under the project when it can be.
    pub path: PathBuf,
    /// Characters a cut keeps.
    pub budget: usize,
}

impl About {
    /// `result` with the summary in place of its output.
    fn summary(&self, result: &ToolResult, summary: &str) -> Outcome {
        let text = format!(
            "Summary of a {}-token `{}` output (full output: `{}`):\n\n{}",
            thousands(self.tokens),
            self.tool,
            self.path.display(),
            summary.trim()
        );
        Outcome::from(result.clone().with_output(ToolOutput::text(text)))
    }

    /// `result` with its output cut to its head and tail, the cut saying
    /// why and where the whole output is.
    fn truncated(&self, result: &ToolResult, why: &str) -> Outcome {
        let note = format!(
            "{why}; the whole {}-token output is in `{}`",
            thousands(self.tokens),
            self.path.display()
        );
        let text = result.output().as_text().unwrap_or_default();
        let cut = head_and_tail(text, self.budget, &note);
        Outcome::from(result.clone().with_output(ToolOutput::text(cut)))
    }
}

/// What the summarizer is told.
const PREAMBLE: &str = "You condense the output of a coding agent's tool call for the agent, \
which sees only what you write. Keep every error and warning message, file path and line \
number verbatim, then end with the output's last lines verbatim. Be brief, and add nothing \
the output does not say.";

/// Characters the cheap model is sent when its input limit is not known.
const MAX_INPUT_CHARS: usize = 400_000;

/// An estimate of `text`'s tokens: four bytes each.
fn tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Whether `call` reads a file under `dir`, such as a saved output.
fn reads_saved(call: &ToolCall, root: &Path, dir: &Path) -> bool {
    let path = call.function.arguments.get("path");
    call.function.name.as_str() == Read::NAME
        && path
            .and_then(|path| path.as_str())
            .is_some_and(|path| Local.join(root, Path::new(path)).starts_with(dir))
}

/// Takes each big tool result as it is reviewed: saves it whole, then
/// holds the call with a summary call on the cheap model, or cuts it when
/// there is none.
fn summarize(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &mut Outcome,
            &Effect,
            &ForAgent,
            Option<&CallOf>,
        ),
        (ReadyToSettle, ToolCalls, Without<Summarized>),
    >,
    config: Res<SummarizeOutputs>,
    root: Res<ProjectRoot>,
    models: Res<Models>,
    mut commands: Commands,
) {
    let dir = root.0.join(&config.dir);
    for (call, request, mut outcome, effect, &ForAgent(agent), turn) in &mut calls {
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let (Some(text), tool) = (
            result.output().as_text(),
            request.call.function.name.as_str(),
        ) else {
            continue;
        };
        let estimate = tokens(text);
        if estimate <= config.threshold_tokens
            || config.skip.iter().any(|skip| skip == tool)
            || reads_saved(&request.call, &root.0, &dir)
        {
            continue;
        }
        commands.entity(call).insert(Summarized);
        let saved = save(&dir, text);
        let path = saved.as_ref().map_or_else(|_| PathBuf::new(), Clone::clone);
        let about = About {
            tool: tool.to_owned(),
            tokens: estimate,
            path: path.strip_prefix(&root.0).unwrap_or(&path).to_owned(),
            budget: config.threshold_tokens.saturating_mul(4),
        };
        let connection = match (saved, &config.model) {
            (Err(why), _) => Err(format!("the output could not be saved ({why})")),
            (Ok(_), None) => Err("no summarizer model is configured".to_owned()),
            (Ok(_), Some(model)) => models.connect(model),
        };
        let connection = match connection {
            Ok(connection) => connection,
            Err(why) => {
                *outcome = about.truncated(result, &why);
                continue;
            }
        };
        // About three bytes a token, a guess on the safe side.
        let cap = connection
            .spec
            .input_limit()
            .and_then(|limit| usize::try_from(limit).ok())
            .map_or(MAX_INPUT_CHARS, |limit| limit.saturating_mul(3));
        let input = if text.len() > cap {
            head_and_tail(text, cap, "cut to fit the summarizer")
        } else {
            text.to_owned()
        };
        let prompt = format!(
            "The `{tool}` tool was called with {}. Its output, about {} tokens:\n\n{input}",
            request.call.function.raw_arguments(),
            thousands(estimate)
        );
        let summary = CompletionRequest::new(message::Message::user(prompt)).preamble(PREAMBLE);
        let mut summary = commands.spawn((
            Name::new("summarizing output"),
            Summarizing {
                call,
                result: result.clone(),
                about,
            },
            ModelRequest::new(summary).on(connection),
            Effect::with_parent_id(Some(effect.id())).with_purpose("summarizer"),
            ForAgent(agent),
            HoldOf(call),
        ));
        if let Some(&CallOf(turn)) = turn {
            summary.insert(CallOf(turn));
        }
        let (summary, secs) = (summary.id(), config.timeout_secs);
        let late = ErrorReport::new(ErrorKind::Timeout, format!("it took longer than {secs} s"));
        commands
            .delayed()
            .duration(Duration::from_secs(secs))
            .queue(move |world: &mut World| {
                if let Some(mut decision) = world.get_mut::<Decision>(summary) {
                    decision.decide(Decider::plugin("summarize-tool-output"), Outcome(Err(late)));
                }
            });
    }
}

/// Puts each finished summary in its tool call's result, or the output's
/// head and tail when the summarizer failed, and lets the call go.
fn take_summaries(
    summaries: Query<(Entity, &Summarizing, &Outcome), (ReadyToSettle, ModelCalls, With<HoldOf>)>,
    mut calls: Query<&mut Outcome, ToolCalls>,
    mut commands: Commands,
) {
    for (
        summary,
        Summarizing {
            call,
            result,
            about,
        },
        outcome,
    ) in &summaries
    {
        let text = outcome.completion().map(|response| response.text());
        let output = match (&outcome.0, text) {
            (Err(why), _) => about.truncated(result, &format!("the summarizer failed: {why}")),
            (Ok(_), Some(text)) if !text.trim().is_empty() => about.summary(result, &text),
            (Ok(_), _) => about.truncated(result, "the summarizer answered nothing"),
        };
        if let Ok(mut held) = calls.get_mut(*call) {
            *held = output;
        }
        commands.entity(summary).remove::<HoldOf>();
    }
}

/// Saves `text` whole in a new `call-<n>.txt` under `dir`.
fn save(dir: &Path, text: &str) -> std::io::Result<PathBuf> {
    use std::io::Write as _;
    let (path, mut file) = Spill(dir.to_owned()).create("call")?;
    file.write_all(text.as_bytes())?;
    Ok(path)
}

/// The first and last whole lines of `text` within `budget` characters in
/// all, with a note on the lines cut between them.
fn head_and_tail(text: &str, budget: usize, note: &str) -> String {
    let half = budget / 2;
    let head: String = text.chars().take(half).collect();
    let head = head
        .rsplit_once('\n')
        .map_or(head.as_str(), |(head, _)| head);
    let mut tail: Vec<char> = text.chars().rev().take(half).collect();
    tail.reverse();
    let tail: String = tail.into_iter().collect();
    let tail = tail
        .split_once('\n')
        .map_or(tail.as_str(), |(_, tail)| tail);
    let lines = |part: &str| part.lines().count();
    let cut = lines(text).saturating_sub(lines(head) + lines(tail));
    format!("{head}\n\n[… {cut} lines cut: {note} …]\n\n{tail}")
}

/// `n` with thousands separated by commas, such as `41,203`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests;
