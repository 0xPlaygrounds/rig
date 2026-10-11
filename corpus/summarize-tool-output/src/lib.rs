//! Big tool outputs are condensed by a cheap model before the agent's model
//! sees them. A tool call whose output is over
//! [`SummarizeOutputs::threshold_tokens`] has its whole output saved to a
//! file under [`SummarizeOutputs::dir`], which the model can `read`, and a
//! [`ModelRequest`] of the turn on the [`CheapModel`] writes the summary the
//! model gets in its place. When there is no cheap model, or it fails or
//! takes longer than [`SummarizeOutputs::timeout`], the model gets the
//! output's head and tail with a note instead.
//!
//! The kernel inserts a tool's [`ToolOutput`] on its call when the tool's
//! [`Running`] task ends, and the turn takes it at once. To get at the
//! output first, the plugin moves each tool call's task onto an entity of
//! its own ([`HeldFor`] the call), so the output lands there, and inserts
//! the call's [`ToolOutput`] itself once it decided what the model gets.

use std::path::{Path, PathBuf};

use rig_core::completion::{CompletionRequest, UsageTotals};
use rig_core::tool::ToolOutput as Content;
use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

/// Summarizes big tool outputs with a cheap model.
#[derive(Default)]
pub struct SummarizeToolOutputPlugin;

impl Plugin for SummarizeToolOutputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SummarizeOutputs>()
            .register_required_components::<Agent, SummarizerSpending>()
            .add_systems(Startup, connect_cheap_model)
            .add_observer(hold_output)
            .add_observer(take_output)
            .add_observer(take_summary)
            .add_observer(time_out);
    }
}

/// How big outputs are summarized.
#[derive(Resource, Clone, Debug)]
pub struct SummarizeOutputs {
    /// Outputs estimated above this many tokens are summarized.
    pub threshold_tokens: usize,
    /// The catalog model (`vendor/model`) that summarizes, connected at
    /// startup into the [`CheapModel`]; `None` truncates instead.
    pub model: Option<String>,
    /// Tools whose outputs are never summarized.
    pub skip: Vec<String>,
    /// How long the summary may take before the output is truncated
    /// instead.
    pub timeout: Duration,
    /// Where whole outputs are saved, as `call-<n>.txt`.
    pub dir: PathBuf,
}

impl Default for SummarizeOutputs {
    fn default() -> Self {
        Self {
            threshold_tokens: 8_000,
            model: None,
            skip: Vec::new(),
            timeout: Duration::from_secs(30),
            dir: PathBuf::from(".rig/outputs"),
        }
    }
}

/// The model that summarizes, or why there is none. Inserted at startup
/// from [`SummarizeOutputs::model`] unless the app inserted one.
#[derive(Resource, Clone)]
pub struct CheapModel(pub Result<Connection, String>);

/// What an agent's summaries used, a part of its total usage, saved with
/// the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct SummarizerSpending(pub UsageTotals);

/// The entity holding the task of the tool call it names: the call's
/// output lands here first. Despawned with the call, which cancels it.
#[derive(Component, Debug)]
#[relationship(relationship_target = Held)]
pub struct HeldFor(pub Entity);

/// The task of a tool call, held by the plugin.
#[derive(Component, Debug)]
#[relationship_target(relationship = HeldFor, linked_spawn)]
pub struct Held(Entity);

/// The summary call of a big output: a [`ModelRequest`] of the call's
/// turn, named `summarizing output` for views.
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// The tool call.
    pub call: Entity,
    /// What the tool returned.
    pub result: ToolResult,
    /// Its text.
    pub text: String,
    /// What the summary says about it.
    pub about: About,
}

/// A big output, as the model is told about it.
#[derive(Clone, Debug)]
pub struct About {
    /// The tool's name.
    pub tool: String,
    /// The output's estimated tokens.
    pub tokens: usize,
    /// The file holding it whole.
    pub path: PathBuf,
}

/// The summary call `entity` ran out of time.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct SummaryTimedOut {
    /// The summary call.
    pub entity: Entity,
}

/// What the summarizer is told.
const PREAMBLE: &str = "You condense the output of a coding agent's tool call for the agent, \
which sees only what you write. Keep every error and warning message, file path and line \
number verbatim, then end with the output's last lines verbatim. Be brief, and add nothing \
the output does not say.";

/// Characters the cheap model is sent when its window is not known.
const MAX_INPUT_CHARS: usize = 400_000;

/// An estimate of `text`'s tokens: four bytes each.
fn tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

fn connect_cheap_model(
    config: Res<SummarizeOutputs>,
    cheap: Option<Res<CheapModel>>,
    models: Res<Models>,
    mut commands: Commands,
) {
    if cheap.is_some() {
        return;
    }
    let connection = match &config.model {
        Some(model) => models.connect(model),
        None => Err("no summarizer model is configured".to_owned()),
    };
    commands.insert_resource(CheapModel(connection));
}

/// Whether `run` reads a file under `dir`, such as a saved output.
fn reads_saved(run: &ToolCallRun, dir: &Path) -> bool {
    let function = &run.call.function;
    let path = function
        .arguments
        .get("path")
        .and_then(|path| path.as_str());
    function.name.as_str() == "read"
        && path.is_some_and(|path| {
            let path = Path::new(path);
            match (path.canonicalize(), dir.canonicalize()) {
                (Ok(path), Ok(dir)) => path.starts_with(dir),
                _ => path.starts_with(dir),
            }
        })
}

/// Moves the task of a tool call that may be summarized onto an entity of
/// its own, so its output lands there.
fn hold_output(
    add: On<Add<Running>>,
    calls: Query<&ToolCallRun, With<CallOf>>,
    config: Res<SummarizeOutputs>,
    mut commands: Commands,
) {
    let call = add.entity;
    let Ok(run) = calls.get(call) else {
        return;
    };
    let name = run.call.function.name.as_str();
    if config.skip.iter().any(|skip| skip == name) || reads_saved(run, &config.dir) {
        return;
    }
    commands
        .entity(call)
        .queue(move |mut entity: EntityWorldMut| {
            if let Some(running) = entity.take::<Running>() {
                entity.world_scope(|world| {
                    world.spawn((running, HeldFor(call)));
                });
            }
        });
}

/// Takes a held output: a small one goes to its call as it is; a big one is
/// saved and summarized, or truncated when there is no cheap model.
fn take_output(
    add: On<Add<ToolOutput>>,
    held: Query<(&HeldFor, &ToolOutput)>,
    runs: Query<(&ToolCallRun, &CallOf)>,
    config: Res<SummarizeOutputs>,
    cheap: Option<Res<CheapModel>>,
    mut commands: Commands,
) {
    let Ok((&HeldFor(call), ToolOutput(result))) = held.get(add.entity) else {
        return;
    };
    commands.entity(add.entity).despawn();
    let (Ok((run, &CallOf(turn))), Some(text)) = (runs.get(call), result.output().as_text()) else {
        commands
            .entity(call)
            .try_insert_if_new(ToolOutput(result.clone()));
        return;
    };
    let estimate = tokens(text);
    if estimate <= config.threshold_tokens {
        commands
            .entity(call)
            .try_insert_if_new(ToolOutput(result.clone()));
        return;
    }
    let budget = config.threshold_tokens.saturating_mul(4);
    let tool = run.call.function.name.as_str().to_owned();
    let path = match save(&config.dir, text) {
        Ok(path) => path,
        Err(why) => {
            let note = format!("the whole output could not be saved: {why}");
            let cut = head_and_tail(text, budget, &note);
            let output = ToolOutput(result.clone().with_output(Content::text(cut)));
            commands.entity(call).try_insert_if_new(output);
            return;
        }
    };
    let about = About {
        tool,
        tokens: estimate,
        path,
    };
    let connection = match cheap.as_deref() {
        Some(CheapModel(Ok(connection))) => connection,
        Some(CheapModel(Err(why))) => {
            let output = truncated(result, text, &about, budget, why);
            commands.entity(call).try_insert_if_new(output);
            return;
        }
        None => {
            let why = "no summarizer model is connected";
            let output = truncated(result, text, &about, budget, why);
            commands.entity(call).try_insert_if_new(output);
            return;
        }
    };
    let window = connection.spec.context_window;
    let cap = window.map_or(MAX_INPUT_CHARS, |window| {
        usize::try_from(window).map_or(MAX_INPUT_CHARS, |window| window.saturating_mul(3))
    });
    let input = if text.len() > cap {
        head_and_tail(text, cap, "cut to fit the summarizer")
    } else {
        text.to_owned()
    };
    let prompt = format!(
        "The `{}` tool was called with {}. Its output, about {} tokens:\n\n{input}",
        about.tool,
        run.call.function.raw_arguments(),
        thousands(estimate)
    );
    let request = CompletionRequest::new(message::Message::user(prompt)).preamble(PREAMBLE);
    let summary = commands
        .spawn((
            Name::new("summarizing output"),
            Summarizing {
                call,
                result: result.clone(),
                text: text.to_owned(),
                about,
            },
            ModelRequest { request },
            connection.clone(),
            CallOf(turn),
        ))
        .id();
    commands
        .delayed()
        .duration(config.timeout)
        .trigger(SummaryTimedOut { entity: summary });
}

/// Takes a finished summary to its call, or the output's head and tail
/// when the summarizer failed.
fn take_summary(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Summarizing, &Done<ModelReply>)>,
    turns: Query<&TurnOf>,
    mut agents: Query<&mut SummarizerSpending>,
    config: Res<SummarizeOutputs>,
    mut commands: Commands,
) {
    let Ok((&CallOf(turn), summarizing, Done(reply))) = calls.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let Summarizing {
        call,
        result,
        text,
        about,
    } = summarizing;
    let summary = match reply {
        Ok(response) => {
            if let Ok(&TurnOf(agent)) = turns.get(turn)
                && let Ok(mut spent) = agents.get_mut(agent)
            {
                spent.0.record(&response.usage);
            }
            Some(response.text()).filter(|summary| !summary.trim().is_empty())
        }
        Err(_) => None,
    };
    let output = match (summary, reply) {
        (Some(summary), _) => {
            let summary = format!(
                "Summary of a {}-token `{}` output (full output: `{}`):\n\n{}",
                thousands(about.tokens),
                about.tool,
                about.path.display(),
                summary.trim()
            );
            ToolOutput(result.clone().with_output(Content::text(summary)))
        }
        (None, Err(report)) => {
            let why = format!("the summarizer failed: {report}");
            truncated(
                result,
                text,
                about,
                config.threshold_tokens.saturating_mul(4),
                &why,
            )
        }
        (None, Ok(_)) => {
            let why = "the summarizer answered nothing";
            truncated(
                result,
                text,
                about,
                config.threshold_tokens.saturating_mul(4),
                why,
            )
        }
    };
    commands.entity(*call).try_insert_if_new(output);
}

/// Truncates the output of a summary that took too long, and cancels it.
fn time_out(
    timed_out: On<SummaryTimedOut>,
    calls: Query<&Summarizing>,
    config: Res<SummarizeOutputs>,
    mut commands: Commands,
) {
    let Ok(summarizing) = calls.get(timed_out.entity) else {
        return;
    };
    let why = format!(
        "the summarizer took longer than {} s",
        config.timeout.as_secs()
    );
    let budget = config.threshold_tokens.saturating_mul(4);
    let output = truncated(
        &summarizing.result,
        &summarizing.text,
        &summarizing.about,
        budget,
        &why,
    );
    commands.entity(summarizing.call).try_insert_if_new(output);
    commands.entity(timed_out.entity).despawn();
}

/// Saves `text` whole in a new `call-<n>.txt` under `dir`.
fn save(dir: &Path, text: &str) -> std::io::Result<PathBuf> {
    use std::io::Write as _;
    let (path, mut file) = Spill(dir.to_owned()).create("call")?;
    file.write_all(text.as_bytes())?;
    Ok(path)
}

/// `result` with its output cut to its head and tail, the cut saying why
/// and where the whole output is.
fn truncated(
    result: &ToolResult,
    text: &str,
    about: &About,
    budget: usize,
    why: &str,
) -> ToolOutput {
    let note = format!(
        "{why}; the whole {}-token output is in `{}`",
        thousands(about.tokens),
        about.path.display()
    );
    let cut = head_and_tail(text, budget, &note);
    ToolOutput(result.clone().with_output(Content::text(cut)))
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
