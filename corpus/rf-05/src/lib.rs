//! `rig-wordcount`: a `word_count` tool that counts the words of a file in
//! the workspace, a gate that refuses it on a file over 1 MiB, each agent's
//! saved total of words counted, `/wc-reset`, and (with the default `tui`
//! feature) how the terminal view draws its calls.

use std::path::PathBuf;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ContextValue;
use rig_harness::rig_tools::workspace::{Local, Stat, Workspace};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The tool's name.
pub const TOOL: &str = "word_count";

/// Who decides the gate's refusals.
pub const GATE: &str = "rig-wordcount";

/// The largest file `word_count` may count, in bytes.
pub const MAX_BYTES: u64 = 1 << 20;

/// What the gate tells the model when it refuses a call.
pub const TOO_BIG: &str = "word_count counts files up to 1 MiB";

/// The plugin: the tool, its gate, the saved totals, `/wc-reset` and,
/// with the `tui` feature, its calls' look.
#[derive(Default)]
pub struct WordCountPlugin;

/// The plugin's Gate system, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WordCountSystems;

impl Plugin for WordCountPlugin {
    fn build(&self, app: &mut App) {
        let here = WordCount {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        };
        app.add_tool_per_call(here, ToolOptions::default(), |at| WordCount {
            workspace: at.workspace,
            base: Some(at.base),
            confine: at.confine,
        })
        .register_required_components::<Agent, WordsCounted>()
        .keeps_saved::<WordsCounted>()
        .add_observer(count)
        .add_command(
            "wc-reset",
            "Set this agent's total of words counted to 0",
            reset,
        )
        .configure_sets(AgentPass, WordCountSystems.in_set(Stage::Gate))
        .add_systems(AgentPass, refuse_big_files.in_set(WordCountSystems));
        #[cfg(feature = "tui")]
        app.add_systems(
            rig_tui::TranscriptRender,
            draw.in_set(rig_tui::RenderSystems::Render),
        );
    }
}

/// On an agent: how many words its `word_count` calls counted, kept with
/// the session.
#[derive(Component, Reflect, Default, Debug, Clone, Copy, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct WordsCounted(pub u64);

/// What a `word_count` call published beside its text: the words it
/// counted.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counted {
    /// The words of the file.
    pub words: u64,
}

impl ContextValue for Counted {
    const KEY: &'static str = "wordcount.counted";
}

/// The `word_count` tool: reads in `workspace`, a path relative to `base`.
pub struct WordCount {
    /// Where it reads.
    pub workspace: Arc<dyn Workspace>,
    /// What a relative path is relative to.
    pub base: Option<PathBuf>,
    /// Whether a path outside `base` is refused.
    pub confine: bool,
}

/// The arguments of `word_count`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// The file, relative to the working directory.
    pub path: String,
}

impl Tool for WordCount {
    const NAME: &'static str = TOOL;
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Count the words of a file. Use it rather than `wc` in `shell`.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    /// Reads the one file it names, so the gate knows which.
    fn footprint(&self, args: &Args) -> Touches {
        let place = self
            .workspace
            .place(self.base.as_deref(), self.confine, &args.path);
        place.map_or(Touches::Reads(Scope::Anything), |path| {
            Touches::Reads(Scope::Paths(vec![path]))
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Args,
    ) -> Result<String, ToolExecutionError> {
        let (workspace, base, confine) = (self.workspace.clone(), self.base.clone(), self.confine);
        let path = args.path.clone();
        let words = blocking(move || {
            let file = workspace.place(base.as_deref(), confine, &path)?;
            let text = workspace.read_text(&file)?;
            Ok::<_, ToolExecutionError>(text.split_whitespace().count() as u64)
        })
        .await?;
        context
            .insert_result(Counted { words })
            .map_err(|failure| ToolExecutionError::other(failure.to_string()))?;
        Ok(format!("{}: {words} words", args.path))
    }
}

/// Refuses a `word_count` call on a file over [`MAX_BYTES`] before it
/// runs. The workspace is asked on the frame only when it is this
/// machine's, as the main-thread rule allows.
fn refuse_big_files(
    mut calls: Query<(&ToolRequest, &Footprint, &mut Decision), (ReadyToStart, ToolCalls)>,
    workspace: Option<Res<ProjectWorkspace>>,
) {
    let Some(workspace) = workspace.filter(|it| it.0.host().is_none()) else {
        return;
    };
    for (request, footprint, mut decision) in &mut calls {
        let Touches::Reads(Scope::Paths(paths)) = &footprint.0 else {
            continue;
        };
        let big = |path: &PathBuf| matches!(workspace.0.stat(path), Ok(Stat::File { size, .. }) if size > MAX_BYTES);
        if request.call.function.name.as_str() == TOOL && paths.iter().any(big) {
            decision.decide(Decider::plugin(GATE), Outcome::refused(TOO_BIG));
        }
    }
}

/// Adds what each `word_count` call counted to its agent's total, once:
/// a restored call answered from what it left behind ([`Interrupted`])
/// published nothing and is not counted again.
fn count(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Published), (ToolCalls, Without<Interrupted>)>,
    mut totals: Query<&mut WordsCounted>,
) {
    let Ok((request, ForAgent(agent), published)) = calls.get(settled.entity) else {
        return;
    };
    let counted = published.0.get::<Counted>().ok().flatten();
    if let (true, Some(counted), Ok(mut total)) = (
        request.call.function.name.as_str() == TOOL,
        counted,
        totals.get_mut(*agent),
    ) {
        total.0 += counted.words;
    }
}

/// `/wc-reset`: the agent's total back to 0.
fn reset(
    In(args): In<CommandArgs>,
    mut totals: Query<&mut WordsCounted>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok(mut total) = totals.get_mut(args.agent) {
        total.0 = 0;
        notices.write(Notice::info(args.agent, "Words counted: 0"));
    }
}

/// Draws the `word_count` calls with [`render`].
#[cfg(feature = "tui")]
fn draw(mut batch: ResMut<rig_tui::RenderBatch>) {
    for (call, output) in batch.tools_mut(TOOL) {
        output.lines = render(&call.view());
    }
}

/// How the terminal view draws a `word_count` call: its file, then the
/// words it counted, or its result while it has no count (a refusal).
#[cfg(feature = "tui")]
pub fn render(call: &rig_tui::ToolCallView<'_>) -> Vec<rig_tui::ratatui::text::Line<'static>> {
    let mut lines = vec![call.header("Count", call.argument("path").unwrap_or_default())];
    let counted = call
        .facts
        .and_then(|facts| facts.get::<Counted>().ok().flatten());
    match counted {
        Some(Counted { words }) => {
            lines.push(format!("{}{words} words", rig_tui::OUTPUT_INDENT).into());
        }
        None => lines.extend(call.result_lines()),
    }
    lines
}

#[cfg(test)]
mod tests;
