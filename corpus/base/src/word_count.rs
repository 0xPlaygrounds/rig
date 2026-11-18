//! Word count tool. `word_count {path}` answers how many words a file of the project holds. It is
//! a per-call tool: each call is built for the place it works in, so a relative path is read from
//! the calling agent's working directory, else the project root, in the session's workspace.

use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_tools::workspace::{Local, Root, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

/// Adds the read-only `word_count` tool.
#[derive(Default)]
pub struct WordCountPlugin;

impl Plugin for WordCountPlugin {
    fn build(&self, app: &mut App) {
        let tool = WordCount(Root::from(Arc::new(Local) as Arc<dyn Workspace>));
        app.add_tool_per_call(tool, ToolOptions::default());
    }
}

/// `word_count`, reading from where its call works.
#[derive(Clone)]
struct WordCount(Root);

impl PerCallTool for WordCount {
    fn at(&self, place: CallPlace) -> Self {
        Self(place.into())
    }
}

/// The arguments of `word_count`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    /// The file, relative to the project.
    path: String,
}

impl Tool for WordCount {
    const NAME: &'static str = "word_count";
    type Args = Args;
    type Output = usize;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Count the words of a file: runs of non-whitespace. Use it rather than `wc -w` in `shell`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    async fn call(&self, _: &mut ToolContext, args: Args) -> Result<usize, ToolExecutionError> {
        let root = self.0.clone();
        blocking(move || {
            let path = root.place(&args.path)?;
            let text = root.workspace.read_text(&path)?;
            Ok(text.split_whitespace().count())
        })
        .await
    }
}

#[cfg(test)]
mod tests;
