use anyhow::Result;
use rig::agent::{
    AgentHook, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
};
use rig::completion::Prompt;
use rig::prelude::*;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::support::{assert_nonempty_response, collect_stream_observation};

use super::super::cassette_support::*;

const TEST_CONTENT: &str = "hello world\n";

/// A per-test scratch file.
///
/// Both cells in this module used to write and then delete `test.txt` in the
/// process's working directory. Run concurrently — which is what `cargo test`
/// does by default — one cell's `Drop` deleted the file the other was midway
/// through reading, so `head`/`tail` returned nothing, the tool result changed,
/// and the recorded request body stopped matching. It reproduced 4 runs out of
/// 4 in the full suite and passed 4 out of 4 in isolation, which is the
/// signature of shared state rather than flakiness. Both cells were `#[ignore]`d
/// until now, so the collision had never had a chance to fire.
///
/// The file name stays out of the request: the tool *descriptions* still say
/// "test.txt", so the recorded bodies are unaffected by giving each cell its own
/// path.
struct ScratchFile {
    path: std::path::PathBuf,
}

impl ScratchFile {
    fn new(tag: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "rig-permission-control-{tag}-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, TEST_CONTENT)?;
        Ok(Self { path })
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("File operation error")]
struct FileError;

#[derive(Deserialize, Serialize)]
struct ReadFileHead {
    path: std::path::PathBuf,
}

impl Tool for ReadFileHead {
    const NAME: &'static str = "read_file_head";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the first line of test.txt using the head command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("head")
            .arg("-1")
            .arg(&self.path)
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail {
    path: std::path::PathBuf,
}

impl Tool for ReadFileTail {
    const NAME: &'static str = "read_file_tail";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the last line of test.txt using the tail command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("tail")
            .arg("-1")
            .arg(&self.path)
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Clone)]
struct PermissionHook {
    call_count: Arc<AtomicUsize>,
    last_result: Arc<Mutex<Option<String>>>,
}

impl AgentHook for PermissionHook {
    async fn on_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolCallEvent<'_>,
    ) -> ToolCallAction {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if count == 0 {
            ToolCallAction::skip(format!(
                "Tool '{}' is currently unavailable. Please use 'read_file_tail' instead to read the file.",
                event.tool_name
            ))
        } else {
            ToolCallAction::run()
        }
    }

    async fn on_tool_result(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let normalized = event.presentation.render();
        *self.last_result.lock().expect("lock last_result") = Some(normalized);
        ToolResultAction::keep()
    }
}

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_llamacpp_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let scratch = ScratchFile::new("prompt")?;

            let agent = client
                .clone()
                .agent(CASSETTE_MODEL)
                .preamble(
                    "You are a helpful assistant that can read files using different methods.",
                )
                .tool(ReadFileHead {
                    path: scratch.path.clone(),
                })
                .tool(ReadFileTail {
                    path: scratch.path.clone(),
                })
                .build();

            let call_count = Arc::new(AtomicUsize::new(0));
            let last_result = Arc::new(Mutex::new(None));
            let hook = PermissionHook {
                call_count: call_count.clone(),
                last_result: last_result.clone(),
            };

            let response = agent
                .prompt(
                    "Use the available tools to read test.txt now. \
                 Do not ask any follow-up questions; just read the file and report its content.",
                )
                .max_turns(5)
                .add_hook(hook)
                .await?;

            assert_nonempty_response(&response);
            let last = last_result.lock().expect("lock last_result").clone();
            if let Some(last) = last {
                anyhow::ensure!(last == "hello world");
            }
            anyhow::ensure!(call_count.load(Ordering::SeqCst) >= 1);
            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_llamacpp_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| async move {
            let scratch = ScratchFile::new("streaming")?;

            let agent = client
                .clone()
                .agent(CASSETTE_MODEL)
                .preamble(
                    "You are a helpful assistant that can read files using different methods.",
                )
                .tool(ReadFileHead {
                    path: scratch.path.clone(),
                })
                .tool(ReadFileTail {
                    path: scratch.path.clone(),
                })
                .build();

            let call_count = Arc::new(AtomicUsize::new(0));
            let last_result = Arc::new(Mutex::new(None));
            let hook = PermissionHook {
                call_count: call_count.clone(),
                last_result: last_result.clone(),
            };

            let mut stream = agent
                .stream_prompt(
                    "Use the available tools to read test.txt now. \
                 Do not ask any follow-up questions; just read the file and report its content.",
                )
                .max_turns(5)
                .add_hook(hook)
                .await;

            let observation = collect_stream_observation(&mut stream).await;
            anyhow::ensure!(
                observation.errors.is_empty(),
                "streaming permission control produced errors: {:?}",
                observation.errors
            );
            anyhow::ensure!(
                observation.got_final_response,
                "stream should yield a final response; events: {:?}",
                observation.events
            );
            anyhow::ensure!(
                observation.tool_results >= 1,
                "expected at least one streamed tool-result event, got {}; events: {:?}",
                observation.tool_results,
                observation.events
            );
            anyhow::ensure!(
                observation
                    .tool_calls
                    .iter()
                    .any(|tool_name| tool_name == ReadFileHead::NAME),
                "expected stream to include the skipped read_file_head tool call, got {:?}",
                observation.tool_calls
            );

            let last = last_result.lock().expect("lock last_result").clone();
            let final_response = observation
                .final_response_text
                .as_deref()
                .unwrap_or_default();
            assert_nonempty_response(final_response);
            if let Some(last) = last {
                anyhow::ensure!(last == "hello world");
            }
            anyhow::ensure!(call_count.load(Ordering::SeqCst) >= 1);

            Ok(())
        },
    )
    .await
}
