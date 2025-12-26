use anyhow::Result;
use rig::agent::{CancelSignal, StreamingPromptHook, ToolCallHookAction, stream_to_stdout};
use rig::completion::{CompletionModel, ToolDefinition};
use rig::prelude::*;
use rig::providers;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("File operation error")]
struct FileError;

#[derive(Deserialize, Serialize)]
struct ReadFileHead;

impl Tool for ReadFileHead {
    const NAME: &'static str = "read_file_head";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file_head".to_string(),
            description: "Read the first line of test.txt using the head command".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[ReadFileHead] Executing: head -1 test.txt");
        let output = std::process::Command::new("head")
            .arg("-1")
            .arg("test.txt")
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail;

impl Tool for ReadFileTail {
    const NAME: &'static str = "read_file_tail";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file_tail".to_string(),
            description: "Read the last line of test.txt using the tail command".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[ReadFileTail] Executing: tail -1 test.txt");
        let output = std::process::Command::new("tail")
            .arg("-1")
            .arg("test.txt")
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Clone)]
struct StreamingPermissionHook {
    call_count: Arc<AtomicUsize>,
}

impl<M: CompletionModel> StreamingPromptHook<M> for StreamingPermissionHook {
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _args: &str,
        _cancel_sig: CancelSignal,
    ) -> ToolCallHookAction {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);

        if count == 0 {
            println!("✗ [Hook] Rejected '{}' (call #{})", tool_name, count + 1);
            ToolCallHookAction::Skip {
                reason: format!(
                    "Tool '{}' is currently unavailable. \
                     Please use 'read_file_tail' instead to read the file.",
                    tool_name
                ),
            }
        } else {
            println!("✓ [Hook] Approved '{}' (call #{})", tool_name, count + 1);
            ToolCallHookAction::Continue
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    std::fs::write("test.txt", "hello world\n")?;
    println!("✓ Created test.txt with content: 'hello world'");
    println!();

    let client = providers::openai::Client::from_env();

    let agent = client
        .agent(providers::openai::GPT_4O_MINI)
        .preamble("You are a helpful assistant that can read files using different methods.")
        .tool(ReadFileHead)
        .tool(ReadFileTail)
        .build();

    let hook = StreamingPermissionHook {
        call_count: Arc::new(AtomicUsize::new(0)),
    };

    println!("=== Starting Streaming Permission Control Demo ===");
    println!("Expected flow:");
    println!("1. Agent tries to read test.txt");
    println!("2. First tool call (likely read_file_head) gets rejected");
    println!("3. Agent receives rejection message suggesting read_file_tail");
    println!("4. Agent calls read_file_tail which gets approved");
    println!("5. Successfully reads 'hello world'");
    println!();

    let mut stream = agent
        .stream_prompt(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .multi_turn(5)
        .with_hook(hook)
        .await;

    let final_res = stream_to_stdout(&mut stream).await?;

    println!();
    println!("=== Final Response ===");
    println!("Token usage response: {usage:?}", usage = final_res.usage());
    println!(
        "Final text response: {message:?}",
        message = final_res.response()
    );

    std::fs::remove_file("test.txt")?;
    println!();
    println!("✓ Cleaned up test.txt");

    Ok(())
}
