use anthropic::completion::{CLAUDE_3_5_HAIKU, CLAUDE_4_SONNET};
use anyhow::Result;
use rig::agent::{Agent, CancelSignal, ToolResultReviewer};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
enum ListDirError {
    #[error("Command not allowed: {0}")]
    NotAllowed(String),
    #[error("Execution failed")]
    ExecFailed,
}

#[derive(Deserialize)]
struct ListDirArgs {
    command: String,
    intention: String,
}

/// A tool that only allows `ls` and `tree` commands.
#[derive(Deserialize, Serialize)]
struct ListDir;

impl Tool for ListDir {
    const NAME: &'static str = "list_dir";
    type Error = ListDirError;
    type Args = ListDirArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "list_dir".to_string(),
            description: "List directory contents. Only 'ls' and 'tree' commands are allowed. Explain your intention.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run (only 'ls ...' or 'tree ...')"
                    },
                    "intention": {
                        "type": "string",
                        "description": "What you want to achieve"
                    }
                },
                "required": ["command", "intention"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let cmd = args.command.trim();
        println!("[list_dir] Command: {}, Intention: {}", cmd, args.intention);

        // Only allow ls and tree commands
        if !cmd.starts_with("ls") && !cmd.starts_with("tree") {
            return Err(ListDirError::NotAllowed(cmd.to_string()));
        }

        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .map_err(|_| ListDirError::ExecFailed)?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

/// A reviewer that uses a small model (via Agent) to critique tool results.
#[derive(Clone)]
struct SmallModelReviewer<M: CompletionModel> {
    agent: Agent<M>,
}

impl<M: CompletionModel> ToolResultReviewer for SmallModelReviewer<M> {
    async fn critique(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        args: &str,
        result: &str,
        _cancel_sig: CancelSignal,
    ) -> String {
        let parsed = serde_json::from_str::<serde_json::Value>(args).ok();
        let Some(intention) = parsed
            .as_ref()
            .and_then(|v| v.get("intention")?.as_str().map(String::from))
        else {
            return result.to_string();
        };
        let Some(command) = parsed
            .as_ref()
            .and_then(|v| v.get("command")?.as_str().map(String::from))
        else {
            return result.to_string();
        };

        let prompt = format!(
            "Review this directory listing command. Be concise (1-2 sentences).\n\n\
             Command: {command}\n\
             Intention: {intention}\n\
             Result:\n{result}\n\n\
             If using 'ls' to explore directory structure, suggest using 'tree' instead.\n\
             Reply: [GOOD/SUGGEST] <reason>"
        );

        match self.agent.prompt(&prompt).await {
            Ok(critique) => {
                println!("[reviewer] Critique: {}", critique);
                format!("{result}\n\n[Critique] {critique}")
            }
            Err(_) => result.to_string(),
        }
    }
}

fn setup_test_dir() -> std::path::PathBuf {
    let test_dir = std::env::temp_dir().join("rig_reviewer_test");
    std::fs::create_dir_all(test_dir.join("src")).unwrap();
    std::fs::create_dir_all(test_dir.join("docs")).unwrap();
    std::fs::create_dir_all(test_dir.join("tests")).unwrap();
    std::fs::write(test_dir.join("README.md"), "# Test Project").unwrap();
    std::fs::write(test_dir.join("src/main.rs"), "fn main() {}").unwrap();
    std::fs::write(test_dir.join("src/lib.rs"), "pub fn hello() {}").unwrap();
    std::fs::write(test_dir.join("docs/guide.md"), "# Guide").unwrap();
    std::fs::write(test_dir.join("tests/test1.rs"), "#[test] fn test() {}").unwrap();
    println!("Test directory created at: {}", test_dir.display());
    test_dir
}

fn cleanup_test_dir(test_dir: &std::path::Path) {
    if test_dir.exists() {
        std::fs::remove_dir_all(test_dir).unwrap();
        println!("Test directory cleaned up");
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let test_dir = setup_test_dir();

    let client = anthropic::Client::from_env();

    let agent = client
        .agent(CLAUDE_4_SONNET)
        .preamble(&format!(
            "You are a directory explorer. Use list_dir to explore directories. \
             The target directory is: {}",
            test_dir.display()
        ))
        .max_tokens(4096)
        .tool(ListDir)
        .build();

    // Reviewer uses a smaller/faster model
    let reviewer = SmallModelReviewer {
        agent: client.agent(CLAUDE_3_5_HAIKU).max_tokens(512).build(),
    };

    let response = agent
        .prompt(&format!(
            "Show me the complete structure of {}",
            test_dir.display()
        ))
        .multi_turn(5)
        .with_reviewer(reviewer)
        .await?;

    println!("Response: {}", response);

    cleanup_test_dir(&test_dir);

    Ok(())
}
