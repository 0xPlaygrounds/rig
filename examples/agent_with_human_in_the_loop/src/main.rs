//! Human-in-the-loop (HITL) tool-call approval with `AgentHook`.
//!
//! An agent is given two side-effecting tools (`send_email`, `delete_file`).
//! Before *any* tool runs, an [`ApprovalHook`] pauses the run on the
//! [`StepEvent::ToolCall`] event, shows the human the tool name and arguments,
//! and waits for a decision on stdin. Each decision maps to an existing [`Flow`]
//! action — no special HITL machinery is required:
//!
//! | Human decision | `Flow` returned        | Effect                                                              |
//! |----------------|------------------------|--------------------------------------------------------------------|
//! | **approve**    | [`Flow::cont`]         | the tool executes as the model requested                           |
//! | **deny**       | [`Flow::skip`]         | the tool does *not* run; the reason becomes the tool result the model sees, so it can adapt |
//! | **edit**       | [`Flow::rewrite_args`] | the tool executes with human-supplied arguments instead            |
//! | **abort**      | [`Flow::terminate`]    | the whole run stops and surfaces the reason as an error            |
//!
//! Because `AgentHook::on_event` is `async`, the hook can simply `.await` the
//! human's input inline (here from stdin; in a real app this might be an HTTP
//! request to an approval UI, a Slack round-trip, or a database poll). The same
//! hook works unchanged on the streaming driver (`stream_prompt`).
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p agent_with_human_in_the_loop`

use anyhow::Result;
use rig::agent::{AgentHook, Flow, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

// ---------------------------------------------------------------------------
// Two side-effecting tools worth gating behind human approval.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("tool failed: {0}")]
struct ToolError(String);

#[derive(Deserialize)]
struct SendEmailArgs {
    to: String,
    subject: String,
    body: String,
}

struct SendEmail;

impl Tool for SendEmail {
    const NAME: &'static str = "send_email";
    type Error = ToolError;
    type Args = SendEmailArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Send an email to a recipient.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "Recipient email address" },
                    "subject": { "type": "string", "description": "Email subject line" },
                    "body": { "type": "string", "description": "Email body" }
                },
                "required": ["to", "subject", "body"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // A real implementation would hit an email API here.
        println!(
            "   📧 [send_email] -> {} (subject: {:?}, {} chars)",
            args.to,
            args.subject,
            args.body.len()
        );
        Ok(format!("email sent to {}", args.to))
    }
}

#[derive(Deserialize)]
struct DeleteFileArgs {
    path: String,
}

struct DeleteFile;

impl Tool for DeleteFile {
    const NAME: &'static str = "delete_file";
    type Error = ToolError;
    type Args = DeleteFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Permanently delete a file at the given path.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path of the file to delete" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // A real implementation would delete the file here.
        println!("   🗑️  [delete_file] -> {}", args.path);
        Ok(format!("deleted {}", args.path))
    }
}

// ---------------------------------------------------------------------------
// The HITL hook: pause on each tool call and ask a human.
// ---------------------------------------------------------------------------

/// Print `prompt`, then read one trimmed line from stdin without blocking the
/// async runtime (the blocking read runs on a dedicated thread).
async fn ask(prompt: &str) -> String {
    use std::io::Write;
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
        line
    })
    .await
    .unwrap_or_default()
    .trim()
    .to_string()
}

/// Gates every tool call behind interactive human approval.
struct ApprovalHook;

impl<M: CompletionModel> AgentHook<M> for ApprovalHook {
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        // Only the pre-execution tool-call event is a decision point; everything
        // else falls through untouched.
        let StepEvent::ToolCall {
            tool_name, args, ..
        } = event
        else {
            return Flow::cont();
        };

        println!("\n⏸  The agent wants to run a tool — your approval is required:");
        println!("     tool: {tool_name}");
        println!("     args: {args}");

        match ask("     [a]pprove / [d]eny / [e]dit args / a[b]ort run? ")
            .await
            .chars()
            .next()
        {
            // Approve (also the default on an empty line): run as requested.
            Some('a') | None => {
                println!("     → approved");
                Flow::cont()
            }
            // Deny: the tool does not run; the reason is fed back to the model as
            // the tool result so it can choose another course of action.
            Some('d') => {
                let reason = ask("     reason (shown to the model): ").await;
                let reason = if reason.is_empty() {
                    "denied by the human reviewer".to_string()
                } else {
                    reason
                };
                println!("     → denied");
                Flow::skip(reason)
            }
            // Edit: run the tool with human-supplied JSON arguments instead.
            Some('e') => {
                let raw = ask("     replacement JSON args: ").await;
                match serde_json::from_str::<serde_json::Value>(&raw) {
                    Ok(value) => {
                        println!("     → running with edited arguments");
                        Flow::rewrite_args(value)
                    }
                    Err(err) => {
                        println!("     ! invalid JSON ({err}); denying instead");
                        Flow::skip(format!(
                            "the reviewer tried to edit the arguments but supplied invalid JSON: {err}"
                        ))
                    }
                }
            }
            // Abort: stop the whole run.
            Some('b') => {
                println!("     → aborting the run");
                Flow::terminate("run aborted by the human reviewer")
            }
            Some(other) => {
                println!("     ! unrecognized choice '{other}', approving by default");
                Flow::cont()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You are an operations assistant. Use the available tools to carry out the user's \
             request. Call one tool at a time and wait for its result before the next step.",
        )
        .tool(SendEmail)
        .tool(DeleteFile)
        .build();

    let prompt = "Email alice@example.com a reminder that the budget review is at 3pm, \
                  then delete the stale file /tmp/old_report.csv.";
    println!("User: {prompt}");

    // Attach the approval hook for this run. It fires before every tool call;
    // the run pauses for your decision each time.
    let response = agent
        .prompt(prompt)
        .max_turns(10)
        .add_hook(ApprovalHook)
        .await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
