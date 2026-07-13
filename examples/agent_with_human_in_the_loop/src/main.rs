//! Human-in-the-loop (HITL) tool-call approval with `AgentHook`.
//!
//! An agent is given two side-effecting tools (`send_email`, `delete_file`).
//! Before *any* tool runs, an [`ApprovalHook`] pauses the run on the
//! [`ToolCallEvent`] event, shows the human the tool name and arguments,
//! and waits for a decision on stdin. Each decision maps to an existing
//! event-specific action — no special HITL machinery is required:
//!
//! | Human decision | Action returned                  | Effect                                                              |
//! |----------------|----------------------------------|--------------------------------------------------------------------|
//! | **approve**    | [`ToolCallAction::run`]          | the tool executes as the model requested                           |
//! | **deny**       | [`ToolCallAction::skip`]         | the tool does *not* run; the reason becomes the tool result the model sees, so it can adapt |
//! | **edit**       | [`ToolCallAction::rewrite`]      | the tool executes with human-supplied arguments instead            |
//! | **abort**      | [`ToolCallAction::stop`]         | the whole run stops and surfaces the reason as an error            |
//!
//! Because `AgentHook::on_tool_call` is `async`, the hook can simply `.await` the
//! human's input inline (here from stdin; in a real app this might be an HTTP
//! request to an approval UI, a Slack round-trip, or a database poll). The same
//! hook works unchanged on the streaming driver (`stream_prompt`).
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p agent_with_human_in_the_loop`

use anyhow::Result;
use rig::agent::{AgentHook, HookContext, ToolCall as ToolCallEvent, ToolCallAction};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

// ---------------------------------------------------------------------------
// Two side-effecting tools worth gating behind human approval.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SendEmailArgs {
    to: String,
    subject: String,
    body: String,
}

struct SendEmail;

impl Tool for SendEmail {
    const NAME: &'static str = "send_email";
    type Args = SendEmailArgs;
    type Output = String;

    fn description(&self) -> String {
        "Send an email to a recipient.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "to": { "type": "string", "description": "Recipient email address" },
                "subject": { "type": "string", "description": "Email subject line" },
                "body": { "type": "string", "description": "Email body" }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
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
    type Args = DeleteFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Permanently delete a file at the given path.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute path of the file to delete" }
            },
            "required": ["path"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        // A real implementation would delete the file here.
        println!("   🗑️  [delete_file] -> {}", args.path);
        Ok(format!("deleted {}", args.path))
    }
}

// ---------------------------------------------------------------------------
// The HITL hook: pause on each tool call and ask a human.
// ---------------------------------------------------------------------------

/// Print `prompt`, then read one trimmed line from stdin without blocking the
/// async runtime (the blocking read runs on a dedicated thread). Returns `None`
/// on EOF / closed stdin (e.g. piped `< /dev/null`, Ctrl-D) or a read error —
/// the caller treats "no input" as fail-closed, never as approval.
async fn ask(prompt: &str) -> Option<String> {
    use std::io::Write;
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let line = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None, // EOF / closed stdin / read error
            Ok(_) => Some(line),
        }
    })
    .await
    .ok()
    .flatten()?;
    Some(line.trim().to_string())
}

/// Gates every tool call behind interactive human approval.
///
/// The gate is **fail-closed**: anything other than an explicit approval (an
/// empty line, an unrecognized choice, or closed stdin) denies or aborts rather
/// than running the tool. An approval prompt that guards side-effecting tools
/// must never run them on ambiguous input — and note that the prompt is a UX
/// affordance, not a security boundary; real authorization belongs inside the
/// tool itself.
struct ApprovalHook;

impl<M: CompletionModel> AgentHook<M> for ApprovalHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        let tool_name = event.tool_name;
        let args = event.args;

        println!("\n⏸  The agent wants to run a tool — your approval is required:");
        println!("     tool: {tool_name}");
        println!("     args: {args}");

        // No input at all (closed stdin) → abort the run; there is no reviewer.
        let Some(choice) = ask("     [a]pprove / [d]eny / [e]dit args / a[b]ort run? ").await
        else {
            println!("     → no input (stdin closed); aborting (fail-closed)");
            return ToolCallAction::stop("no reviewer input available (stdin closed)");
        };

        // Match the whole (lowercased) answer, accepting either the hotkey or the
        // full word, so typing "abort" can never be mistaken for "approve".
        match choice.to_ascii_lowercase().as_str() {
            "a" | "approve" => {
                println!("     → approved");
                ToolCallAction::run()
            }
            // Deny: the tool does not run; the reason is fed back to the model as
            // the tool result so it can choose another course of action.
            "d" | "deny" | "n" | "no" => {
                let reason = ask("     reason (shown to the model): ")
                    .await
                    .filter(|r| !r.is_empty())
                    .unwrap_or_else(|| "denied by the human reviewer".to_string());
                println!("     → denied");
                ToolCallAction::skip(reason)
            }
            // Edit: run the tool with human-supplied JSON arguments instead.
            "e" | "edit" => {
                match ask("     replacement JSON args (single line): ")
                    .await
                    .as_deref()
                    .map(serde_json::from_str::<serde_json::Value>)
                {
                    Some(Ok(value)) => {
                        println!("     → running with edited arguments");
                        ToolCallAction::rewrite(value)
                    }
                    other => {
                        println!("     ! no valid JSON ({other:?}); denying instead");
                        ToolCallAction::skip(
                            "the reviewer tried to edit the arguments but supplied no valid JSON",
                        )
                    }
                }
            }
            // Abort: stop the whole run.
            "b" | "abort" | "q" | "quit" => {
                println!("     → aborting the run");
                ToolCallAction::stop("run aborted by the human reviewer")
            }
            // Fail closed: empty or unrecognized input denies rather than runs.
            "" => {
                println!("     → empty input; denying (fail-closed)");
                ToolCallAction::skip("denied: the reviewer gave no decision")
            }
            other => {
                println!("     ! unrecognized choice '{other}'; denying (fail-closed)");
                ToolCallAction::skip(format!("denied: unrecognized reviewer input '{other}'"))
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
