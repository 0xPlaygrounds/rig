//! Policy-based (non-interactive) human-in-the-loop: approval *rules* decided up
//! front, evaluated per tool call by an `AgentHook` — no human prompt in the
//! loop. This mirrors the OpenAI Agents SDK's `needs_approval(fn)`, Vercel AI
//! SDK's `needsApproval(({input}) => ...)`, and LangGraph's `interrupt_on`
//! predicates: a person encodes the policy once, and the agent runs within it.
//!
//! The [`ApprovalPolicy`] hook fires on [`StepEvent::ToolCall`] and returns:
//! - [`Flow::cont`] to allow a tool that is on the safe allow-list, or a guarded
//!   tool whose arguments satisfy the rule;
//! - [`Flow::skip`] to **deny** otherwise — the denial reason is fed back to the
//!   model as the tool result, so it can adjust (e.g. transfer a smaller amount
//!   or ask the user) rather than the run simply failing.
//!
//! The policy is **fail-closed**: any tool not explicitly allowed is denied. As
//! always, this is guardrail/UX logic, not a security boundary — enforce real
//! authorization inside the tool.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p agent_with_approval_policy`

use std::collections::HashSet;

use anyhow::Result;
use rig::agent::{AgentHook, Flow, HookContext, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[error("tool failed: {0}")]
struct ToolError(String);

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

struct SearchWeb;

impl Tool for SearchWeb {
    const NAME: &'static str = "search_web";
    type Error = ToolError;
    type Args = SearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search the web for a query (read-only).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "Search query" } },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("   🔎 [search_web] -> {}", args.query);
        Ok(format!("top result for '{}': $1000 is plenty.", args.query))
    }
}

#[derive(Deserialize)]
struct TransferArgs {
    to: String,
    amount: u64,
}

struct TransferFunds;

impl Tool for TransferFunds {
    const NAME: &'static str = "transfer_funds";
    type Error = ToolError;
    type Args = TransferArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Transfer funds to an account.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "Destination account id" },
                    "amount": { "type": "integer", "description": "Amount in whole dollars" }
                },
                "required": ["to", "amount"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("   🏦 [transfer_funds] -> ${} to {}", args.amount, args.to);
        Ok(format!("transferred ${} to {}", args.amount, args.to))
    }
}

// ---------------------------------------------------------------------------
// The approval policy, evaluated on every tool call.
// ---------------------------------------------------------------------------

struct ApprovalPolicy {
    /// Tools allowed to run unconditionally (read-only / low risk).
    auto_approve: HashSet<&'static str>,
    /// Transfers at or below this amount are auto-approved; above it they are
    /// denied (a real app would route those to a human instead).
    max_auto_transfer: u64,
}

impl<M: CompletionModel> AgentHook<M> for ApprovalPolicy {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        let StepEvent::ToolCall {
            tool_name, args, ..
        } = event
        else {
            return Flow::cont();
        };

        if self.auto_approve.contains(tool_name) {
            println!("[policy] auto-approve `{tool_name}` (safe)");
            return Flow::cont();
        }

        if tool_name == TransferFunds::NAME {
            let amount = serde_json::from_str::<serde_json::Value>(args)
                .ok()
                .and_then(|v| v.get("amount").and_then(|a| a.as_u64()));
            return match amount {
                Some(a) if a <= self.max_auto_transfer => {
                    println!(
                        "[policy] approve transfer ${a} (<= ${})",
                        self.max_auto_transfer
                    );
                    Flow::cont()
                }
                Some(a) => {
                    println!(
                        "[policy] DENY transfer ${a} (over ${})",
                        self.max_auto_transfer
                    );
                    Flow::skip(format!(
                        "denied by policy: transfers over ${} require human approval; \
                         ${a} exceeds the limit",
                        self.max_auto_transfer
                    ))
                }
                None => Flow::skip("denied by policy: could not read the transfer amount"),
            };
        }

        // Fail closed: anything not explicitly allowed is denied.
        println!("[policy] DENY `{tool_name}` (not on the approved list)");
        Flow::skip(format!(
            "denied by policy: `{tool_name}` is not on the approved tool list"
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You are a banking assistant. Use the tools to carry out the user's request. \
             If a tool is denied by policy, explain the limit to the user instead of retrying.",
        )
        .tool(SearchWeb)
        .tool(TransferFunds)
        .build();

    let policy = ApprovalPolicy {
        auto_approve: HashSet::from([SearchWeb::NAME]),
        max_auto_transfer: 1000,
    };

    let prompt = "Look up how much I should send, then transfer $5000 to account B-2.";
    println!("User: {prompt}\n");

    // The transfer of $5000 exceeds the $1000 policy limit, so it is denied and
    // the reason is handed to the model, which should explain rather than retry.
    let response = agent.prompt(prompt).max_turns(10).add_hook(policy).await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
