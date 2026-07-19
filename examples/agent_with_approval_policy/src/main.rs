//! Policy-based (non-interactive) human-in-the-loop: approval *rules* decided up
//! front, evaluated per tool call by an `AgentHook` — no human prompt in the
//! loop. This mirrors the OpenAI Agents SDK's `needs_approval(fn)`, Vercel AI
//! SDK's `needsApproval(({input}) => ...)`, and LangGraph's `interrupt_on`
//! predicates: a person encodes the policy once, and the agent runs within it.
//!
//! The [`ApprovalPolicy`] hook fires on [`ToolCallEvent`] and returns:
//! - [`ToolCallAction::run`] to allow a tool that is on the safe allow-list, or a
//!   guarded tool whose arguments satisfy the rule;
//! - [`ToolCallAction::skip`] to **deny** otherwise — the denial reason is fed back to the
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
use rig::agent::tool::Tool;
use rig::agent::{AgentHook, HookContext, ToolCall as ToolCallEvent, ToolCallAction};
use rig::client::{AgentClientExt, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
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

    fn description(&self) -> String {
        "Search the web for a query (read-only).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "query": { "type": "string", "description": "Search query" } },
            "required": ["query"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

    fn description(&self) -> String {
        "Transfer funds to an account.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "to": { "type": "string", "description": "Destination account id" },
                "amount": { "type": "integer", "description": "Amount in whole dollars" }
            },
            "required": ["to", "amount"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl AgentHook for ApprovalPolicy {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        let tool_name = event.tool_name;
        if self.auto_approve.contains(tool_name) {
            println!("[policy] auto-approve `{tool_name}` (safe)");
            return ToolCallAction::run();
        }
        if tool_name == TransferFunds::NAME {
            let amount = serde_json::from_str::<serde_json::Value>(event.args)
                .ok()
                .and_then(|value| value.get("amount").and_then(|amount| amount.as_u64()));
            return match amount {
                Some(amount) if amount <= self.max_auto_transfer => {
                    println!(
                        "[policy] approve transfer ${amount} (<= ${})",
                        self.max_auto_transfer
                    );
                    ToolCallAction::run()
                }
                Some(amount) => ToolCallAction::skip(format!(
                    "denied by policy: transfers over ${} require human approval; ${amount} exceeds the limit",
                    self.max_auto_transfer
                )),
                None => {
                    ToolCallAction::skip("denied by policy: could not read the transfer amount")
                }
            };
        }
        ToolCallAction::skip(format!(
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
