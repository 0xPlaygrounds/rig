//! Steering an agent on *why* a tool failed, using structured tool outcomes.
//!
//! Rig delivers each tool result to hooks as a structured
//! [`ToolExecutionResult`](rig::tool::ToolExecutionResult): the model-visible
//! text **plus** a machine-readable [`ToolOutcome`](rig::tool::ToolOutcome). A
//! hook can therefore branch on *why* a tool failed — a timeout vs. a 404 — with
//! no string parsing.
//!
//! This example wires up:
//!
//! - `HttpFetch` — a tool whose [`Tool::classify_error`](rig::tool::Tool::classify_error)
//!   maps its own error variants onto standard
//!   [`ToolFailureKind`](rig::tool::ToolFailureKind)s (timeout, not-found, …). A
//!   URL containing `slow` times out; one containing `missing` returns 404;
//!   anything else succeeds.
//! - `OutcomePolicy` — a hook that **counts timeouts in the run scratchpad and
//!   terminates** after a threshold, while letting a **404 flow back to the model
//!   as recoverable feedback** so it can try another path.
//!
//! This is the motivating contrast: repeated HTTP timeouts should abort the
//! agent; a 404 should not. `main` drives the **404-recovery** half (it prompts a
//! `missing` URL and the run continues); the timeout-abort branch is implemented
//! and ready but not exercised by this single prompt — reaching the threshold
//! would require the model to repeatedly fetch a `slow` URL, which is left out to
//! keep the example deterministic. Point the prompt at a `slow` URL to see it fire.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, Flow, HookContext, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::{Tool, ToolFailure, ToolFailureKind, ToolOutcome};

// ---------------------------------------------------------------------------
// A tool that classifies its own failures into structured kinds.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct FetchArgs {
    url: String,
}

#[derive(Debug, thiserror::Error)]
enum FetchError {
    #[error("request to {url} timed out")]
    Timeout { url: String },
    #[error("404 Not Found: {url}")]
    NotFound { url: String },
}

struct HttpFetch;

impl Tool for HttpFetch {
    const NAME: &'static str = "http_fetch";
    type Error = FetchError;
    type Args = FetchArgs;
    type Output = String;

    async fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch a URL and return its body. URLs containing 'slow' time out; \
                          URLs containing 'missing' return HTTP 404."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch" }
                },
                "required": ["url"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.url.contains("slow") {
            Err(FetchError::Timeout { url: args.url })
        } else if args.url.contains("missing") {
            Err(FetchError::NotFound { url: args.url })
        } else {
            Ok(format!("200 OK: fetched {}", args.url))
        }
    }

    // Map the tool's own error variants onto standard failure kinds — no string
    // parsing anywhere downstream. Hooks match on `ToolFailureKind`, not text.
    fn classify_error(&self, error: &Self::Error) -> ToolFailure {
        match error {
            FetchError::Timeout { .. } => ToolFailure::timeout(error.to_string()),
            FetchError::NotFound { .. } => {
                ToolFailure::not_found(error.to_string()).with_http_status(404)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A hook that steers the run on the structured outcome.
// ---------------------------------------------------------------------------

/// Run-scoped timeout tally, kept in the shared [`HookContext::scratchpad`].
#[derive(Clone, Default)]
struct TimeoutCount(usize);

/// Terminates the run after `max_timeouts` tool timeouts; lets a 404 continue.
struct OutcomePolicy {
    max_timeouts: usize,
}

impl<M: CompletionModel> AgentHook<M> for OutcomePolicy {
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        let StepEvent::ToolResult {
            tool_name,
            result,
            outcome,
            ..
        } = event
        else {
            return Flow::cont();
        };

        // Repeated timeouts should abort the agent: count them in the scratchpad
        // and terminate once the budget is exhausted.
        if outcome.is_error_kind(ToolFailureKind::Timeout) {
            let count = ctx.scratchpad().update(|c: &mut TimeoutCount| {
                c.0 += 1;
                c.0
            });
            println!(
                "[policy] {tool_name} timed out ({count}/{})",
                self.max_timeouts
            );
            if count >= self.max_timeouts {
                return Flow::terminate(format!("aborting after {count} tool timeouts"));
            }
            return Flow::cont();
        }

        // A 404 is not fatal: the model sees the error text as `result` and can
        // recover by trying another URL, so we just observe and continue.
        if outcome.is_error_kind(ToolFailureKind::NotFound) {
            let status = outcome
                .failure()
                .and_then(|failure| failure.http_status)
                .unwrap_or(404);
            println!("[policy] {tool_name} returned {status}; letting the model recover: {result}");
            return Flow::cont();
        }

        match outcome {
            ToolOutcome::Success => println!("[policy] {tool_name} succeeded: {result}"),
            ToolOutcome::Error(failure) => {
                println!(
                    "[policy] {tool_name} failed ({}): {}",
                    failure.kind, failure.message
                )
            }
            ToolOutcome::Skipped => println!("[policy] {tool_name} was skipped"),
            ToolOutcome::Denied => println!("[policy] {tool_name} was denied"),
            // `ToolOutcome` is `#[non_exhaustive]`; tolerate future variants.
            _ => println!(
                "[policy] {tool_name} finished with outcome {}",
                outcome.as_str()
            ),
        }
        Flow::cont()
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
            "You are a web assistant. Use the `http_fetch` tool to retrieve any URL the user \
             mentions. If a fetch fails, tell the user what went wrong.",
        )
        .tool(HttpFetch)
        .build();

    // The 404 path: the model fetches a missing URL, the tool reports a
    // structured `NotFound` outcome, the policy lets it flow back as feedback,
    // and the model recovers instead of the run aborting.
    let response = agent
        .prompt("Fetch https://example.com/missing-page and tell me what happened.")
        .max_turns(5)
        .add_hook(OutcomePolicy { max_timeouts: 3 })
        .await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
