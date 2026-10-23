//! Cap a run's total tokens and tool calls with one hook, without credentials.
//!
//! `on_outcome` sees the usage of every model response and `on_dispatch` sees
//! each tool call before it runs. The counts live in the run's scratchpad, so
//! every prompt starts from zero even though the agent shares one hook.
//!
//! The token check runs when a turn finishes, so a run can overshoot by that
//! turn. Patching `max_tokens` down from `on_completion_call` bounds the
//! overshoot's output; the next request's input still counts in full.
//!
//! ```not_rust
//! cargo run -p rig-agent --example run_budget
//! ```

use std::convert::Infallible;

use anyhow::Result;
use rig_agent::{
    AgentBuilder,
    agent::{
        AgentHook, DispatchAction, DispatchEvent, HookContext, ModelTurnAction, ModelTurnFinished,
        OutcomeAction, OutcomeEvent,
    },
    completion::Usage,
    tool::{Tool, ToolContext},
};
use rig_core::effect::EffectKind;
use rig_core::message::{AssistantContent, ToolName};
use rig_core::test_utils::{MockCompletionModel, MockTurn};

/// What the run has used so far, kept in its scratchpad.
#[derive(Clone, Default)]
struct Used {
    tokens: u64,
    tool_calls: u64,
}

/// Stops the run past `max_tokens`, and skips tool calls past
/// `max_tool_calls` so the model answers with what it has.
struct RunBudget {
    max_tokens: u64,
    max_tool_calls: u64,
}

impl AgentHook for RunBudget {
    /// Counts every model response, a retried or repaired attempt included.
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if let Some(response) = event.completion() {
            let usage = response.usage;
            let tokens = usage
                .total_tokens
                .unwrap_or(usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0));
            ctx.scratchpad()
                .update(|used: &mut Used| used.tokens += tokens);
        }
        OutcomeAction::proceed()
    }

    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let used = ctx.scratchpad().get::<Used>().unwrap_or_default().tokens;
        if used > self.max_tokens {
            return ModelTurnAction::stop(format!(
                "token budget: {used} of {} used",
                self.max_tokens
            ));
        }
        ModelTurnAction::continue_run()
    }

    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if !matches!(event.kind, EffectKind::ToolCall { .. }) {
            return DispatchAction::proceed();
        }
        let calls = ctx.scratchpad().update(|used: &mut Used| {
            used.tool_calls += 1;
            used.tool_calls
        });
        if calls > self.max_tool_calls {
            // `DispatchAction::stop` would end the run instead.
            return DispatchAction::skip(format!(
                "tool-call budget of {} reached",
                self.max_tool_calls
            ));
        }
        DispatchAction::proceed()
    }
}

#[derive(Clone)]
struct Search;

impl Tool for Search {
    const NAME: &'static str = "search";
    type Args = serde_json::Value;
    type Output = &'static str;
    type Error = Infallible;

    fn description(&self) -> String {
        "Search the web".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _: &mut ToolContext, _: Self::Args) -> Result<Self::Output, Infallible> {
        Ok("a result")
    }
}

fn search(id: &str) -> MockTurn {
    MockTurn::tool_call(id, "search", serde_json::json!({}))
        .with_usage(Usage::new().total_tokens(600))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Two 600-token turns against a 1,000-token budget: the second stops the run.
    let model =
        MockCompletionModel::from_turns([search("s1"), search("s2"), MockTurn::text("done")]);
    let budget = RunBudget {
        max_tokens: 1_000,
        max_tool_calls: 10,
    };
    let agent = AgentBuilder::new(model)
        .tool(Search)
        .add_hook(budget)
        .build();
    match agent.prompt("research").max_turns(5).await {
        Ok(answer) => println!("tokens: answered {answer}"),
        Err(error) => println!("tokens: {error}"),
    }

    // Three calls in one turn against a budget of two: the third is skipped
    // and the model answers with the two results it got.
    let name = ToolName::new("search")?;
    let calls = ["s1", "s2", "s3"]
        .map(|id| AssistantContent::tool_call(id, name.clone(), serde_json::json!({})));
    let model = MockCompletionModel::from_turns([
        MockTurn::from_contents(calls),
        MockTurn::text("answered from two results"),
    ]);
    let budget = RunBudget {
        max_tokens: 100_000,
        max_tool_calls: 2,
    };
    let agent = AgentBuilder::new(model)
        .tool(Search)
        .add_hook(budget)
        .build();
    let answer = agent.prompt("research").max_turns(5).await?;
    println!("tool calls: {answer}");

    Ok(())
}
