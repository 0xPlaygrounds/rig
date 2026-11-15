//! Cap a run's total *dollars* instead of tokens, without credentials.
//!
//! A million tokens differs by two orders of magnitude between a haiku-class
//! and a frontier model, so a token cap is a leaky proxy for money. The hook
//! below books every model response against an [`llm_budget::AgentBudget`] at
//! the model's real price (prompt-cache reads and writes included) and stops
//! the run before the next call once the cap is crossed.
//!
//! ```not_rust
//! cargo run -p rig-agent --example dollar_budget
//! ```

use std::convert::Infallible;

use anyhow::Result;
use rig_agent::{
    AgentBuilder,
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, OutcomeAction,
        OutcomeEvent,
    },
    tool::{Tool, ToolContext},
};
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockTurn};

/// The model the run is priced as. Production hooks take this from the
/// agent's configured model; the mock model here answers as anything.
const PRICED_AS: &str = "claude-haiku-4-5";

/// Stops the run once the dollar budget is spent.
struct DollarBudget {
    budget: llm_budget::AgentBudget,
}

impl DollarBudget {
    fn new(limit_usd: f64) -> Result<Self, llm_budget::BudgetError> {
        Ok(Self {
            budget: llm_budget::AgentBudget::builder("run", limit_usd).build()?,
        })
    }
}

impl AgentHook for DollarBudget {
    /// Books every model response at real prices, a retried or repaired
    /// attempt included, and kills the budget when the cap is crossed so a
    /// run sharing the budget stops before its next call.
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if let Some(response) = event.completion() {
            let usage = response.usage;
            let input = usage.input_tokens.or(usage.total_tokens).unwrap_or(0);
            let output = usage.output_tokens.unwrap_or(0);
            let Ok(price) = llm_budget::PriceTable::builtin().require(PRICED_AS) else {
                // Unpriced model: let the run continue rather than guess a rate.
                return OutcomeAction::proceed();
            };
            let cost = price.cost(input, output);
            let outcome = self.budget.charge_usd_force(
                cost,
                llm_budget::ChargeContext::new()
                    .model(PRICED_AS)
                    .tokens(input, output),
            );
            if outcome.over_cap {
                self.budget.kill("settled spend crossed the cap");
            }
        }
        OutcomeAction::proceed()
    }

    /// Runs before every model call, a retry included.
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if self.budget.is_killed() || self.budget.is_exhausted() {
            return CompletionCallAction::stop(format!(
                "dollar budget: ${:.4} of ${:.4} spent",
                self.budget.spent_usd(),
                self.budget.limit_usd()
            ));
        }
        CompletionCallAction::continue_run()
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
    // 400 input + 200 output tokens on claude-haiku-4-5
    // ($1/M in, $5/M out) costs $0.0014 per call.
    MockTurn::tool_call(id, "search", serde_json::json!({}))
        .with_usage(Usage::new().input_tokens(400).output_tokens(200))
}

#[tokio::main]
async fn main() -> Result<()> {
    // A $0.002 cap against three $0.0014 calls: the second books $0.0028,
    // crossing the cap, so the third call never goes out.
    let model =
        MockCompletionModel::from_turns([search("s1"), search("s2"), MockTurn::text("done")]);
    let budget = DollarBudget::new(0.002)?;
    let agent = AgentBuilder::new(model)
        .tool(Search)
        .add_hook(budget)
        .build();
    match agent.prompt("keep researching").max_turns(5).await {
        Ok(answer) => println!("answered: {answer}"),
        Err(error) => println!("stopped: {error}"),
    }
    Ok(())
}
