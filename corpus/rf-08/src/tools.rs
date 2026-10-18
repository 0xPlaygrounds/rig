//! The demo registry: plain tools, a failing and a slow one, one that reads
//! the calling user from the call's context, and one that returns an image.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_agent::tool::ToolSet;
use rig_core::message::{ImageMediaType, ToolResultContent};
use rig_core::tool::{ContextValue, Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The calling user, passed by the caller in the call's [`ToolContext`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct User(pub String);

impl ContextValue for User {
    const KEY: &'static str = "user";
}

/// The demo registry. `add_calls` counts how often `add`'s body ran.
pub fn registry(add_calls: Arc<AtomicUsize>) -> ToolSet {
    let mut tools = ToolSet::default();
    tools.add_tool(Add { calls: add_calls });
    tools.add_tool(ConvertCurrency);
    tools.add_tool(FailAlways);
    tools.add_tool(Slow);
    tools.add_tool(WhoAmI);
    tools.add_tool(Chart);
    tools
}

/// Arguments of `add`.
#[derive(Deserialize, JsonSchema)]
pub struct AddArgs {
    /// The first addend.
    pub a: i64,
    /// The second addend.
    pub b: i64,
}

/// `add(a, b)`: their sum, as JSON.
pub struct Add {
    /// How often the body ran.
    pub calls: Arc<AtomicUsize>,
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Adds two integers.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<AddArgs>()
    }

    async fn call(&self, _context: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        args.a
            .checked_add(args.b)
            .ok_or_else(|| ToolExecutionError::invalid_args("the sum overflows"))
    }
}

/// A currency of the fixed rate table.
#[derive(Clone, Copy, Deserialize, JsonSchema)]
pub enum Currency {
    /// US dollar.
    USD,
    /// Euro.
    EUR,
    /// Pound sterling.
    GBP,
}

impl Currency {
    /// US dollars per unit.
    fn in_usd(self) -> f64 {
        match self {
            Self::USD => 1.0,
            Self::EUR => 1.08,
            Self::GBP => 1.27,
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::USD => "USD",
            Self::EUR => "EUR",
            Self::GBP => "GBP",
        }
    }
}

/// Arguments of `convert_currency`.
#[derive(Deserialize, JsonSchema)]
pub struct ConvertArgs {
    /// How much of `from`.
    pub amount: f64,
    /// The currency `amount` is in.
    pub from: Currency,
    /// The currency to convert to.
    pub to: Currency,
}

/// `convert_currency(amount, from, to)` with a fixed rate table: text such
/// as `10.80 USD`.
pub struct ConvertCurrency;

impl Tool for ConvertCurrency {
    const NAME: &'static str = "convert_currency";
    type Args = ConvertArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Converts an amount between currencies at fixed rates.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ConvertArgs>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: ConvertArgs,
    ) -> Result<String, Infallible> {
        let converted = args.amount * args.from.in_usd() / args.to.in_usd();
        Ok(format!("{converted:.2} {}", args.to.code()))
    }
}

/// The failure `fail_always` reports: operator detail the model must not see.
#[derive(Debug, thiserror::Error)]
#[error(
    "connection to ledger db at 10.0.0.7:5432 refused: password authentication failed for svc_ledger"
)]
pub struct LedgerDown;

/// Arguments of a tool that takes none.
#[derive(Deserialize, JsonSchema)]
pub struct NoArgs {}

/// `fail_always`: always fails with [`LedgerDown`].
pub struct FailAlways;

impl Tool for FailAlways {
    const NAME: &'static str = "fail_always";
    type Args = NoArgs;
    type Output = String;
    type Error = LedgerDown;

    fn description(&self) -> String {
        "Always fails.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<NoArgs>()
    }

    async fn call(&self, _context: &mut ToolContext, _args: NoArgs) -> Result<String, LedgerDown> {
        Err(LedgerDown)
    }
}

/// `slow`: never finishes.
pub struct Slow;

impl Tool for Slow {
    const NAME: &'static str = "slow";
    type Args = NoArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Never finishes.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<NoArgs>()
    }

    async fn call(&self, _context: &mut ToolContext, _args: NoArgs) -> Result<String, Infallible> {
        std::future::pending().await
    }
}

/// `whoami`: the calling user, from the call's context.
pub struct WhoAmI;

impl Tool for WhoAmI {
    const NAME: &'static str = "whoami";
    type Args = NoArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Says who is calling.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<NoArgs>()
    }

    async fn call(&self, context: &mut ToolContext, _args: NoArgs) -> Result<String, Self::Error> {
        let User(user) = context.require::<User>()?;
        Ok(format!("you are {user}"))
    }
}

/// A 1x1 transparent PNG.
const PIXEL_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

/// `chart`: an image.
pub struct Chart;

impl Tool for Chart {
    const NAME: &'static str = "chart";
    type Args = NoArgs;
    type Output = ToolResultContent;
    type Error = Infallible;

    fn description(&self) -> String {
        "Draws a chart.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<NoArgs>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: NoArgs,
    ) -> Result<ToolResultContent, Infallible> {
        Ok(ToolResultContent::image_base64(
            PIXEL_PNG,
            Some(ImageMediaType::PNG),
            None,
        ))
    }
}
