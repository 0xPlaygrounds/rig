//! Tool fixtures for the Gemini tool-pipeline cassette suites: a zero-argument
//! tool and embeddable arithmetic tools for retrieved-tool selection.
//!
//! Cassettes are re-recorded against the live provider, so values shaped by
//! model output (generated text, call count or order, token counts, embedding
//! rankings) get loose assertions. Only values rig synthesizes with no model
//! input are pinned to exact equality.

use rig::tool::{Tool, ToolEmbedding};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::agent_run_support::{MathError, OperationArgs, operation_definition};

#[derive(Deserialize, Serialize)]
pub(crate) struct EmptyArgs {}

/// Zero-argument tool: pins the null-arguments -> `{}` normalization on the
/// execution path and a verbatim string output.
pub(crate) struct Ping;

pub(crate) const PING_OUTPUT: &str = "pong-crimson-7423";

impl Tool for Ping {
    const NAME: &'static str = "ping";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the current ping marker. Takes no arguments.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(PING_OUTPUT.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("init error")]
pub(crate) struct InitError;

macro_rules! embeddable_operation {
    ($name:ident, $tool_name:literal, $description:literal, $embedding_doc:literal, $op:expr) => {
        #[derive(Clone, Default, Deserialize, Serialize)]
        pub(crate) struct $name;

        impl Tool for $name {
            const NAME: &'static str = $tool_name;
            type Error = MathError;
            type Args = OperationArgs;
            type Output = i64;

            fn description(&self) -> String {
                $description.to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                operation_definition(Self::NAME, $description).parameters
            }

            async fn call(
                &self,
                _context: &mut rig::tool::ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                let op: fn(i64, i64) -> i64 = $op;
                Ok(op(args.x, args.y))
            }
        }

        impl ToolEmbedding for $name {
            type InitError = InitError;
            type Context = ();
            type State = ();

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self)
            }

            fn embedding_docs(&self) -> Vec<String> {
                vec![$embedding_doc.into()]
            }

            fn context(&self) -> Self::Context {}
        }
    };
}

embeddable_operation!(
    EmbedAdd,
    "add",
    "Add x and y together",
    "Add two numbers together to get their sum",
    |x, y| x + y
);
embeddable_operation!(
    EmbedSubtract,
    "subtract",
    "Subtract y from x (i.e. x - y)",
    "Subtract one number from another to get their difference",
    |x, y| x - y
);
embeddable_operation!(
    EmbedMultiply,
    "multiply",
    "Multiply x and y together",
    "Multiply two numbers together to get their product",
    |x, y| x * y
);
