//! Shared fixtures, tiny tools, and durable assertions for ignored smoke tests.
#![allow(dead_code)]

use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";

pub(crate) const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];
pub(crate) const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";

pub(crate) const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
pub(crate) const TOOLS_PROMPT: &str = "Calculate 2 - 5.";

pub(crate) const LOADERS_GLOB: &str = "rig-core/examples/*.rs";
pub(crate) const LOADERS_PROMPT: &str = "Which example file builds an agent from files loaded via FileLoader::with_glob(\"rig-core/examples/*.rs\")? Answer with just the file name.";

#[derive(Deserialize)]
pub(crate) struct OperationArgs {
    pub(crate) x: i32,
    pub(crate) y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
pub(crate) struct MathError;

#[derive(Deserialize, Serialize)]
pub(crate) struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

pub(crate) fn assert_nonempty_response(response: &str) {
    let trimmed = response.trim();

    assert!(
        !trimmed.is_empty(),
        "Response was empty or whitespace-only."
    );
}

pub(crate) fn assert_contains_any_case_insensitive(response: &str, expected: &[&str]) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let matched = expected
        .iter()
        .any(|needle| response_lower.contains(&needle.to_ascii_lowercase()));

    assert!(
        matched,
        "Response {:?} did not contain any of {:?}.",
        response, expected
    );
}

pub(crate) fn assert_mentions_expected_number(response: &str, expected: i32) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let abs = expected.abs();
    let mut candidates = vec![expected.to_string()];

    if expected < 0 {
        candidates.push(format!("minus {abs}"));
        candidates.push(format!("negative {abs}"));
    }

    let matched = candidates
        .iter()
        .any(|candidate| response_lower.contains(&candidate.to_ascii_lowercase()));

    assert!(
        matched,
        "Response {:?} did not mention the expected number {:?}.",
        response, expected
    );
}

pub(crate) fn assert_loader_answer_is_relevant(response: &str) {
    assert_contains_any_case_insensitive(
        response,
        &[
            "agent_with_loaders",
            "agent_with_loaders.rs",
            "agent with loaders",
        ],
    );
}
