//! A tool schema that refers to itself.
//!
//! Inlining `$defs` into Gemini's OpenAPI-subset `parameters` cannot express a
//! recursive schema: expanding `Node.child` yields another `Node`, forever.
//! The converter recursed until the stack ran out, which aborts the process —
//! in an ECS host every other run dies with it and no failure is ever written.
//! rig now refuses the request and names the cycle.
//!
//! **What this cassette records, and why it is not the fix.** Gemini's
//! `parametersJsonSchema` takes a standard JSON Schema, recursion included, so
//! the schema rig cannot inline is one the provider would have accepted
//! untouched. This recording is that fact, captured from the wire: the request
//! below sends the recursive schema verbatim through `additional_params` and
//! Gemini answers with a correctly nested `functionCall`.
//!
//! Switching rig's own tool path to `parametersJsonSchema` is therefore a real
//! follow-up — but it is not this fix. The typed `Schema` conversion is still
//! consumed by `GenerationConfig::response_schema` and by
//! `rig-gemini-grpc`, whose gRPC `FunctionDeclaration` has no
//! `parametersJsonSchema` field, so the converter cannot simply be deleted;
//! and the swap changes every Gemini tool request body, which fails 315 of the
//! suite's cassettes on request mismatch. The refusal fixes the crash with no
//! wire change at all; this recording is the evidence the follow-up is worth
//! doing.

use rig::prelude::*;
use rig::providers::gemini;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn gemini_accepts_a_recursive_tool_schema_as_parameters_json_schema() {
    with_gemini_cassette(
        "recursive_schema/gemini_accepts_parameters_json_schema",
        |client| async move {
            // Sent through `additional_params` so the recursive schema reaches
            // the wire exactly as authored: this asks what Gemini does with
            // it, not what rig's converter does.
            let tools = serde_json::json!({
                "tools": [{
                    "functionDeclarations": [{
                        "name": "add_node",
                        "description": "add a tree node",
                        "parametersJsonSchema": {
                            "type": "object",
                            "$defs": {
                                "Node": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "child": {"$ref": "#/$defs/Node"}
                                    }
                                }
                            },
                            "properties": {"root": {"$ref": "#/$defs/Node"}},
                            "required": ["root"]
                        }
                    }]
                }]
            });

            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .additional_params(tools)
                .build();

            // No local tool is registered: the agent loop refuses the call it
            // gets back. That refusal is the proof — it carries the arguments
            // Gemini produced for the recursive schema, so the schema was
            // accepted, understood, and answered in the nested shape.
            let error = agent
                .prompt("Call add_node with a root node named a whose child is named b.")
                .await
                .expect_err("no local tool is registered for the declared name");

            let reported = format!("{error:?}");
            assert!(
                reported.contains("add_node"),
                "Gemini called the tool declared by the recursive schema: {reported}"
            );
            assert!(
                reported.contains("child"),
                "the arguments nest a Node inside a Node, which is what the \
                 recursion declares: {reported}"
            );
        },
    )
    .await;
}
