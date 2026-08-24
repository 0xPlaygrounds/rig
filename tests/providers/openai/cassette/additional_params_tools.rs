//! Chat Completions: builder tools must survive `additional_params` that also
//! carries a `tools` key (issue #1890).
//!
//! The chat request body flattens `additional_params` into the serialized
//! struct, so a raw `tools` array supplied there used to silently *replace*
//! the builder's function tools (the body is built via `serde_json::to_value`,
//! where the flattened key wins last). The intended contract — matching the
//! Responses API path, which merges `additional_params["tools"]` into the
//! typed tool list — is that both sets reach the wire.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;

use super::super::support::with_openai_completions_cassette;
use crate::support::zero_arg_tool_definition;

#[tokio::test]
async fn builder_tools_survive_additional_params_tools() {
    with_openai_completions_cassette(
        "additional_params_tools/builder_tools_survive_additional_params_tools",
        |client| async move {
            let model = client.completion_model("gpt-4o-mini");
            let request = model
                .completion_request(
                    "Call the lookup_alpha tool now. Do not call any other tool.",
                )
                .tool(zero_arg_tool_definition("lookup_alpha"))
                .tool_choice(ToolChoice::Required)
                .additional_params(serde_json::json!({
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "lookup_beta",
                            "description": "Returns the beta signal. Only call when explicitly asked for beta.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": false
                            }
                        }
                    }]
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            let called: Vec<&str> = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                called.contains(&"lookup_alpha"),
                "the builder-registered tool should reach the wire and be callable; got {called:?}"
            );
        },
    )
    .await;
}
