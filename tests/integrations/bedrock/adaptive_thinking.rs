//! Live Bedrock Anthropic adaptive-thinking regression tests.

use futures::StreamExt;
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel as _, Prompt};
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use super::{
    anthropic_adaptive_model, anthropic_signature_only_model, client,
    support::{ALPHA_SIGNAL_OUTPUT, AlphaSignal, assert_contains_all_case_insensitive},
};

fn adaptive_thinking_params() -> serde_json::Value {
    json!({
        "thinking": {
            "type": "adaptive",
            "effort": "high"
        }
    })
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock Anthropic adaptive-thinking model access"]
async fn adaptive_thinking_prompt_caching_tool_roundtrip_regression() {
    let model = client()
        .completion_model(anthropic_adaptive_model())
        .with_prompt_caching();
    let agent = AgentBuilder::new(model)
        .preamble(
            "You must call tools when the user asks for their result. \
             After a tool result is available, answer with the exact result.",
        )
        .max_tokens(4096)
        .additional_params(adaptive_thinking_params())
        .tool(AlphaSignal)
        .build();

    let response = agent
        .prompt("Call `lookup_harbor_label` exactly once, then answer with the exact tool output.")
        .await
        .expect("adaptive-thinking prompt-caching tool roundtrip should succeed");

    assert_contains_all_case_insensitive(&response, &[ALPHA_SIGNAL_OUTPUT]);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock Anthropic adaptive-thinking model access"]
async fn streaming_emits_signature_only_adaptive_reasoning_regression() {
    let model = client().completion_model(anthropic_signature_only_model());
    let request = model
        .completion_request("What is 2 + 2? Answer with only the number.")
        .max_tokens(4096)
        .additional_params(adaptive_thinking_params())
        .build();
    let mut stream = model
        .stream(request)
        .await
        .expect("adaptive-thinking Bedrock stream should start");

    let mut reasoning_chunks = 0;
    let mut signature_chunks = 0;
    let mut signature_only_chunks = 0;
    let mut got_final = false;

    while let Some(item) = stream.next().await {
        match item.expect("adaptive-thinking Bedrock stream item should succeed") {
            StreamedAssistantContent::Reasoning(reasoning) => {
                reasoning_chunks += 1;
                if reasoning.first_signature().is_some() {
                    signature_chunks += 1;
                    if reasoning.display_text().is_empty() {
                        signature_only_chunks += 1;
                    }
                }
            }
            StreamedAssistantContent::Final(_) => got_final = true,
            _ => {}
        }
    }

    assert!(got_final, "stream should emit a final response");
    assert!(
        reasoning_chunks > 0,
        "expected at least one adaptive-thinking reasoning chunk"
    );
    assert!(
        signature_chunks > 0,
        "expected adaptive-thinking reasoning to include a Bedrock signature"
    );
    assert!(
        signature_only_chunks > 0,
        "expected at least one signature-only reasoning chunk"
    );
}
