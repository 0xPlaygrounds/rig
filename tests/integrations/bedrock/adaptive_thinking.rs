//! Live Bedrock Anthropic adaptive-thinking regression tests.

use futures::StreamExt;
use rig::agent::AgentBuilder;
use rig::completion::AssistantContent;
use rig::streaming::StreamEvent;
use serde_json::json;

use super::{
    anthropic_adaptive_model, anthropic_signature_only_model, client,
    support::{ALPHA_SIGNAL_OUTPUT, AlphaSignal, assert_contains_all_case_insensitive},
};
use rig::completion::CompletionRequest;

fn adaptive_thinking_params() -> serde_json::Value {
    json!({
        "thinking": {
            "type": "adaptive"
        }
    })
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock Anthropic adaptive-thinking model access"]
async fn adaptive_thinking_prompt_caching_tool_roundtrip_regression() {
    let model = client().completion(anthropic_adaptive_model());
    // A checkpoint after the reasoning turn is refused by default; this
    // round trip skips it with a warning and checks the rest still works.
    let agent = AgentBuilder::new(model)
        .options(
            rig::completion::GenerationOptions::default()
                .cache(rig::completion::CacheRetention::Short)
                .on_unsupported(rig::completion::OnUnsupported::Ignore),
        )
        .preamble(
            "You must call tools when the user asks for their result. \
             After a tool result is available, answer with the exact result.",
        )
        .max_tokens(2048)
        .additional_params(adaptive_thinking_params())
        .tool(AlphaSignal)
        .build();

    let response = agent
        .prompt("Call `lookup_harbor_label` exactly once, then answer with the exact tool output.")
        .await
        .expect("adaptive-thinking prompt-caching tool roundtrip should succeed")
        .output();

    assert_contains_all_case_insensitive(&response, &[ALPHA_SIGNAL_OUTPUT]);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock Anthropic adaptive-thinking model access"]
async fn streaming_emits_signature_only_adaptive_reasoning_regression() {
    let model = client().completion(anthropic_signature_only_model());
    let request = CompletionRequest::new("What is 2 + 2? Answer with only the number.")
        .max_tokens(2048)
        .additional_params(adaptive_thinking_params());
    let mut stream = model
        .stream(request)
        .expect("adaptive-thinking Bedrock stream should start");

    let mut reasoning_chunks = 0;
    let mut signature_chunks = 0;
    let mut signature_only_chunks = 0;

    while let Some(item) = stream.next().await {
        if let rig::streaming::Item::Event(StreamEvent::End {
            content: content @ AssistantContent::Reasoning(_),
            ..
        }) = item.expect("adaptive-thinking Bedrock stream item should succeed")
        {
            reasoning_chunks += 1;
            let signed = content
                .native_item()
                .is_some_and(|item| item.get("signature").is_some());
            if signed {
                signature_chunks += 1;
                if matches!(&content, AssistantContent::Reasoning(r) if r.text.is_empty()) {
                    signature_only_chunks += 1;
                }
            }
        }
    }
    stream
        .finish()
        .await
        .expect("stream should emit a final response");
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
