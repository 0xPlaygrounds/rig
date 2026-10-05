//! Ollama's OpenAI-compatible `/v1/chat/completions` route, reached through
//! `Ollama::openai_compatible_completion`: streamed text with `think` sent
//! as `reasoning_effort`, a tool loop, structured output through
//! `response_format`, and reasoning carried back across turns.
//!
//! Replays by default; record with a local daemon serving `qwen3:4b`:
//! `cargo xtask cassette record ollama/openai_compatible/<scenario>.yaml`.

use rig::completion::CompletionRequest;
use rig_agent::test_utils::optional_argument;

use super::super::{CASSETTE_MODEL, support::with_ollama_cassette};
use crate::cassettes::recorded_json_request;
use crate::reasoning::{self, ReasoningRoundtripAgent};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput,
    assert_nonempty_response, assert_smoke_structured_output, collect_stream_final_response,
};

const PROVIDER: &str = "ollama";

/// `think: false` reaches this API as `reasoning_effort: "none"`.
#[tokio::test]
async fn streaming_smoke() {
    let scenario = "openai_compatible/streaming_smoke";
    with_ollama_cassette("openai_compatible/streaming_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.openai_compatible_completion(CASSETTE_MODEL))
            .preamble(STREAMING_PREAMBLE)
            .additional_params(serde_json::json!({ "think": false }))
            .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
    let sent = recorded_json_request(PROVIDER, scenario);
    assert_eq!(sent["reasoning_effort"], "none");
    assert!(sent.get("think").is_none());
}

#[tokio::test]
async fn tool_with_optional_argument() {
    with_ollama_cassette("openai_compatible/optional_argument", |client| async move {
        let report = optional_argument(
            client.openai_compatible_completion(CASSETTE_MODEL),
            |builder| builder,
        )
        .await
        .expect("optional-argument conformance scenario should succeed");
        eprintln!("[ollama] {report:?}");
    })
    .await;
}

#[tokio::test]
async fn structured_output_smoke() {
    with_ollama_cassette(
        "openai_compatible/structured_output_smoke",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.openai_compatible_completion(CASSETTE_MODEL))
                .output_schema::<SmokeStructuredOutput>()
                .additional_params(serde_json::json!({ "think": false }))
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput = serde_json::from_str(&response.output())
                .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_roundtrip_streaming() {
    with_ollama_cassette(
        "openai_compatible/reasoning_roundtrip_streaming",
        |client| async move {
            reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
                client.openai_compatible_completion(CASSETTE_MODEL),
                None,
            ))
            .await;
        },
    )
    .await;
}

/// `num_ctx` has no place on this API, which would drop it and cut a long
/// prompt to the model's default context, so it is refused before sending.
#[tokio::test]
async fn native_options_are_refused() {
    let client = rig_test_support::cassette_models::OllamaModels::new(
        rig::providers::ollama::OllamaConfig::new(),
        rig_test_support::cassettes::local_http(),
    );
    let refused = client
        .openai_compatible_completion(CASSETTE_MODEL)
        .call(CompletionRequest::new("hi").additional_params(serde_json::json!({"num_ctx": 8192})))
        .await
        .expect_err("the request is refused");
    assert!(refused.to_string().contains("num_ctx"), "{refused}");
}
