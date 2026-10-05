//! Adversarial response-identity coverage on OpenAI (rig#2265 / PR #2313
//! follow-up): structured output, response chaining, live hook retries, error
//! responses, and raw-vs-normalized agreement.

use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::support::assert_transport_request_id;
use rig::completion::CompletionRequest;

/// Family A: `output_schema` reshapes the request (structured output);
/// identity still rides it.
#[tokio::test]
async fn structured_output_and_identity() {
    #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Sum {
        value: i64,
    }

    with_openai_cassette(
        "response_identity_edge/structured_output_and_identity",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let schema = schemars::schema_for!(Sum);
            let response = model
                .call(
                    CompletionRequest::new("What is 2 + 3? Respond with the JSON object.")
                        .output_schema(schema),
                )
                .await
                .expect("structured completion should succeed");
            assert_transport_request_id(
                response.provider_request_id.as_deref(),
                "structured-output response",
            );
        },
    )
    .await;
}

/// Family A: a `previous_response_id` chain — exactly where response-scoped
/// and transport ids are most likely to be crossed. The second call reuses
/// the first's *response id* on the wire, yet reports its own transport id.
#[tokio::test]
async fn previous_response_id_chain_keeps_axes_distinct() {
    with_openai_cassette(
        "response_identity_edge/previous_response_id_chain_keeps_axes_distinct",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let first = model
                .call(CompletionRequest::new(
                    "Remember the code word 'heliotrope'. Reply with exactly: noted",
                ))
                .await
                .expect("first chained call should succeed");
            let first_response_id = first
                .response_id()
                .expect("Responses API reports a response id");
            assert_transport_request_id(first.provider_request_id.as_deref(), "chain call 1");

            let second = model
                .call(
                    CompletionRequest::new("What was the code word? Reply with just the word.")
                        .additional_params(serde_json::json!({
                            "previous_response_id": first_response_id,
                        })),
                )
                .await
                .expect("chained call should succeed");

            assert_transport_request_id(second.provider_request_id.as_deref(), "chain call 2");
            assert_ne!(
                first.provider_request_id, second.provider_request_id,
                "each chained call has its own transport id"
            );
            let second_response_id = second.response_id().expect("second response id");
            assert_ne!(
                first_response_id, second_response_id,
                "chaining reuses the first response id as *input*; the second \
                 response still gets its own"
            );
            assert_ne!(
                Some(second_response_id),
                second.provider_request_id.as_deref(),
                "response-scoped and transport ids are never conflated"
            );
        },
    )
    .await;
}
