//! Contract-vs-reality (rig#2265 / PR #2313 follow-up): OpenRouter is an
//! OpenAI-compatible provider whose `REQUEST_ID_HEADER` contract is the
//! conservative default `None` — and the live census matches: the gateway
//! sends **no** `x-request-id` at all (its own per-call id rides a different
//! header, `x-generation-id`, deliberately not adopted as a transport
//! request id). Contract and reality agree; contrast with Groq, whose
//! fixture shows the header present while the contract still captures
//! `None` — together they are the evidence for the compat-default question
//! raised in PR #2313.

use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_openrouter_cassette;

const MODEL: &str = "openai/gpt-5.2";

#[tokio::test]
async fn blocking_contract_and_gateway_both_report_none() {
    with_openrouter_cassette(
        "response_identity_edge/blocking_contract_and_gateway_both_report_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");
            assert_eq!(
                response.provider_request_id, None,
                "no x-request-id on the wire (see the fixture's headers) and \
                 the compat contract is None: agreement, not a silent drop"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_contract_and_gateway_both_report_none() {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    with_openrouter_cassette(
        "response_identity_edge/streaming_contract_and_gateway_both_report_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .stream()
                .await
                .expect("stream should open");
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) =
                    item.expect("stream item should succeed")
                {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("terminal record");
            assert_eq!(terminal.provider_request_id, None);
        },
    )
    .await;
}

/// 2xx-envelope / routed-failure hunt (rig#2314 error matrix): a model id
/// that parses but routes nowhere — record what OpenRouter actually answers
/// (assertion derived from the recording).
#[tokio::test]
async fn routed_failure_error_shape() {
    with_openrouter_cassette(
        "response_identity_edge/routed_failure_error_shape",
        |client| async move {
            let model = client.completion_model("openai/gpt-nonexistent-routed-model");
            let error = model
                .completion_request("Never routed")
                .send()
                .await
                .expect_err("an unroutable model must fail");
            // Derived from the recording: OpenRouter answers a body-ful 4xx;
            // contract-less classification keeps the transport shape.
            assert!(
                matches!(error, rig::completion::CompletionError::HttpError(_)),
                "got {error:?}"
            );
            assert!(error.provider_response_body().is_some());
            assert_eq!(error.provider_request_id(), None);
        },
    )
    .await;
}
