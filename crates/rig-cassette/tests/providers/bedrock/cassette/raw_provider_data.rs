//! What Bedrock sends beyond what rig normalizes has to survive into the
//! response: `raw` is the body Bedrock sent, guardrail trace included, and
//! the AWS request id rides the response.
//!
//! The guardrail trace is the one that costs a caller real information. A
//! blocked turn normalizes to `FinishReason::ContentFilter` and nothing else;
//! only the trace says which policy fired and on what text.

use rig::bedrock;
use rig::bedrock::completion::Converse;
use rig::driver::Model;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
use rig::completion::CompletionRequest;

/// The guardrail this scenario was recorded against. It is an account-scoped
/// resource name, not a credential, and the guardrail itself was deleted after
/// recording; replay only has to send the same identifier the cassette saw.
const GUARDRAIL_ID: &str = "fytaiyvapuzp";
const GUARDRAIL_VERSION: &str = "DRAFT";

/// Recorded against a guardrail that blocks a specific phrase: Bedrock stops
/// the turn with `guardrail_intervened` and explains itself in `trace`,
/// which `raw` carries as Bedrock sent it (#2311).
#[tokio::test]
async fn guardrail_trace_survives_into_raw() {
    with_bedrock_cassette(
        "raw_provider_data/guardrail_trace_survives_into_raw_completion",
        |client| async move {
            let model = Model::new(
                Converse::new(bedrock::completion::AMAZON_NOVA_LITE).with_guardrail(
                    GUARDRAIL_ID,
                    GUARDRAIL_VERSION,
                    aws_sdk_bedrockruntime::types::GuardrailTrace::Enabled,
                ),
                client.0,
            );

            let request =
                CompletionRequest::new("Explain a gravitational singularity in one sentence.")
                    .max_tokens(64);

            let response = model
                .call(request)
                .await
                .expect("guardrail-intervened completion should still return a response");

            assert_eq!(
                response.raw["stopReason"], "guardrail_intervened",
                "expected the guardrail to intervene: {}",
                response.raw
            );
            let input_assessment = response
                .raw
                .pointer("/trace/guardrail/inputAssessment")
                .and_then(Value::as_object)
                .expect("the blocked input's guardrail assessment reaches raw");
            assert!(
                !input_assessment.is_empty(),
                "expected the input assessment naming the policy that fired"
            );
        },
    )
    .await;
}

/// Blocking/streaming parity (rig#2265): the same AWS request id semantics on
/// the streaming surface — the converse-stream operation output's id reaches
/// the normalized terminal record.
///
/// Ignored until recorded: the AWS credentials available when this test was
/// written had expired, so no cassette exists yet. The conversion semantics
/// are unit-tested in `rig-bedrock`
/// (`streaming::response_identity_tests`); this scenario adds the live wire
/// proof once recorded with `RIG_PROVIDER_TEST_MODE=record`.
#[ignore]
#[tokio::test]
async fn request_id_survives_into_streamed_terminal() {
    use futures::StreamExt;

    with_bedrock_cassette(
        "raw_provider_data/request_id_survives_into_streamed_terminal",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request =
                CompletionRequest::new("Reply with the single word: ready.").max_tokens(16);

            let mut stream = model.stream(request).expect("stream should start");
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }
            let terminal = stream
                .finish()
                .await
                .expect("stream should yield a terminal record");
            assert!(
                terminal
                    .provider_request_id
                    .as_deref()
                    .is_some_and(|id| !id.trim().is_empty()),
                "the AWS request id must reach the streamed terminal, got {:?}",
                terminal.provider_request_id
            );
        },
    )
    .await;
}
