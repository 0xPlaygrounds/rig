//! What Bedrock sends beyond what rig normalizes has to survive into the
//! response: `raw` is the body Bedrock sent, guardrail trace included, and
//! the AWS request id rides the response.
//!
//! The guardrail trace is the one that costs a caller real information. A
//! blocked turn normalizes to `FinishReason::ContentFilter` and nothing else;
//! only the trace says which policy fired and on what text.

use rig::bedrock;
use rig::bedrock::extension::{Bedrock, BedrockOptions, Guardrail, GuardrailTrace};
use rig::completion::{CompletionRequest, ProviderOptions};
use serde_json::Value;

use super::super::support::with_bedrock_cassette;

/// The guardrail this scenario was recorded against. It is an account-scoped
/// resource name, not a credential, and the guardrail itself was deleted after
/// recording; replay only has to send the same identifier the cassette saw.
const GUARDRAIL_ID: &str = "fytaiyvapuzp";
const GUARDRAIL_VERSION: &str = "DRAFT";

/// Recorded against a guardrail that blocks a specific phrase: Bedrock stops
/// the turn with `guardrail_intervened` and explains itself in `trace`,
/// which `raw` carries as Bedrock sent it (#2311), and the typed extras
/// read.
#[tokio::test]
async fn guardrail_trace_survives_into_raw() {
    with_bedrock_cassette(
        "raw_provider_data/guardrail_trace_survives_into_raw_completion",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let guardrail = BedrockOptions::default().guardrail(
                Guardrail::new(GUARDRAIL_ID, GUARDRAIL_VERSION).trace(GuardrailTrace::Enabled),
            );
            let request =
                CompletionRequest::new("Explain a gravitational singularity in one sentence.")
                    .max_tokens(64)
                    .provider_options(
                        ProviderOptions::new()
                            .with::<Bedrock>(&guardrail)
                            .expect("the options serialize"),
                    );

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

            let extras = response
                .extras::<Bedrock>()
                .expect("a Bedrock reply")
                .expect("the extras read");
            assert_eq!(extras.stop_reason.as_deref(), Some("guardrail_intervened"));
            assert_eq!(extras.latency_ms, Some(271));
            let trace = extras.trace.expect("the guardrail trace");
            assert_eq!(
                trace.pointer("/guardrail/actionReason"),
                Some(&Value::from("Guardrail blocked."))
            );
            assert!(
                trace
                    .pointer(&format!("/guardrail/inputAssessment/{GUARDRAIL_ID}"))
                    .is_some(),
                "{trace}"
            );
            assert_eq!(extras.service_tier, None);
            assert_eq!(extras.performance_latency, None);
            assert_eq!(extras.cache_details, None);
            assert_eq!(extras.invoked_model_id, None);
            assert_eq!(extras.additional_model_response_fields, None);
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
