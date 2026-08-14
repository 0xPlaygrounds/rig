//! `raw_completion` is the escape hatch for everything rig does not
//! normalize, so what Bedrock sends has to survive the trip through
//! `InternalConverseOutput`. It did not: the conversion's rest pattern
//! discarded the guardrail trace, the performance configuration, the service
//! tier and the AWS request id.
//!
//! The guardrail trace is the one that costs a caller real information. A
//! blocked turn normalizes to `FinishReason::ContentFilter` and nothing else;
//! only the trace says which policy fired and on what text.

use rig::bedrock;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_bedrock_cassette;

/// The guardrail this scenario was recorded against. It is an account-scoped
/// resource name, not a credential, and the guardrail itself was deleted after
/// recording; replay only has to send the same identifier the cassette saw.
const GUARDRAIL_ID: &str = "fytaiyvapuzp";
const GUARDRAIL_VERSION: &str = "DRAFT";

/// Recorded against a guardrail that blocks a specific phrase: Bedrock stops
/// the turn with `guardrail_intervened` and explains itself in `trace`.
#[tokio::test]
async fn guardrail_trace_survives_into_raw_completion() {
    with_bedrock_cassette(
        "raw_provider_data/guardrail_trace_survives_into_raw_completion",
        |client| async move {
            let model = client
                .completion_model(bedrock::completion::AMAZON_NOVA_LITE)
                .with_guardrail(
                    GUARDRAIL_ID,
                    GUARDRAIL_VERSION,
                    aws_sdk_bedrockruntime::types::GuardrailTrace::Enabled,
                );

            let request = model
                .completion_request("Explain a gravitational singularity in one sentence.")
                .max_tokens(64)
                .build();

            let response = model
                .raw_completion(request)
                .await
                .expect("guardrail-intervened completion should still return a response");

            let output = &response.0;
            assert!(
                matches!(
                    output.stop_reason,
                    bedrock::types::converse_output::StopReason::GuardrailIntervened
                ),
                "expected the guardrail to intervene, got {:?}",
                output.stop_reason
            );

            let trace = output
                .trace()
                .expect("the guardrail trace must reach raw_completion");
            let guardrail = trace
                .guardrail()
                .expect("a guardrail-intervened turn carries a guardrail assessment");
            let input_assessment = guardrail
                .input_assessment()
                .expect("the blocked input carries an assessment");
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
    use rig::streaming::StreamedAssistantContent;

    with_bedrock_cassette(
        "raw_provider_data/request_id_survives_into_streamed_terminal",
        |client| async move {
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request("Reply with the single word: ready.")
                .max_tokens(16)
                .build();

            let mut stream = model.stream(request).await.expect("stream should start");
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) =
                    item.expect("stream item should succeed")
                {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("stream should yield a terminal record");
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

/// The AWS request id rides an HTTP header, so it is present on every call —
/// including the ordinary ones — and it is what AWS support asks for.
#[tokio::test]
async fn request_id_survives_into_raw_completion() {
    with_bedrock_cassette(
        "raw_provider_data/request_id_survives_into_raw_completion",
        |client| async move {
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request("Reply with the single word: ready.")
                .max_tokens(16)
                .build();

            let response = model
                .raw_completion(request)
                .await
                .expect("completion should succeed");

            let request_id = response
                .0
                .request_id()
                .expect("the AWS request id must reach raw_completion");
            assert!(
                !request_id.trim().is_empty(),
                "expected a non-empty AWS request id"
            );
        },
    )
    .await;
}
