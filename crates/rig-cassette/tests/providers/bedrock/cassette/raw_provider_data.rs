//! The unary Converse frame, `InternalConverseOutput`, carries everything
//! rig does not normalize, so what Bedrock sends has to survive the trip
//! into it. A transport decorator reads it off the frame stream, SDK-typed
//! fields included.
//!
//! The guardrail trace is the one that costs a caller real information. A
//! blocked turn normalizes to `FinishReason::ContentFilter` and nothing else;
//! only the trace says which policy fired and on what text.

use rig::wire::Wire as _;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::bedrock;
use rig::bedrock::client::BedrockRuntime;
use rig::bedrock::completion::{Converse, ConverseFrame, ConverseRequest};
use rig::bedrock::types::converse_output::InternalConverseOutput;
use rig::driver::{Observation, Opened, Transport};
use rig::error::ProviderError;
use rig::wire::Mode;

use super::super::support::with_bedrock_cassette;
use rig::completion::CompletionRequestBuilder;

/// Keeps every unary Converse output the runtime returns.
#[derive(Clone)]
struct Keep {
    runtime: BedrockRuntime,
    outputs: Arc<Mutex<Vec<InternalConverseOutput>>>,
}

impl Keep {
    fn new(runtime: BedrockRuntime) -> Self {
        Self {
            runtime,
            outputs: Arc::default(),
        }
    }

    fn output(&self) -> InternalConverseOutput {
        self.outputs
            .lock()
            .expect("kept outputs should not be poisoned")
            .pop()
            .expect("the call should return a unary Converse output")
    }
}

impl Transport<Converse> for Keep {
    fn send(
        &self,
        payload: ConverseRequest,
        mode: Mode,
        observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<ConverseRequest, ConverseFrame>> + Send + 'static + use<>,
        ProviderError,
    > {
        let sent = Transport::<Converse>::send(&self.runtime, payload, mode, observation)?;
        let outputs = Arc::clone(&self.outputs);
        Ok(async move {
            let mut opened = sent.await;
            opened.frames = Box::pin(opened.frames.inspect(
                move |frame: &Result<ConverseFrame, ProviderError>| {
                    if let Ok(ConverseFrame::Whole(output)) = frame {
                        outputs
                            .lock()
                            .expect("kept outputs should not be poisoned")
                            .push((**output).clone());
                    }
                },
            ));
            opened
        })
    }
}

/// The guardrail this scenario was recorded against. It is an account-scoped
/// resource name, not a credential, and the guardrail itself was deleted after
/// recording; replay only has to send the same identifier the cassette saw.
const GUARDRAIL_ID: &str = "fytaiyvapuzp";
const GUARDRAIL_VERSION: &str = "DRAFT";

/// Recorded against a guardrail that blocks a specific phrase: Bedrock stops
/// the turn with `guardrail_intervened` and explains itself in `trace`.
#[tokio::test]
async fn guardrail_trace_survives_into_the_converse_frame() {
    with_bedrock_cassette(
        "raw_provider_data/guardrail_trace_survives_into_raw_completion",
        |client| async move {
            let keep = Keep::new(client.0);
            let model = Converse::new(bedrock::completion::AMAZON_NOVA_LITE)
                .with_guardrail(
                    GUARDRAIL_ID,
                    GUARDRAIL_VERSION,
                    aws_sdk_bedrockruntime::types::GuardrailTrace::Enabled,
                )
                .on(keep.clone());

            let request = CompletionRequestBuilder::new(
                "Explain a gravitational singularity in one sentence.",
            )
            .max_tokens(64)
            .build();

            model
                .call(request)
                .await
                .expect("guardrail-intervened completion should still return a response");

            let output = keep.output();
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
                .expect("the guardrail trace must reach the Converse frame");
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
    use rig::streaming::StreamEvent;

    with_bedrock_cassette(
        "raw_provider_data/request_id_survives_into_streamed_terminal",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequestBuilder::new("Reply with the single word: ready.")
                .max_tokens(16)
                .build();

            let mut stream = model.stream(request).expect("stream should start");
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamEvent::Final(final_record) = item.expect("stream item should succeed")
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
async fn request_id_survives_into_the_converse_frame() {
    with_bedrock_cassette(
        "raw_provider_data/request_id_survives_into_raw_completion",
        |client| async move {
            let keep = Keep::new(client.0);
            let model = Converse::new(bedrock::completion::AMAZON_NOVA_LITE).on(keep.clone());
            let request = CompletionRequestBuilder::new("Reply with the single word: ready.")
                .max_tokens(16)
                .build();

            let response = model
                .call(request)
                .await
                .expect("completion should succeed");

            let output = keep.output();
            let request_id = output
                .request_id()
                .expect("the AWS request id must reach the Converse frame");
            assert_eq!(response.provider_request_id.as_deref(), Some(request_id));
            assert!(
                !request_id.trim().is_empty(),
                "expected a non-empty AWS request id"
            );
        },
    )
    .await;
}
