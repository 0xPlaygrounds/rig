//! Native counterparts of `stream_faults.rs`: the same recorded setup
//! failure and scripted faults through the ECS runtime and the real
//! adapter, with the world's witness installed. Each cell runs with and
//! without a witness: observation is a side channel, so the failure, the
//! record and the history must not change with it, and the trace must name
//! the fault without carrying the request or its credential.

use bytes::Bytes;
use rig::error::ErrorKind;
use rig::observe::{AdapterEnding, AdapterErrorBoundary, AdapterEvent, AdapterUsage};
use rig::prelude::*;
use rig::providers::gemini::{
    self,
    completion::{
        GEMINI_2_5_FLASH, GEMINI_3_FLASH_PREVIEW,
        gemini_api_types::{AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel},
    },
};
use rig::test_utils::SequencedStreamingHttpClient;
use rig_ecs::agent::{AdditionalParams, MaxTokens, Preamble, Role};

use super::super::support::with_gemini_cassette;
use super::stream_faults::{
    BLOCKED_FRAME, CONTENT_FRAME, CONTENT_TEXT, IN_BAND_ERROR_FRAME, MISSING_MODEL, SCRIPTED_KEY,
    SETUP_MAX_TOKENS, SETUP_PROMPT, gemini_sse, scripted_client,
};
use crate::{
    ecs_agent::EcsAgent,
    stream_faults::{
        NativeRun, adapter_events, assert_setup_failure, bus_actions, comparable_failure, endings,
        native_run, sole_failed_completion, trace_json, truncations,
    },
    support::{STREAMING_PREAMBLE, STREAMING_PROMPT},
};

/// A scripted-transport model: one streaming exchange, then EOF.
fn scripted_model(chunks: Vec<Bytes>) -> gemini::CompletionModel<SequencedStreamingHttpClient> {
    scripted_client(chunks).completion_model(GEMINI_2_5_FLASH)
}

/// The scripted cells' witness check, over this module's credential.
fn assert_witness_is_a_side_channel(observed: &NativeRun, plain: &NativeRun) {
    crate::stream_faults::assert_witness_is_a_side_channel(observed, plain, SCRIPTED_KEY);
}

/// Every provider-boundary fact of the run's one operation.
fn boundary(run: &NativeRun) -> Vec<AdapterEvent> {
    let events = adapter_events(run.trace());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AdapterEvent::Started { .. })),
        "the adapter announces its request: {events:?}"
    );
    events
}

/// The recorded 404 through the native runtime: the run fails as the
/// provider's response, streams nothing, commits only the prompt, and the
/// witness sees the request, the status, the envelope and the ending.
#[tokio::test]
async fn setup_failure_fails_the_run_with_the_recorded_status() {
    // The recording's request carries no system instruction at all, which
    // strict matching distinguishes from an empty one: `Preamble(None)`,
    // not the helper's `Some("")`.
    let configure = |ecs: &mut EcsAgent| {
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert((Preamble(None), MaxTokens(Some(SETUP_MAX_TOKENS))));
    };
    let mut runs = Vec::new();
    for witness in [true, false] {
        let runs = &mut runs;
        with_gemini_cassette(
            "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
            |client| async move {
                let run = native_run(
                    client.completion_model(MISSING_MODEL),
                    "",
                    SETUP_PROMPT,
                    witness,
                    configure,
                )
                .await;
                assert_setup_failure(run.provider_report(), ErrorKind::ProviderResponse, 404);
                assert_setup_failure(
                    sole_failed_completion(&run.log),
                    ErrorKind::ProviderResponse,
                    404,
                );
                assert_eq!(run.roles, [Role::User], "only the prompt is history");
                assert!(run.stream().text.is_empty(), "{:?}", run.stream);
                assert_eq!(
                    run.stream().errors.len(),
                    1,
                    "one error item: {:?}",
                    run.stream().errors
                );
                assert_eq!(run.stream().errors[0].0, 0, "the error is the first item");
                runs.push(run);
            },
        )
        .await;
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_eq!(
        comparable_failure(observed.failure()),
        comparable_failure(plain.failure())
    );
    assert_eq!(observed.log_json(), plain.log_json());

    let trace = observed.trace();
    assert_eq!(bus_actions(trace), ["issued", "landed_err"]);
    assert_eq!(endings(trace), ["provider"]);
    let events = boundary(observed);
    assert!(
        events.contains(&AdapterEvent::Response { status: 404 }),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::ErrorEnvelope { error }
                if error.code.as_deref() == Some("404") && error.status.as_deref() == Some("NOT_FOUND")
        )),
        "the envelope's own fields are projected: {events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::ProviderResponse,
                    status: Some(404),
                    retryable: false,
                    ..
                }
            }
        )),
        "the attempt closes as a non-retryable provider response: {events:?}"
    );
    let rendered = trace_json(trace);
    assert!(
        !rendered.contains(SETUP_PROMPT),
        "the request body never reaches the trace"
    );
}

/// The refusal through the native runtime: a non-retryable provider
/// failure naming the block reason, nothing committed, and the witness sees
/// the verdict and the usage the provider billed for the refused prompt.
#[tokio::test]
async fn blocked_prompt_is_a_provider_refusal_not_a_truncation() {
    let mut runs = Vec::new();
    for witness in [true, false] {
        let run = native_run(
            scripted_model(vec![gemini_sse(&[BLOCKED_FRAME])]),
            "",
            "pong?",
            witness,
            |_| {},
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::Provider, "{report:?}");
        assert!(!report.is_retryable(), "{report:?}");
        assert!(report.message.contains("block_reason=SAFETY"), "{report:?}");
        let recorded = sole_failed_completion(&run.log);
        assert_eq!(recorded.kind, ErrorKind::Provider, "{recorded:?}");
        assert!(
            recorded.message.contains("block_reason=SAFETY"),
            "the record holds the refusal: {recorded:?}"
        );
        assert_eq!(run.roles, [Role::User]);
        assert!(run.stream().text.is_empty());
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    assert_eq!(endings(observed.trace()), ["provider"]);
    assert!(truncations(observed.trace()).is_empty(), "not a truncation");
    let events = boundary(observed);
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Provider { verdict } if verdict.block_reason.as_deref() == Some("SAFETY")
        )),
        "the verdict names the block: {events:?}"
    );
    assert!(
        events.contains(&AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: Some(795),
                total_tokens: Some(795),
                ..AdapterUsage::default()
            }
        }),
        "the refused prompt's billed usage is observed: {events:?}"
    );
}

/// EOF after content through the native runtime: the run fails as a
/// truncation, the stream keeps the prefix and no error item (the wire sent
/// none), history keeps only the prompt, and the bus witnesses the
/// truncation with what was delivered.
#[tokio::test]
async fn truncation_after_content_fails_the_run_and_keeps_the_prefix() {
    let mut runs = Vec::new();
    for witness in [true, false] {
        let run = native_run(
            scripted_model(vec![gemini_sse(&[CONTENT_FRAME])]),
            "",
            "pong?",
            witness,
            |_| {},
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
        assert_eq!(report.message, rig::serve::stream_truncated().message);
        let recorded = sole_failed_completion(&run.log);
        assert_eq!(recorded.message, rig::serve::stream_truncated().message);
        assert_eq!(run.stream().text, CONTENT_TEXT);
        assert!(run.stream().errors.is_empty(), "EOF is not an item");
        // No terminal record and no error item: the fold never closed, so
        // the truncation is the effect's outcome (asserted through the
        // record above), not a folded stream outcome.
        assert!(run.stream().outcome.is_none(), "{:?}", run.stream().outcome);
        assert_eq!(run.roles, [Role::User], "the cut turn is not history");
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    assert_eq!(endings(observed.trace()), ["provider"]);
    let delivered = observed.stream().events.len();
    assert_eq!(truncations(observed.trace()), [(delivered, 0)]);
    let events = boundary(observed);
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Eof { after: 1 }
            }
        )),
        "the attempt closes at EOF after one frame: {events:?}"
    );
}

/// An error envelope after content through the native runtime: the run
/// fails with the envelope's classification, the stream keeps the prefix
/// and the error item at its position, and the witness sees the envelope.
#[tokio::test]
async fn in_band_error_after_content_fails_with_the_envelope() {
    let mut runs = Vec::new();
    for witness in [true, false] {
        let run = native_run(
            scripted_model(vec![gemini_sse(&[CONTENT_FRAME, IN_BAND_ERROR_FRAME])]),
            "",
            "pong?",
            witness,
            |_| {},
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
        assert_eq!(report.http_status, Some(503), "{report:?}");
        assert!(report.is_retryable(), "{report:?}");
        assert_eq!(
            report
                .provider_response_body()
                .map(|body| serde_json::from_str::<serde_json::Value>(body).expect("JSON")),
            Some(serde_json::from_str(IN_BAND_ERROR_FRAME).expect("JSON")),
            "the envelope is preserved: {report:?}"
        );
        let recorded = sole_failed_completion(&run.log);
        assert_eq!(recorded.http_status, Some(503), "{recorded:?}");
        assert_eq!(run.stream().text, CONTENT_TEXT);
        assert_eq!(run.stream().errors.len(), 1, "{:?}", run.stream().errors);
        assert_eq!(
            run.stream().errors[0].0,
            run.stream().events.len(),
            "the error item follows the delivered content"
        );
        assert_eq!(run.roles, [Role::User]);
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    assert_eq!(endings(observed.trace()), ["provider"]);
    let events = boundary(observed);
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::ErrorEnvelope { error }
                if error.code.as_deref() == Some("503") && error.status.as_deref() == Some("UNAVAILABLE")
        )),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::ProviderResponse,
                    status: Some(503),
                    retryable: true,
                    ..
                }
            }
        )),
        "{events:?}"
    );
}

/// The recorded successful stream through the native runtime with a
/// witness: the answer and the record match an unwitnessed replay, and the
/// trace carries the dispatch, the verdict, the usage and the settlement.
#[tokio::test]
async fn witnessed_success_matches_the_unwitnessed_run() {
    let configure = |ecs: &mut EcsAgent| {
        let config = GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                thinking_budget: None,
                thinking_level: Some(ThinkingLevel::Medium),
                include_thoughts: Some(true),
            }),
            ..Default::default()
        };
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(AdditionalParams(Some(
                serde_json::to_value(AdditionalParameters::default().with_config(config))
                    .expect("thinking configuration"),
            )));
    };
    let mut runs = Vec::new();
    for witness in [true, false] {
        let runs = &mut runs;
        with_gemini_cassette("streaming/streaming_smoke", |client| async move {
            let run = native_run(
                client.completion_model(GEMINI_3_FLASH_PREVIEW),
                STREAMING_PREAMBLE,
                STREAMING_PROMPT,
                witness,
                configure,
            )
            .await;
            let answer = run.outcome.as_ref().expect("the recorded answer");
            assert!(!answer.trim().is_empty());
            assert_eq!(run.roles, [Role::User, Role::Assistant]);
            runs.push(run);
        })
        .await;
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_eq!(observed.outcome, plain.outcome);
    assert_eq!(observed.log_json(), plain.log_json());

    let trace = observed.trace();
    assert_eq!(bus_actions(trace), ["issued", "landed_ok"]);
    assert_eq!(endings(trace), ["settled"]);
    assert!(truncations(trace).is_empty());
    let events = boundary(observed);
    assert!(
        events.contains(&AdapterEvent::Response { status: 200 }),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Provider { verdict } if verdict.finish_reason.as_deref() == Some("STOP")
        )),
        "{events:?}"
    );
    let usage = observed
        .stream()
        .events
        .iter()
        .rev()
        .find_map(|event| match event {
            rig::streaming::StreamEvent::Final(final_event) => Some(final_event.usage),
            _ => None,
        })
        .expect("the terminal record's usage");
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Usage { usage: observed }
                if observed.total_tokens == Some(usage.total_tokens)
        )),
        "the observed usage is the terminal record's: {events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Terminal
            }
        )),
        "{events:?}"
    );
    let rendered = trace_json(trace);
    assert!(
        !rendered.contains(STREAMING_PROMPT) && !rendered.contains(STREAMING_PREAMBLE),
        "the request never reaches the trace"
    );
}
