//! The recorded Messages setup failure through the native runtime with the
//! world's witness installed: the Messages adapter's boundary facts beside
//! the bus's, on the recording (`corpus_outcome/model_error_streamed`) that
//! `ecs_outcome::model_error_streamed` pins. Observation is a side channel: the failure, the record and the
//! history read the same with and without it.

use rig::error::ErrorKind;
use rig::observe::{AdapterEnding, AdapterErrorBoundary, AdapterEvent};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_ecs::agent::{Role, Temperature};

use super::super::support::with_anthropic_cassette_bogus_key;
use crate::{
    ecs_agent::EcsAgent,
    stream_faults::{
        adapter_events, assert_setup_failure, bus_actions, comparable_failure, endings, native_run,
        sole_failed_completion, trace_json,
    },
    support::{BASIC_PREAMBLE, BASIC_PROMPT},
};

/// The recorded 401 through the native runtime: the run fails as the
/// provider's response, streams nothing, commits only the prompt, and the
/// witness sees the request, the status, the envelope and the ending.
#[tokio::test]
async fn setup_failure_fails_the_run_with_the_recorded_status() {
    let configure = |ecs: &mut EcsAgent| {
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
    };
    let mut runs = Vec::new();
    for witness in [true, false] {
        let runs = &mut runs;
        with_anthropic_cassette_bogus_key(
            "corpus_outcome/model_error_streamed",
            |client| async move {
                let run = native_run(
                    client.completion_model(CLAUDE_SONNET_4_6),
                    BASIC_PREAMBLE,
                    BASIC_PROMPT,
                    witness,
                    configure,
                )
                .await;
                assert_setup_failure(run.provider_report(), ErrorKind::ProviderResponse, 401);
                assert_setup_failure(
                    sole_failed_completion(&run.log),
                    ErrorKind::ProviderResponse,
                    401,
                );
                assert_eq!(run.roles, [Role::User], "only the prompt is history");
                assert!(run.stream().text.is_empty(), "{:?}", run.stream());
                assert_eq!(run.stream().errors.len(), 1, "{:?}", run.stream().errors);
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
    let events = adapter_events(trace);
    assert!(
        matches!(events.first(), Some(AdapterEvent::Started { route, .. }) if route == "/v1/messages"),
        "{events:?}"
    );
    assert!(
        events.contains(&AdapterEvent::Response { status: 401 }),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::ErrorEnvelope { error }
                if error.status.as_deref() == Some("authentication_error")
        )),
        "the envelope's own fields are projected: {events:?}"
    );
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(401),
                retryable: false,
            }
        }),
        "{events:?}"
    );
    let rendered = trace_json(trace);
    assert!(
        !rendered.contains(BASIC_PROMPT) && !rendered.contains("sk-invalid"),
        "neither the request nor the credential reaches the trace"
    );
}
