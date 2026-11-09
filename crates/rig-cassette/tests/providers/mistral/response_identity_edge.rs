//! Response identity for Mistral.
//!
//! Mistral labels its transport request id `mistral-correlation-id` and sends
//! it on every response, and the `MISTRAL` dialect names it. The header must be
//! on the cassette harness's response-header allowlist, or the recorded fixture
//! would carry no id for these cells to find.

use anyhow::Result;
use futures::StreamExt;
use rig::error::ErrorReport;
use rig::providers::mistral;

use super::support::with_mistral_cassette_result;
use rig::completion::CompletionRequest;

/// The id is a UUID minted per call, so the assertion is on its presence and
/// shape rather than a literal the scrubber placeholders anyway.
fn assert_is_request_id(id: Option<&str>) {
    let id = id.expect("Mistral reports a transport request id on every response");
    assert!(
        !id.is_empty(),
        "the captured request id must not be empty; an empty id is indistinguishable from none"
    );
}

/// The streaming twin: an in-band stream failure is an `ErrorReport`, which
/// carries the same preserved id and body as the `ProviderError`.
fn assert_report_keeps_id_and_body(report: &ErrorReport) {
    assert_is_request_id(report.provider_request_id());
    assert!(
        report.provider_response_body().is_some(),
        "the error body must survive alongside the id: {report:?}"
    );
}

/// The streaming twin of the cell above.
///
/// Derived from the recording rather than assumed: `stream()` returns `Ok` for
/// an unroutable model — Mistral answers the POST with a 400 that the SSE
/// layer surfaces on the first polled item, not at the handshake. The
/// assertion is that the id reaches the caller wherever the failure lands, so
/// a streamed turn's failures are no less diagnosable than its blocking twin's.
#[tokio::test]
async fn streaming_error_carries_the_correlation_id() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/streaming_error_carries_the_correlation_id",
        |client| async move {
            let error = match client
                .completion("definitely-not-a-model")
                .stream(CompletionRequest::new("Reply with exactly: identity probe"))
            {
                Err(error) => ErrorReport::from(&error),
                Ok(mut stream) => {
                    let mut failure = None;
                    while let Some(item) = stream.next().await {
                        if let Err(error) = item {
                            failure = Some(error);
                            break;
                        }
                    }
                    rig::ErrorReport::from(
                        &failure.expect("an unroutable model must fail the stream"),
                    )
                }
            };
            assert_report_keeps_id_and_body(&error);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The same on a 401, whose body is a different shape again (`{"detail": …}`
/// rather than Mistral's `{"object":"error", …}` envelope).
#[tokio::test]
async fn blocking_unauthorized_carries_the_correlation_id() -> Result<()> {
    super::support::with_mistral_cassette_bogus_key_result(
        "response_identity_edge/blocking_unauthorized_carries_the_correlation_id",
        |client| async move {
            let error = client
                .completion(mistral::MISTRAL_SMALL)
                .call(CompletionRequest::new("Reply with exactly: identity probe"))
                .await
                .expect_err("a bogus key must fail");
            assert_is_request_id(error.provider_request_id());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
