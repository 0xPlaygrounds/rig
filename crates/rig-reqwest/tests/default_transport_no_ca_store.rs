//! The shared transport never fails to construct: when the reqwest client
//! cannot be built, every send on it reports the build failure in-band.
//!
//! reqwest's rustls backend loads the platform CA store while building the
//! client; on a host with none (a bare `ubuntu` container, say)
//! `reqwest::Client::new()` panics with "No CA certificates were loaded from
//! the system". [`rig_reqwest::shared`] promises a transport, so that failure
//! must come back from the first call as `ProviderError::Http`.
//!
//! The store is emptied through the `SSL_CERT_FILE` / `SSL_CERT_DIR`
//! overrides rustls-native-certs honours, which is why this test is its own
//! binary (the variables are process-wide) and Linux-only (the macOS and
//! Windows verifiers read their keychains instead). It also needs the
//! rustls backend to be the one reqwest builds with: with `native-tls`
//! enabled as well (CI's `--all-features`), reqwest defaults to OpenSSL,
//! which does not read the store at build time, so there is no failure to
//! observe and the test compiles to nothing.

#![cfg(all(target_os = "linux", feature = "rustls", not(feature = "native-tls")))]

use futures::StreamExt;
use rig_core::completion::CompletionRequestBuilder;
use rig_core::error::{ErrorKind, ProviderError};
use rig_core::providers::openai::OpenAI;
use rig_core::wire::Wire as _;

fn empty_the_ca_store() {
    // SAFETY: this test binary has one test and no other threads read the
    // environment before it runs.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", "/nonexistent/rig-no-ca.pem");
        std::env::set_var("SSL_CERT_DIR", "/nonexistent/rig-no-ca");
    }
}

#[tokio::test]
async fn the_shared_transport_reports_a_missing_ca_store_on_send() {
    empty_the_ca_store();

    let model = OpenAI::new("test-key")
        .completion("gpt-5.2")
        .on(rig_reqwest::shared());
    let request = CompletionRequestBuilder::new("hello").build();
    let outcome = match model.call(request.clone()).await {
        Err(ProviderError::Http(error)) => error.to_string(),
        Err(other) => format!("wrong variant: {other}"),
        Ok(_) => "sent a request with no CA store".to_owned(),
    };
    assert!(
        outcome.contains("CA certificates"),
        "expected the Http error naming the CA store, got: {outcome}"
    );

    // A stream opens (nothing is sent until it is polled) and the same
    // failure is its first item.
    let mut stream = model
        .stream(request)
        .expect("a stream opens before sending");
    let first = stream
        .next()
        .await
        .expect("the build failure is the first item");
    let report = first.expect_err("the first item is the failure");
    assert_eq!(report.kind, ErrorKind::Http, "{report:?}");
    assert!(
        report.message.contains("CA certificates"),
        "expected the report to name the CA store, got: {}",
        report.message
    );
}
