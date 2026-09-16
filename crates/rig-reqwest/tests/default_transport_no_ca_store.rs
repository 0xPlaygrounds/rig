//! The default-transport constructors return an error, not a panic, when the
//! bundled reqwest client cannot be built.
//!
//! reqwest's rustls backend loads the platform CA store while building the
//! client; on a host with none (a bare `ubuntu` container, say)
//! `reqwest::Client::new()` panics with "No CA certificates were loaded from
//! the system". [`DefaultTransport::bound`] promises a `Result`, so that
//! failure must come back as `ProviderClientError::Http`.
//!
//! `bound()` is the one place the bundled transport is built now: the
//! provider config itself is infallible data, so a construction error can
//! only arise where a socket is made. The two spellings below are the two
//! ways a caller reaches it — a config read from the environment, and one
//! built from a literal key.
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

use rig_core::client::ProviderClientError;
use rig_core::providers::openai::OpenAI;
use rig_reqwest::prelude::*;

fn empty_the_ca_store() {
    // SAFETY: this test binary has one test and no other threads read the
    // environment before it runs.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", "/nonexistent/rig-no-ca.pem");
        std::env::set_var("SSL_CERT_DIR", "/nonexistent/rig-no-ca");
        std::env::set_var("OPENAI_API_KEY", "test-key");
    }
}

#[test]
fn every_default_transport_constructor_reports_a_missing_ca_store() {
    empty_the_ca_store();

    let from_env = OpenAI::from_env()
        .expect("reading the environment does not need a transport")
        .bound();
    let new = OpenAI::new("test-key").bound();

    for (name, result) in [("from_env", from_env.map(drop)), ("new", new.map(drop))] {
        let outcome = match result {
            Err(ProviderClientError::Http(error)) => error.to_string(),
            Err(other) => format!("wrong variant: {other}"),
            Ok(()) => "built a client with no CA store".to_owned(),
        };
        assert!(
            outcome.contains("CA certificates"),
            "{name}: expected the Http error naming the CA store, got: {outcome}"
        );
    }
}
