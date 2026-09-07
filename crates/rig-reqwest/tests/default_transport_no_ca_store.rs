//! The default-transport constructors return an error, not a panic, when the
//! bundled reqwest client cannot be built.
//!
//! reqwest's rustls backend loads the platform CA store while building the
//! client; on a host with none (a bare `ubuntu` container, say) `Client::new()`
//! panics with "No CA certificates were loaded from the system". `from_env`,
//! `new`, `from_val` and `build` all promise a `Result`, so that failure must
//! come back as `ProviderClientError::Http`.
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
use rig_core::providers::openai;
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

    let from_env = openai::Client::from_env();
    let new = openai::Client::new("test-key");
    let built = openai::Client::builder().api_key("test-key").build();

    for (name, result) in [
        ("from_env", from_env.map(drop)),
        ("new", new.map(drop)),
        ("build", built.map(drop)),
    ] {
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
