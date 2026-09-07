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
//! Windows verifiers read their keychains instead).

#![cfg(target_os = "linux")]

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
        match result {
            Err(ProviderClientError::Http(error)) => {
                let text = error.to_string();
                assert!(
                    text.contains("CA certificates"),
                    "{name}: expected the CA-store failure, got {text}"
                );
            }
            Err(other) => panic!("{name}: expected ProviderClientError::Http, got {other}"),
            Ok(()) => panic!("{name}: built a client with no CA store"),
        }
    }
}
