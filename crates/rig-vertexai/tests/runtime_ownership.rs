//! Who owns the Tokio runtime a Vertex AI client needs, and for how long.
//!
//! Two facts about the real `google-cloud-auth` credential chain drive the
//! contract documented on [`rig_vertexai::VertexAi::from_env`]:
//!
//! 1. Building credentials *spawns*. Every Application Default Credentials
//!    branch wraps its token provider in the auth crate's token cache, and
//!    that cache spawns its refresh task during construction.
//! 2. That task belongs to the runtime that accepted it, not to any one
//!    completion — so the runtime has to outlive the client, and a completion
//!    polled from elsewhere must be polled *on* it.
//!
//! The host-supplied `PredictionService` path is the way out of (1): it
//! resolves no credentials, so it spawns nothing of its own.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

mod support;

use rig_vertexai::VertexAi;

/// Fact (1), against the real credential chain: constructing metadata-service
/// credentials — the branch ADC falls back to, and the one Rig's `from_env`
/// reaches with no credentials file present — spawns, so it fails outside a
/// runtime context. This is why `VertexAi::from_env` documents a runtime
/// requirement instead of looking like an inert constructor.
#[test]
fn building_adc_credentials_requires_a_runtime_context() {
    let rig_error = VertexAi::builder()
        .with_project("offline-test")
        .with_location("global")
        .build()
        .unwrap_err();
    assert!(matches!(
        rig_error,
        rig_vertexai::client::VertexAiClientError::RuntimeRequired
    ));
    let outside = std::panic::catch_unwind(|| {
        google_cloud_auth::credentials::mds::Builder::default().build()
    });
    assert!(
        outside.is_err(),
        "credential construction spawns a refresh task, so it cannot happen off a runtime"
    );

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    // Enter but never drive this runtime: the real metadata refresh task may
    // be constructed, but must never contact a metadata service in this test.
    let inside = {
        let _entered = runtime.enter();
        google_cloud_auth::credentials::mds::Builder::default().build()
    };
    assert!(
        inside.is_ok(),
        "inside a runtime the same construction succeeds"
    );
}
