//! Who owns the Tokio runtime a Vertex AI client needs, and for how long.
//!
//! Two facts about the real `google-cloud-auth` credential chain drive the
//! contract documented on [`rig_vertexai::Client::from_env`]:
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

use google_cloud_aiplatform_v1::client::PredictionService;
use rig_core::completion::{CompletionModel as _, CompletionRequest};
use rig_core::message::{AssistantContent, Message, Text, UserContent};
use rig_vertexai::Client;
use rig_vertexai::completion::CompletionModel;
use support::{LocalEndpoint, Reply, SentinelCredentials, text_response};

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::User {
            content: vec![UserContent::Text(Text::new(prompt.to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Fact (1), against the real credential chain: constructing metadata-service
/// credentials — the branch ADC falls back to, and the one Rig's `from_env`
/// reaches with no credentials file present — spawns, so it fails outside a
/// runtime context. This is why `Client::from_env` documents a runtime
/// requirement instead of looking like an inert constructor.
#[test]
fn building_adc_credentials_requires_a_runtime_context() {
    let rig_error = Client::builder()
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

/// Fact (2): a host that builds the client on its own retained runtime can
/// hand the model to a worker thread that owns no runtime of its own, as long
/// as the work is driven on the retained one. The SDK client is prepared
/// asynchronously up front; the request is polled later, from the other
/// thread, through the runtime handle.
#[test]
fn a_worker_thread_completes_through_the_retained_runtime() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");

    let credentials = SentinelCredentials::rotating("retained-token");
    let (endpoint, model) = runtime.block_on({
        let credentials = credentials.clone();
        async move {
            let endpoint =
                LocalEndpoint::spawn([Reply::ok(text_response("from the worker"))]).await;
            let service = PredictionService::builder()
                .with_endpoint(endpoint.url())
                .with_attempt_timeout(std::time::Duration::from_secs(60))
                .with_credentials(credentials.credentials())
                .build()
                .await
                .expect("SDK client prepared on the host runtime");
            let client = Client::builder()
                .with_project("rig-test-project")
                .with_location("us-central1")
                .with_prediction_service(service)
                .build()
                .expect("client");
            (endpoint, CompletionModel::new(client, "gemini-2.5-flash"))
        }
    });

    let handle = runtime.handle().clone();
    let worker = std::thread::spawn(move || {
        assert!(
            tokio::runtime::Handle::try_current().is_err(),
            "the worker thread is not itself a runtime"
        );
        handle.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                model.completion(request("hello from a worker")),
            )
            .await
            .expect("worker completion deadline")
        })
    });
    let response = worker
        .join()
        .expect("worker thread")
        .expect("the completion ran on the retained runtime");

    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "from the worker")
    );
    assert_eq!(endpoint.request_count(), 1);
    assert_eq!(credentials.issued(), 1);

    // Shutdown ordering: the operation finished, so releasing the runtime
    // releases only shared resources.
    drop(endpoint);
    runtime.shutdown_background();
}
