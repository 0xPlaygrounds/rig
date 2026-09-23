//! The real Rig Vertex AI [`CompletionModel`] driving the real
//! `google-cloud-aiplatform-v1` `PredictionService` against a local HTTP
//! endpoint.
//!
//! Every test here goes through the production request conversion, the
//! production SDK client and the production response mapping. The only
//! substitutions are the socket the SDK dials and the credentials it presents
//! — both supplied by the host through
//! [`ClientBuilder::with_prediction_service`], which is the seam these tests
//! exist to exercise. No Application Default Credentials are read and no
//! Google endpoint is contacted.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

mod support;

use google_cloud_aiplatform_v1::client::PredictionService;
use rig_core::completion::{CompletionModel as _, CompletionRequest, ToolDefinition};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Message, Text, ToolChoice, UserContent};
use rig_vertexai::Client;
use rig_vertexai::client::VertexAiClientError;
use rig_vertexai::completion::CompletionModel;
use std::time::Duration;
use support::{LocalEndpoint, Reply, SentinelCredentials, text_response, tool_call_response};

const PROJECT: &str = "rig-test-project";
const LOCATION: &str = "us-central1";
const MODEL: &str = rig_vertexai::completion::GEMINI_2_5_FLASH;

/// A Rig client whose SDK client is host-built against `endpoint`.
async fn hosted_model(
    endpoint: &LocalEndpoint,
    credentials: &SentinelCredentials,
) -> CompletionModel {
    let service = PredictionService::builder()
        .with_endpoint(endpoint.url())
        .with_attempt_timeout(std::time::Duration::from_secs(60))
        .with_credentials(credentials.credentials())
        .build()
        .await
        .expect("the host builds the SDK client against the local endpoint");
    let client = Client::builder()
        .with_project(PROJECT)
        .with_location(LOCATION)
        .with_prediction_service(service)
        .build()
        .expect("supplied client plus explicit project and location");
    CompletionModel::new(client, MODEL)
}

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

/// The whole unary path: Rig's request conversion, the SDK's URL and query
/// construction, the credentials it attaches, and the response mapping back
/// into Rig's types.
#[tokio::test]
async fn unary_completion_converts_the_request_and_maps_the_response() {
    let endpoint = LocalEndpoint::spawn([Reply::ok(tool_call_response(
        "lookup_weather",
        serde_json::json!({"city": "Lisbon"}),
    ))])
    .await;
    let credentials = SentinelCredentials::rotating("sentinel-token");
    let model = hosted_model(&endpoint, &credentials).await;

    let mut request = request("weather in Lisbon?");
    request.chat_history.insert(
        0,
        Message::System {
            content: "you are terse".into(),
        },
    );
    request.temperature = Some(0.25);
    request.max_tokens = Some(64);
    request.tool_choice = Some(ToolChoice::Required);
    request.tools = vec![ToolDefinition {
        name: "lookup_weather".to_string(),
        description: "look up the weather".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
    }];

    let response = model
        .completion(request)
        .await
        .expect("completion succeeds");

    let captured = endpoint.requests();
    let [captured] = captured.as_slice() else {
        panic!("exactly one RPC, got {}", captured.len());
    };
    assert_eq!(captured.method, "POST");
    assert_eq!(
        captured.path(),
        format!(
            "/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
        ),
        "the model path is built from the explicitly configured project and location"
    );
    assert!(
        captured.query().is_some_and(|query| query.contains("alt")),
        "the SDK's own JSON encoding query survives: {:?}",
        captured.query()
    );
    assert_eq!(
        captured.header("authorization"),
        Some(format!("Bearer {}", credentials.token(1)).as_str()),
        "the supplied credentials authorize the request"
    );

    let body = captured.json();
    assert_eq!(
        body["contents"],
        serde_json::json!([{"role": "user", "parts": [{"text": "weather in Lisbon?"}]}]),
        "the system message is hoisted out of contents"
    );
    assert_eq!(
        body["systemInstruction"],
        serde_json::json!({"role": "user", "parts": [{"text": "you are terse"}]})
    );
    assert_eq!(
        body["tools"],
        serde_json::json!([{
            "functionDeclarations": [{
                "name": "lookup_weather",
                "description": "look up the weather",
                "parametersJsonSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }]
        }])
    );
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"]["mode"],
        serde_json::json!(2),
        "`ToolChoice::Required` is the `ANY` function-calling mode"
    );
    assert_eq!(
        body["generationConfig"]["temperature"],
        serde_json::json!(0.25)
    );
    assert_eq!(
        body["generationConfig"]["maxOutputTokens"],
        serde_json::json!(64)
    );

    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one tool call, got {:?}", response.choice);
    };
    assert_eq!(call.function.name, "lookup_weather");
    assert_eq!(
        call.function.arguments,
        serde_json::json!({"city": "Lisbon"})
    );
}

/// Credentials are a client-lifetime resource that re-issues per request: two
/// completions on one model present two different sentinels, without either
/// completion rebuilding the client.
#[tokio::test]
async fn rotated_credentials_are_presented_per_request() {
    let endpoint = LocalEndpoint::spawn([
        Reply::ok(text_response("first")),
        Reply::ok(text_response("second")),
    ])
    .await;
    let credentials = SentinelCredentials::rotating("rotating-token");
    let model = hosted_model(&endpoint, &credentials).await;

    model.completion(request("one")).await.expect("first call");
    model.completion(request("two")).await.expect("second call");

    let captured = endpoint.requests();
    let tokens: Vec<_> = captured
        .iter()
        .map(|request| request.header("authorization").unwrap_or_default())
        .collect();
    assert_eq!(
        tokens,
        vec![
            format!("Bearer {}", credentials.token(1)),
            format!("Bearer {}", credentials.token(2)),
        ],
        "each request carries the freshly issued sentinel"
    );
    assert_eq!(credentials.issued(), 2);
}

/// A credential that permanently refuses to issue a token fails the call
/// before anything reaches the wire: no unauthenticated request escapes, and
/// a refusal that is not transient is not retried.
#[tokio::test]
async fn a_credential_failure_keeps_the_request_off_the_wire() {
    let endpoint = LocalEndpoint::spawn([Reply::ok(text_response("unreachable"))]).await;
    let credentials = SentinelCredentials::rotating("never-issued").failing_from(1);
    let model = hosted_model(&endpoint, &credentials).await;

    let error = model
        .completion(request("hello"))
        .await
        .expect_err("the credentials refuse to issue a token");

    assert_eq!(endpoint.request_count(), 0, "no request was sent");
    assert!(
        !error.to_string().contains("never-issued"),
        "the failure names no token: {error}"
    );
}

/// A provider error reply is preserved as the provider sent it, with the HTTP
/// status and the RPC code the SDK read from it. Rig does not sanitize the
/// provider's body — it reproduces it — so what this asserts is that the
/// *request's* credentials are not part of what the error carries.
#[tokio::test]
async fn a_provider_error_preserves_the_reply_and_carries_no_request_credentials() {
    let body = serde_json::json!({
        "error": {
            "code": 400,
            "message": "Unable to submit request because the model is not supported",
            "status": "INVALID_ARGUMENT",
        }
    })
    .to_string();
    let endpoint = LocalEndpoint::spawn([Reply::error(400, body)]).await;
    let credentials = SentinelCredentials::rotating("secret-token");
    let model = hosted_model(&endpoint, &credentials).await;

    let error = model
        .completion(request("hello"))
        .await
        .expect_err("the endpoint refuses the request");

    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    let preserved = error
        .provider_response_body()
        .expect("the provider's reply is preserved");
    assert!(
        preserved.contains("the model is not supported"),
        "the provider's own message survives: {preserved}"
    );
    assert!(!error.is_retryable(), "400 is the caller's fault");
    assert!(
        !format!("{error:?}").contains("secret-token"),
        "the error is built from the reply, which never contained the request's token"
    );
    assert_eq!(credentials.issued(), 1);
}

/// Dropping the completion future releases the in-flight RPC: the endpoint,
/// still holding the request open, sees the connection go away. The client and
/// its credentials are not operation-scoped, so the next completion works.
#[tokio::test]
async fn a_cancelled_completion_releases_its_rpc_and_leaves_the_client_usable() {
    let endpoint = LocalEndpoint::spawn([Reply::Hang, Reply::ok(text_response("after"))]).await;
    let credentials = SentinelCredentials::rotating("cancel-token");
    let model = hosted_model(&endpoint, &credentials).await;

    // Arrival, not elapsed time, establishes that there is real work to cancel.
    {
        let completion = model.completion(request("abandoned"));
        tokio::pin!(completion);
        tokio::select! {
            result = &mut completion => panic!("held RPC completed before cancellation: {result:?}"),
            () = endpoint.wait_for_requests(1) => {}
        }
    }

    endpoint.wait_for_disconnects(1).await;
    assert_eq!(
        endpoint.disconnects(),
        1,
        "the abandoned RPC's connection was actually closed, not left dangling"
    );

    let response = tokio::time::timeout(Duration::from_secs(10), model.completion(request("next")))
        .await
        .expect("subsequent completion deadline")
        .expect("the shared client and its credentials survived the cancellation");
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "after")
    );
    assert_eq!(credentials.issued(), 2);
}

/// This integration implements no streaming path, and says so rather than
/// pretending: the model reports the operation as unsupported without dialing
/// anything.
#[tokio::test]
async fn streaming_is_explicitly_unsupported() {
    let endpoint = LocalEndpoint::spawn([]).await;
    let credentials = SentinelCredentials::rotating("stream-token");
    let model = hosted_model(&endpoint, &credentials).await;

    let error = model
        .stream(request("stream please"))
        .await
        .err()
        .expect("streaming is not implemented for Vertex AI");
    assert!(
        matches!(&error, ProviderError::Provider(message) if message.contains("Streaming is not supported")),
        "unexpected error: {error}"
    );
    assert_eq!(endpoint.request_count(), 0);
    assert_eq!(credentials.issued(), 0);
}

/// A supplied client already fixes its credentials. Configuring both is a
/// contradiction, and is refused at build time rather than by silently
/// ignoring one of them.
#[tokio::test]
async fn supplying_both_a_client_and_credentials_is_refused() {
    let endpoint = LocalEndpoint::spawn([]).await;
    let credentials = SentinelCredentials::rotating("conflicting-token");
    let service = PredictionService::builder()
        .with_endpoint(endpoint.url())
        .with_attempt_timeout(std::time::Duration::from_secs(60))
        .with_credentials(credentials.credentials())
        .build()
        .await
        .expect("host-built client");

    let error = Client::builder()
        .with_project(PROJECT)
        .with_location(LOCATION)
        .with_prediction_service(service)
        .with_credentials(credentials.credentials())
        .build()
        .expect_err("the two inputs contradict each other");
    assert!(matches!(error, VertexAiClientError::ConflictingCredentials));
}

/// On the deferred path Rig builds the SDK client on the first completion, so
/// an initialization failure surfaces there — as a Rig-side provider error,
/// not as a provider reply — and the cached failure is reported identically to
/// every later caller.
#[tokio::test]
async fn deferred_client_initialization_failure_surfaces_on_first_use() {
    // A credential claiming another universe domain is rejected by the SDK
    // while it constructs the client, before any request exists.
    let credentials =
        SentinelCredentials::rotating("unused-token").with_universe_domain("test-universe.invalid");
    let client = Client::builder()
        .with_project(PROJECT)
        .with_location(LOCATION)
        .with_credentials(credentials.credentials())
        .build()
        .expect("building the Rig client resolves no SDK client yet");
    let model = CompletionModel::new(client, MODEL);

    for attempt in 1..=2 {
        let error = model
            .completion(request("hello"))
            .await
            .expect_err("the SDK client cannot be built");
        assert!(
            matches!(&error, ProviderError::Provider(message)
                if message.contains("universe domain")),
            "attempt {attempt}: unexpected error: {error}"
        );
        assert_eq!(
            error.provider_response_body(),
            None,
            "attempt {attempt}: a setup failure is not a provider reply"
        );
    }
    assert_eq!(
        credentials.issued(),
        0,
        "no token was ever minted for a client that never existed"
    );
}
