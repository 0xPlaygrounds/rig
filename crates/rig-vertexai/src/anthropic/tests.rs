use super::*;
use google_cloud_auth::credentials::{CredentialsProvider, EntityTag};
use rig_core::message::AssistantContent;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

/// Google credentials that always sign with one bearer token.
#[derive(Debug)]
struct Bearer;

impl CredentialsProvider for Bearer {
    async fn headers(
        &self,
        _extensions: http::Extensions,
    ) -> Result<CacheableResource<http::HeaderMap>, google_cloud_auth::errors::CredentialsError>
    {
        let mut data = http::HeaderMap::new();
        data.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_static("Bearer test-token"),
        );
        Ok(CacheableResource::New {
            entity_tag: EntityTag::new(),
            data,
        })
    }

    async fn universe_domain(&self) -> Option<String> {
        None
    }
}

fn client(location: &str, http: SequencedHttpClient) -> anyhow::Result<VertexAi> {
    Ok(VertexAi::builder()
        .with_project("proj")
        .with_location(location)
        .with_credentials(Bearer.into())
        .with_http_client(http)
        .build()?)
}

const REPLY: &str = r#"{"id":"msg_vrtx_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1}}"#;

/// A unary request goes to the model's `rawPredict` URL with the
/// credentials' headers, and its body names no model but carries Vertex AI's
/// `anthropic_version`; the reply reads as a Messages reply.
#[tokio::test]
async fn a_call_posts_to_raw_predict_with_google_auth() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([MockHttpResponse::success(REPLY)]);
    let response = client("global", http.clone())?
        .anthropic("claude-sonnet-4-6")
        .call(CompletionRequest::new("hello"))
        .await?;
    anyhow::ensure!(
        matches!(response.choice.first(), Some(AssistantContent::Text(text)) if text.text == "Hi")
    );

    let requests = http.requests();
    let request = requests
        .first()
        .ok_or_else(|| anyhow::anyhow!("no request"))?;
    anyhow::ensure!(
        request.uri
            == "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/publishers/anthropic/models/claude-sonnet-4-6:rawPredict",
        "{}",
        request.uri
    );
    anyhow::ensure!(
        request.headers.get(http::header::AUTHORIZATION)
            == Some(&http::HeaderValue::from_static("Bearer test-token"))
    );
    anyhow::ensure!(request.headers.get("x-api-key").is_none());
    anyhow::ensure!(request.headers.get("anthropic-version").is_none());
    anyhow::ensure!(request.headers.get("anthropic-beta").is_none());
    let body: serde_json::Value = serde_json::from_slice(&request.body)?;
    anyhow::ensure!(body.get("model").is_none(), "{body}");
    anyhow::ensure!(body.get("anthropic_version") == Some(&ANTHROPIC_VERSION_VERTEX.into()));
    anyhow::ensure!(
        body.get("max_tokens")
            .is_some_and(serde_json::Value::is_u64)
    );
    anyhow::ensure!(body.get("stream").is_none());
    Ok(())
}

/// A streamed request goes to `streamRawPredict` with `stream: true`,
/// configured betas travel as the `anthropic-beta` header, and the event
/// stream reads as Anthropic's.
#[tokio::test]
async fn a_stream_posts_to_stream_raw_predict() -> anyhow::Result<()> {
    let events = [
        r#"{"type":"message_start","message":{"id":"msg_vrtx_2","type":"message","role":"assistant","model":"claude-opus-4-1","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#,
        r#"{"type":"message_stop"}"#,
    ];
    let sse: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    let http =
        SequencedHttpClient::new([MockHttpResponse::success_typed(sse, "text/event-stream")]);
    let mut model = client("us-east5", http.clone())?.anthropic("claude-opus-4-1@20250805");
    model.wire.messages.provider = model.wire.messages.provider.with_beta("beta-a");
    let response = model
        .stream(CompletionRequest::new("hello"))?
        .finish()
        .await?;
    anyhow::ensure!(
        matches!(response.choice.first(), Some(AssistantContent::Text(text)) if text.text == "Hi")
    );

    let requests = http.requests();
    let request = requests
        .first()
        .ok_or_else(|| anyhow::anyhow!("no request"))?;
    anyhow::ensure!(
        request.uri
            == "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/publishers/anthropic/models/claude-opus-4-1@20250805:streamRawPredict",
        "{}",
        request.uri
    );
    anyhow::ensure!(
        request.headers.get("anthropic-beta") == Some(&http::HeaderValue::from_static("beta-a"))
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body)?;
    anyhow::ensure!(body.get("stream") == Some(&true.into()), "{body}");
    anyhow::ensure!(body.get("model").is_none(), "{body}");
    // The dated Vertex AI id takes Claude Opus 4.1's output limit from the
    // catalog, not the family default of 64000.
    anyhow::ensure!(body.get("max_tokens") == Some(&32_000.into()), "{body}");
    Ok(())
}

/// A request that names another dated Vertex AI id is sent to that model
/// and takes its catalog facts too.
#[tokio::test]
async fn a_request_naming_a_dated_id_takes_its_facts() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([MockHttpResponse::success(REPLY)]);
    let mut request = CompletionRequest::new("hello");
    request.model = Some("claude-opus-4-1@20250805".to_owned());
    client("global", http.clone())?
        .anthropic("claude-sonnet-4-6")
        .call(request)
        .await?;

    let requests = http.requests();
    let request = requests
        .first()
        .ok_or_else(|| anyhow::anyhow!("no request"))?;
    anyhow::ensure!(
        request
            .uri
            .ends_with("/models/claude-opus-4-1@20250805:rawPredict"),
        "{}",
        request.uri
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body)?;
    anyhow::ensure!(body.get("max_tokens") == Some(&32_000.into()), "{body}");
    Ok(())
}

/// `global` and the `us` and `eu` multi-regions have hosts of their own;
/// any other location is a regional host.
#[test]
fn each_location_has_its_host() {
    for (location, host) in [
        ("global", "https://aiplatform.googleapis.com/"),
        ("us", "https://aiplatform.us.rep.googleapis.com/"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com/"),
        (
            "europe-west1",
            "https://europe-west1-aiplatform.googleapis.com/",
        ),
    ] {
        let wire = RawPredict::new("proj", location, "claude-opus-4-6");
        assert_eq!(
            wire.messages.provider.base_url,
            format!("{host}v1/projects/proj/locations/{location}/publishers/anthropic/models")
        );
    }
}
