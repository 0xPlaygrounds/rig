use super::*;
use aws_credential_types::Credentials;
use rig_core::completion::CompletionRequest;
use rig_core::test_utils::RecordingHttpClient;

const MESSAGES_REPLY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;

const CHAT_REPLY: &str = r#"{"id":"c1","object":"chat.completion","created":0,"model":"openai.gpt-oss-120b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

fn sigv4() -> Credential {
    Credential::SigV4(SharedCredentialsProvider::new(Credentials::new(
        "AKIDEXAMPLE",
        "secret",
        Some("session".to_owned()),
        None,
        "test",
    )))
}

fn header<'a>(headers: &'a rig_core::http_client::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// Claude on `bedrock-mantle` goes to the Messages path, signed for
/// `bedrock-mantle` with the session token and the `anthropic-version`
/// header, and a request without `max_tokens` takes the model's limit.
#[tokio::test]
async fn messages_on_mantle_are_signed_for_bedrock_mantle() {
    let http = RecordingHttpClient::new(MESSAGES_REPLY);
    let bedrock = BedrockEndpoint::new(Host::Mantle, "us-east-1", sigv4(), http.clone());

    bedrock
        .messages(MANTLE_CLAUDE_SONNET_5)
        .call(CompletionRequest::new("hi"))
        .await
        .expect("a reply");

    let requests = http.requests();
    let request = requests.first().expect("one request");
    assert_eq!(
        request.uri,
        "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
    );
    let authorization = header(&request.headers, "authorization").expect("signed");
    assert!(authorization.contains("/us-east-1/bedrock-mantle/aws4_request"));
    assert!(authorization.contains("anthropic-version"));
    assert!(authorization.contains("x-amz-security-token"));
    assert_eq!(header(&request.headers, "x-api-key"), None);
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("JSON");
    assert!(
        body.get("max_tokens")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|tokens| tokens > 0)
    );
}

/// An API key goes out as a bearer token, unsigned, to the runtime's
/// OpenAI-compatible path.
#[tokio::test]
async fn chat_on_runtime_with_an_api_key_is_bearer_authenticated() {
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let bedrock = BedrockEndpoint::new(
        Host::Runtime,
        "us-west-2",
        Credential::ApiKey("bedrock-key".into()),
        http.clone(),
    );

    bedrock
        .openai()
        .chat(RUNTIME_GPT_OSS_120B)
        .call(CompletionRequest::new("hi"))
        .await
        .expect("a reply");

    let requests = http.requests();
    let request = requests.first().expect("one request");
    assert_eq!(
        request.uri,
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    );
    assert_eq!(
        header(&request.headers, "authorization"),
        Some("Bearer bedrock-key")
    );
    assert_eq!(header(&request.headers, "x-amz-date"), None);
}

/// A signed request on the runtime is scoped to `bedrock`, and `openai_at`
/// moves the OpenAI-compatible path.
#[tokio::test]
async fn openai_at_keeps_the_host_and_its_signing_name() {
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let bedrock = BedrockEndpoint::new(Host::Runtime, "us-east-1", sigv4(), http.clone());

    bedrock
        .openai_at("/custom/v1")
        .chat(RUNTIME_GPT_OSS_20B)
        .call(CompletionRequest::new("hi"))
        .await
        .expect("a reply");

    let requests = http.requests();
    let request = requests.first().expect("one request");
    assert_eq!(
        request.uri,
        "https://bedrock-runtime.us-east-1.amazonaws.com/custom/v1/chat/completions"
    );
    let authorization = header(&request.headers, "authorization").expect("signed");
    assert!(authorization.contains("/us-east-1/bedrock/aws4_request"));
}
