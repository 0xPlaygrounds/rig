use super::*;
use crate::client::CompletionClient;
use crate::completion::CompletionModel as _;
use crate::test_utils::RecordingHttpClient;

const REQUEST_ID: &str = "req_unit_copilot_0001";

/// A chat-completions body carrying `system_fingerprint`, which the
/// normalized response provably lacks.
const CHAT_BODY: &str = r#"{
        "id": "chatcmpl-copilot-raw",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-11-20",
        "system_fingerprint": "fp_copilot_chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    }"#;

/// A Responses body carrying `service_tier`, which the normalized
/// response provably lacks.
const RESPONSES_BODY: &str = r#"{
        "id": "resp_copilot_raw",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-5.3-codex",
        "service_tier": "default",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 7
        },
        "output": [{
            "type": "message",
            "id": "msg_copilot_raw",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }],
        "tools": []
    }"#;

fn model(model: &str, body: &'static str) -> CompletionModel<RecordingHttpClient> {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
    let http_client =
        RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, body, headers);
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    client.completion_model(model)
}

/// Run one completion for a route and check the shared capture contract:
/// `raw` deserializes into [`CopilotCompletionResponse`] under the
/// expected route tag and re-serializes identically; re-normalizing the
/// capture (with the header id reattached, exactly as `completion()`
/// does) reproduces every normalized field; and the response reports the
/// header's id.
async fn assert_capture_contract(
    model: &CompletionModel<RecordingHttpClient>,
    expected_api_tag: &str,
) -> (completion::CompletionResponse, CopilotCompletionResponse) {
    let response = model
        .completion(model.completion_request("hello").build())
        .await
        .expect("completion");
    let raw = &response.raw;
    assert_eq!(raw["api"], expected_api_tag);
    let typed: CopilotCompletionResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "the capture must be exactly what the route-tagged raw type serializes to"
    );

    let renormalized = typed
        .clone()
        .normalize(PROVIDER_NAME)
        .expect("re-normalize the capture")
        .with_optional_provider_request_id(Some(REQUEST_ID.to_string()));
    assert_eq!(response.identity(), renormalized.identity());
    assert_eq!(response.finish_reason(), renormalized.finish_reason());
    assert_eq!(response.model, renormalized.model);
    assert_eq!(response.usage, renormalized.usage);
    assert_eq!(response.choice, renormalized.choice);
    assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    (response, typed)
}

/// Part A parity for one route: `raw_completion_with_request_id` →
/// `normalize` → `with_optional_provider_request_id` reproduces
/// `completion()` on identity, finish reason, model and usage, and the
/// id is the header on both.
async fn assert_parity_contract(model: &CompletionModel<RecordingHttpClient>) {
    let (raw, id) = model
        .raw_completion_with_request_id(model.completion_request("hello").build())
        .await
        .expect("typed route");
    assert_eq!(id.as_deref(), Some(REQUEST_ID));
    let reassembled = raw
        .normalize(PROVIDER_NAME)
        .expect("normalize")
        .with_optional_provider_request_id(id);

    let normalized = model
        .completion(model.completion_request("hello").build())
        .await
        .expect("normalized route");

    assert_eq!(reassembled.identity(), normalized.identity());
    assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
    assert_eq!(reassembled.model, normalized.model);
    assert_eq!(reassembled.usage, normalized.usage);
    assert_eq!(reassembled.provider_request_id.as_deref(), Some(REQUEST_ID));
    assert_eq!(normalized.provider_request_id.as_deref(), Some(REQUEST_ID));
    assert_eq!(normalized.provider, PROVIDER_NAME);
}

/// Chat route: the capture is tagged `api: chat`, wraps the shared OpenAI
/// chat wire type, and keeps `system_fingerprint`.
#[tokio::test]
async fn chat_route_raw_round_trips_into_the_route_tagged_type() {
    let model = model("gpt-4o", CHAT_BODY);

    let (response, typed) = assert_capture_contract(&model, "chat").await;

    let CopilotCompletionResponse::Chat(chat) = typed else {
        panic!("the chat route must capture the chat variant");
    };
    assert_eq!(chat.system_fingerprint.as_deref(), Some("fp_copilot_chat"));
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
    assert_eq!(
        response.identity().response_id.as_deref(),
        Some("chatcmpl-copilot-raw")
    );
}

/// Chat route Part A: the wire type has no id slot, so only the pair
/// reproduces `completion()` — this is the case the method exists for.
#[tokio::test]
async fn chat_route_raw_completion_with_request_id_reproduces_completion() {
    let model = model("gpt-4o", CHAT_BODY);

    assert_parity_contract(&model).await;

    // And plain `raw_completion` → `normalize` provably lacks the id:
    // the reason the pair is public.
    let raw = model
        .raw_completion(model.completion_request("hello").build())
        .await
        .expect("typed route");
    let normalized = raw.normalize(PROVIDER_NAME).expect("normalize");
    assert_eq!(normalized.provider_request_id, None);
}

/// Responses route: the capture is tagged `api: responses` and wraps the
/// Responses wire type, whose hand-written `Serialize` mirrors the body
/// (`service_tier` kept; the stamped transport id, which is not body,
/// deliberately not emitted — so the deserialized capture reports `None`
/// there while the normalized response beside it carries the header).
#[tokio::test]
async fn responses_route_raw_round_trips_into_the_route_tagged_type() {
    let model = model("gpt-5.3-codex", RESPONSES_BODY);

    let (response, typed) = assert_capture_contract(&model, "responses").await;

    let CopilotCompletionResponse::Responses(responses) = typed else {
        panic!("the responses route must capture the responses variant");
    };
    assert!(matches!(
        responses.additional_parameters.service_tier,
        Some(responses_api::OpenAIServiceTier::Default)
    ));
    assert_eq!(responses.provider_request_id, None);
    assert_eq!(
        response.identity().message_id.as_deref(),
        Some("msg_copilot_raw")
    );
    assert_eq!(
        response.identity().response_id.as_deref(),
        Some("resp_copilot_raw")
    );
}

/// Responses route Part A: the wire type carries the id itself, so the
/// pair's second element equals the raw type's own id and reattaching it
/// is a no-op — the same pair still reproduces `completion()`.
#[tokio::test]
async fn responses_route_raw_completion_with_request_id_reproduces_completion() {
    let model = model("gpt-5.3-codex", RESPONSES_BODY);

    assert_parity_contract(&model).await;

    let (raw, id) = model
        .raw_completion_with_request_id(model.completion_request("hello").build())
        .await
        .expect("typed route");
    let CopilotCompletionResponse::Responses(responses) = &raw else {
        panic!("codex models route to /responses");
    };
    assert_eq!(responses.provider_request_id, id);
    assert_eq!(id.as_deref(), Some(REQUEST_ID));
}

/// Both variants of the route-tagged unary raw type round-trip through
/// serde, hand-built from parsed wire bodies rather than through the
/// transport: the internally tagged enum has to merge its `api` tag into
/// whatever the inner type serializes as, and the responses variant's
/// inner type serializes through a hand-written `Serialize` (with a
/// flattened tail) rather than a derive.
#[test]
fn copilot_completion_response_round_trips_both_variants() {
    let chat: openai::completion::CompletionResponse =
        serde_json::from_str(CHAT_BODY).expect("chat body parses");
    let responses: responses_api::CompletionResponse =
        serde_json::from_str(RESPONSES_BODY).expect("responses body parses");

    for (variant, tag) in [
        (CopilotCompletionResponse::Chat(Box::new(chat)), "chat"),
        (
            CopilotCompletionResponse::Responses(Box::new(responses)),
            "responses",
        ),
    ] {
        let value = serde_json::to_value(&variant).expect("serialize");
        assert_eq!(value["api"], tag);
        let back: CopilotCompletionResponse =
            serde_json::from_value(value.clone()).expect("deserialize");
        assert_eq!(
            serde_json::to_value(&back).expect("re-serialize"),
            value,
            "{tag}: the route-tagged raw type must round-trip"
        );
        assert_eq!(
            back.normalize(PROVIDER_NAME).expect("normalize").provider,
            PROVIDER_NAME
        );
    }
}
