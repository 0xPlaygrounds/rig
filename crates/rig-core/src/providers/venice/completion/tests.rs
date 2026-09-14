use super::*;

/// Serialization shape of the request block is definitory, not observed:
/// the cassette suite pins that Venice *accepts* it, this pins that
/// unset fields stay off the wire entirely rather than being sent null.
#[test]
fn venice_parameters_only_serialize_set_fields() {
    let params = VeniceParameters::new()
        .enable_web_search(WebSearchMode::Auto)
        .disable_thinking(true);

    let json = serde_json::to_value(&params).expect("parameters should serialize");

    assert_eq!(
        json,
        serde_json::json!({
            "enable_web_search": "auto",
            "disable_thinking": true,
        })
    );
}

#[test]
fn venice_parameters_wrap_into_additional_params() {
    let json = VeniceParameters::new()
        .character_slug("venice")
        .into_additional_params();

    assert_eq!(
        json,
        serde_json::json!({ "venice_parameters": { "character_slug": "venice" } })
    );
}

/// Response decoding is pinned by cassettes; this asserts the flattened
/// wrapper keeps *both* halves — an OpenAI-only decode would silently
/// drop citations and cost.
#[test]
fn completion_response_preserves_venice_blocks() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3-5-9b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "cost": {"usd": 0.000_002_65, "diem": 0.0},
        "venice_parameters": {
            "enable_web_search": "on",
            "enable_e2ee": true,
            "web_search_citations": [{
                "title": "Rust",
                "url": "https://example.com",
                "content": "text",
                "date": ""
            }]
        }
    });

    let response: CompletionResponse =
        serde_json::from_value(body).expect("response should decode");

    assert_eq!(response.openai.id, "chatcmpl-1");
    assert_eq!(response.text_response().as_deref(), Some("hi"));
    assert_eq!(response.cost.expect("cost").diem, 0.0);
    assert_eq!(response.web_search_citations().len(), 1);
    assert_eq!(response.web_search_citations()[0].title, "Rust");
    assert_eq!(
        response
            .venice_parameters
            .expect("venice parameters")
            .parameters
            .enable_web_search,
        Some(WebSearchMode::On)
    );
}
