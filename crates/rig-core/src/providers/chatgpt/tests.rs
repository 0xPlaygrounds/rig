use super::*;

#[test]
fn test_parse_chatgpt_sse_completion() {
    let body = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"hi"}]}],"tools":[]}}
data: [DONE]"#;

    let response = responses_api::streaming::parse_sse_completion_body(body, "ChatGPT")
        .expect("expected response");
    assert_eq!(response.id, "resp_1");
    assert_eq!(response.model, "gpt-5");
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::chatgpt::Client::builder()
        .oauth()
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder()");
}

#[test]
fn test_merge_instructions_uses_default_when_missing() {
    assert_eq!(
        merge_instructions(DEFAULT_INSTRUCTIONS, None),
        DEFAULT_INSTRUCTIONS
    );
}

#[test]
fn test_merge_instructions_appends_existing_request_instructions() {
    let merged = merge_instructions(DEFAULT_INSTRUCTIONS, Some("Respond tersely."));
    assert!(merged.starts_with(DEFAULT_INSTRUCTIONS));
    assert!(merged.ends_with("Respond tersely."));
}

#[test]
fn test_merge_instructions_avoids_duplicate_default() {
    let merged = merge_instructions(
        DEFAULT_INSTRUCTIONS,
        Some("You are ChatGPT, a helpful AI assistant.\n\nRespond tersely."),
    );
    assert_eq!(
        merged,
        "You are ChatGPT, a helpful AI assistant.\n\nRespond tersely."
    );
}

fn chatgpt_conversion_request(chat_history: Vec<completion::Message>) -> ResponsesRequest {
    let client = crate::providers::chatgpt::Client::builder()
        .oauth()
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("client");
    let model = ResponsesCompletionModel::new(client, GPT_5_3_CODEX);

    model
        .openai_model()
        .create_completion_request(completion::CompletionRequest {
            model: Some("gpt-5.4".to_string()),
            chat_history: std::iter::once(completion::Message::system("System one"))
                .chain(chat_history)
                .collect(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .expect("request")
}

#[test]
fn test_conversion_lifts_leading_system_messages_into_instructions() {
    let request = chatgpt_conversion_request(vec![
        completion::Message::system("System two"),
        completion::Message::user("hi"),
    ]);

    assert_eq!(
        request.instructions.as_deref(),
        Some("System one\n\nSystem two")
    );
    assert_eq!(request.input.len(), 1);
}

#[test]
fn test_conversion_lifts_mid_conversation_system_messages() {
    let request = chatgpt_conversion_request(vec![
        completion::Message::user("hi"),
        completion::Message::system("Mid-conversation instruction"),
        completion::Message::user("again"),
    ]);

    assert_eq!(
        request.instructions.as_deref(),
        Some("System one\n\nMid-conversation instruction")
    );
    assert_eq!(request.input.len(), 2);
}

#[test]
fn test_create_request_merges_default_and_request_instructions() {
    let client = crate::providers::chatgpt::Client::builder()
        .oauth()
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("client");
    let model = ResponsesCompletionModel::new(client, GPT_5_3_CODEX);

    let request = model
        .create_request(completion::CompletionRequest {
            record_telemetry_content: false,
            model: None,
            chat_history: vec![
                completion::Message::system("Respond tersely."),
                completion::Message::user("hello"),
            ],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        })
        .expect("request");

    let expected = format!("{DEFAULT_INSTRUCTIONS}\n\nRespond tersely.");
    assert_eq!(request.instructions.as_deref(), Some(expected.as_str()));
}

#[test]
fn test_create_request_drops_temperature() {
    let client = crate::providers::chatgpt::Client::builder()
        .oauth()
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("client");
    let model = ResponsesCompletionModel::new(client, GPT_5_3_CODEX);

    let request = model
        .create_request(completion::CompletionRequest {
            model: None,
            chat_history: vec![completion::Message::user("hello")],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(0.5),
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .expect("request");

    assert!(request.temperature.is_none());
}

#[tokio::test]
async fn test_completion_response_from_sse_body_falls_back_to_streamed_text() {
    let body = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}
data: [DONE]"#;

    let raw_response = responses_api::streaming::parse_sse_completion_body(body, "ChatGPT")
        .expect("expected response");
    let response = responses_api::streaming::completion_response_from_sse_body(
        PROVIDER_NAME,
        body,
        raw_response,
    )
    .await
    .expect("fallback response");

    let text: String = response
        .choice
        .iter()
        .filter_map(|content| match content {
            completion::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(text, "hi");
    assert_eq!(response.usage.total_tokens, 2);
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::test_utils::RecordingHttpClient;

    let cases = [
        (
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"expired access token","type":"invalid_request_error"}}"#,
            "expired access token",
        ),
        (
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#,
            "rate limited",
        ),
    ];

    for (status, body, message) in cases {
        let http_client = RecordingHttpClient::with_error_response(status, body);
        let client = crate::providers::chatgpt::Client::builder()
            .api_key(ChatGPTAuth::AccessToken {
                access_token: "test-token".to_string(),
                account_id: Some("account-id".to_string()),
            })
            .http_client(http_client)
            .build()
            .expect("client should build");
        let model = client.completion_model(GPT_5_4);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        assert!(matches!(&error, CompletionError::HttpError(_)));
        assert_eq!(error.provider_response_status(), Some(status));
        assert_eq!(error.provider_response_body(), Some(body));
        assert!(
            error.to_string().contains(message),
            "error should include provider body: {error}"
        );
    }
}

/// Raw-capture tests for the ChatGPT model — the `other` seam shape: the
/// `/responses` endpoint answers a non-streaming call with an SSE body, so
/// `raw_completion` is a wire response *reassembled* from the event
/// stream, and the normalized path has an empty-`output` fallback that
/// rebuilds the choice from that same stream. The capture must be the
/// reassembled `responses_api::CompletionResponse` on both branches.
/// Driven end to end over the recording mock transport with an access
/// token, the same way the error-path tests above reach `completion()`.
mod raw_capture {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::test_utils::RecordingHttpClient;

    /// A complete turn: the terminal `response.completed` carries the
    /// assembled output plus `service_tier`, which the normalized
    /// response provably lacks.
    const SSE_BODY: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[{"type":"message","id":"msg_chatgpt_raw","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"hi"}]}],"tools":[]}}
data: [DONE]"#;

    /// The same turn with an empty terminal `output`: the normalized path
    /// takes the streamed-text fallback.
    const EMPTY_OUTPUT_SSE_BODY: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}
data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}
data: [DONE]"#;

    fn model(body: &'static str) -> ResponsesCompletionModel<RecordingHttpClient> {
        let client = crate::providers::chatgpt::Client::builder()
            .api_key(ChatGPTAuth::AccessToken {
                access_token: "test-token".to_string(),
                account_id: Some("account-id".to_string()),
            })
            .http_client(RecordingHttpClient::new(body))
            .build()
            .expect("client should build");
        client.completion_model(GPT_5_4)
    }

    /// The load-bearing capture property, on both normalization branches:
    /// `raw` is the reassembled Responses `CompletionResponse` — it
    /// deserializes back into that type and re-serializes to the identical
    /// value, and equals what `raw_completion` returns for the same body.
    /// On the empty-`output` body the choice comes from the streamed text,
    /// and the capture is still the terminal record (with its empty
    /// `output`), because that is what `raw_completion` would have
    /// returned.
    #[tokio::test]
    async fn completion_captures_raw_on_both_normalization_branches() {
        for (body, case) in [
            (SSE_BODY, "assembled output"),
            (EMPTY_OUTPUT_SSE_BODY, "empty-output fallback"),
        ] {
            let model = model(body);

            let response = model
                .completion(model.completion_request("hello").build())
                .await
                .expect("completion");
            let escape_hatch = model
                .raw_completion(model.completion_request("hello").build())
                .await
                .expect("raw completion");

            let raw = &response.raw;
            let typed: responses_api::CompletionResponse =
                serde_json::from_value(raw.clone()).expect("raw must deserialize");
            assert_eq!(
                serde_json::to_value(&typed).expect("re-serialize"),
                *raw,
                "{case}: the capture must be exactly what the wire type serializes to"
            );
            assert_eq!(
                serde_json::to_value(&escape_hatch).expect("serialize raw_completion"),
                *raw,
                "{case}: the capture must be what raw_completion returns"
            );
            assert_eq!(raw["service_tier"], "default", "{case}");
            assert_eq!(typed.id, "resp_chatgpt_raw", "{case}");

            assert_eq!(response.usage.total_tokens, 2, "{case}");
            assert_eq!(
                response.choice,
                vec![completion::AssistantContent::text("hi")],
                "{case}: both branches yield the streamed text"
            );
            assert_eq!(
                response.identity().response_id.as_deref(),
                Some("resp_chatgpt_raw"),
                "{case}"
            );
        }
    }
}
