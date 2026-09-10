use super::*;

#[test]
fn api_response_detects_nested_error_before_permissive_success() {
    #[derive(Debug, Deserialize)]
    struct PermissiveResponse {
        #[serde(default)]
        candidates: Vec<serde_json::Value>,
    }

    let response: ApiResponse<PermissiveResponse> =
        serde_json::from_str(r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#)
            .expect("nested Gemini error should deserialize");

    match response {
        ApiResponse::Err(err) => assert_eq!(err.error.message, "boom"),
        ApiResponse::Ok(response) => panic!(
            "expected nested error, got success with {} candidates",
            response.candidates.len()
        ),
    }
}

#[test]
fn api_response_allows_top_level_message_in_success() {
    #[derive(Debug, Deserialize)]
    struct MessageResponse {
        message: String,
    }

    let response: ApiResponse<MessageResponse> = serde_json::from_str(r#"{"message":"success"}"#)
        .expect("success response should deserialize");

    match response {
        ApiResponse::Ok(response) => assert_eq!(response.message, "success"),
        ApiResponse::Err(err) => panic!("expected success, got error: {err:?}"),
    }
}

#[test]
fn test_client_initialization() {
    let _client: Client<_> =
        Client::new_with("dummy-key", crate::test_utils::RecordingHttpClient::new(""))
            .expect("Client::new() failed");
    let _client_from_builder: Client<_> = Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}
