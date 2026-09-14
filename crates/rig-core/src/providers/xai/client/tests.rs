#[test]
fn test_client_initialization() {
    let _client_from_builder = crate::providers::xai::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}
