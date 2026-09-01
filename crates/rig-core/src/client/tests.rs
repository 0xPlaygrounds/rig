use crate::providers::anthropic;

/// Type-level test that `Client::builder()` methods do not require annotation to determine
/// backig HTTP client
#[test]
fn ensures_client_builder_no_annotation() {
    let http_client = crate::test_utils::RecordingHttpClient::new("");
    let _ = anthropic::Client::builder()
        .http_client(http_client)
        .api_key("Foo")
        .build()
        .unwrap();
}
