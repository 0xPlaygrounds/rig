//! `updates()` over a replayed mid-stream fault of the `streamGenerateContent`
//! adapter: a real content frame, then the error envelope Gemini returned in
//! band for an overloaded model (both from captures, see `stream_faults`).
//! The hypothesis is that the updates end with one `Failed` carrying the
//! envelope's classification, and that its `partial` holds the text the
//! content frame finished before the fault.

use rig::completion::CompletionRequest;
use rig::error::ErrorKind;
use rig::message::AssistantContent;
use rig::providers::gemini::completion::GEMINI_2_5_FLASH;
use rig_test_support::updates::{assert_failed_update_contract, collect_updates};

use super::stream_faults::{
    CONTENT_FRAME, CONTENT_TEXT, IN_BAND_ERROR_FRAME, gemini_sse, scripted_client,
};

/// The envelope after content fails the updates as the provider's 503,
/// retryable, with the envelope preserved; the text the content frame
/// carried ended before the fault and is `partial`'s one part.
#[tokio::test]
async fn an_in_band_error_after_content_fails_with_the_envelope_and_the_text() {
    let (client, http) = scripted_client(vec![gemini_sse(&[CONTENT_FRAME, IN_BAND_ERROR_FRAME])]);
    let model = client.connect(http).completion(GEMINI_2_5_FLASH);
    let mut stream = model
        .stream(CompletionRequest::new("pong?"))
        .expect("the stream opens");
    let updates = collect_updates(&mut stream).await;

    let (error, partial, delivered) = assert_failed_update_contract(&updates);
    assert_eq!(error.kind, ErrorKind::ProviderResponse, "{error:?}");
    assert_eq!(error.http_status, Some(503), "{error:?}");
    assert!(error.retryable, "{error:?}");
    assert_eq!(
        error
            .provider_response_body()
            .map(|body| serde_json::from_str::<serde_json::Value>(body).expect("JSON")),
        Some(serde_json::from_str(IN_BAND_ERROR_FRAME).expect("JSON")),
        "the envelope is preserved: {error:?}"
    );
    assert_eq!(partial.choice, [AssistantContent::text(CONTENT_TEXT)]);
    assert_eq!(delivered[0].text, CONTENT_TEXT);
    assert_eq!(partial, stream.partial());
}
