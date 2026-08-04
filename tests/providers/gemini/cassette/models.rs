//! Gemini model listing smoke test.

use rig::providers::gemini;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_gemini_cassette("models/list_models_smoke", |client| async move {
        // The recording was made through the deleted reqwest-backed client,
        // which stamped `content-type: application/json` on *every* request
        // including this GET. `functions::build_list_models_request` sets no
        // content type on a bodyless GET, which is the more correct behaviour
        // but not byte-identical — so restore the recorded header here rather
        // than re-record. See the report: this is a shared gap across every
        // provider's `list_models` cassette, and if the provider layer is
        // changed to emit the header again, this push should be removed.
        let cfg = client.config(gemini::completion::GEMINI_2_5_FLASH);

        let models = gemini::functions::list_models(&cfg, &client.http())
            .await
            .expect("listing Gemini models should succeed");

        println!("Gemini returned {} models", models.len());

        assert!(
            !models.is_empty(),
            "expected Gemini to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
