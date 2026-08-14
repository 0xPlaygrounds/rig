//! Gemini model listing smoke test.

use rig::client::ModelListingClient;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_gemini_cassette("models/list_models_smoke", |client| async move {
        let models = client
            .list_models()
            .await
            .expect("listing Gemini models should succeed");

        println!("Gemini returned {} models", models.len());

        assert!(
            !models.is_empty(),
            "expected Gemini to return at least one model\nModel list: {models:#?}"
        );

        // rig#2322 — Gemini reports an output ceiling for every model and rig
        // used to drop it during conversion. Nothing in the library knew the
        // real limit, which is how a hardcoded 4096 default survived: the
        // recorded listing says 65536 for this model, ~16x larger.
        let flash = models
            .iter()
            .find(|model| model.id == "gemini-2.5-flash")
            .expect("the recorded listing should contain gemini-2.5-flash");

        assert_eq!(
            flash.max_output_tokens,
            Some(65_536),
            "the model's output ceiling must survive listing conversion"
        );
        assert_eq!(
            flash.context_length,
            Some(1_048_576),
            "the input window must stay distinct from the output ceiling"
        );

        // Not merely "some model reported one": every entry the wire annotates
        // must round-trip, so a conversion that dropped the field for a subset
        // still fails.
        assert!(
            models
                .iter()
                .filter(|model| model.max_output_tokens.is_some())
                .count()
                > 40,
            "the recorded listing annotates ~50 models with outputTokenLimit; \
             far fewer surviving means the conversion is dropping them"
        );
    })
    .await;
}
