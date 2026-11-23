//! Copilot model listing smoke test.
//!
//! This test is ignored by default and requires Copilot credentials: set
//! `GITHUB_COPILOT_API_KEY` or `COPILOT_GITHUB_ACCESS_TOKEN`, or leave both
//! unset to authorize through the OAuth device code flow, which reuses a
//! cached token when one exists.
//!
//! ```sh
//! cargo test -p rig-cassette --test integration copilot::models::list_models_smoke -- --ignored --nocapture
//! ```

use crate::copilot::with_copilot_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_copilot_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(error) => {
                panic!("listing Copilot models should succeed\nDisplay: {error}\nDebug: {error:#?}")
            }
        };

        assert!(
            !models.is_empty(),
            "expected Copilot to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
