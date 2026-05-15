//! Copilot model listing smoke test.
//!
//! This test is ignored by default and requires Copilot credentials. Run it
//! with one of the auth methods below:
//!
//! **API key:**
//! ```sh
//! GITHUB_COPILOT_API_KEY=<key> \
//!   cargo test -p rig --test copilot list_models_smoke -- --ignored --nocapture
//! ```
//!
//! **GitHub personal access token:**
//! ```sh
//! COPILOT_GITHUB_ACCESS_TOKEN=<token> \
//!   cargo test -p rig --test copilot list_models_smoke -- --ignored --nocapture
//! ```
//!
//! **OAuth (device code flow):** leave the env vars unset; the test will open a
//! browser prompt for authorization, or reuse a cached token from a previous run.
//! ```sh
//! cargo test -p rig --test copilot list_models_smoke -- --ignored --nocapture
//! ```

use crate::copilot::live_client;
use rig::client::ModelListingClient;

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn list_models_smoke() {
    let client = live_client();
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
}
