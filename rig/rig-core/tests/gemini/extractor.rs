//! Gemini extractor smoke test covering the additional-params request path.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn extractor_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    let client = gemini::Client::from_env();
    let extractor = client
        .extractor::<SmokePerson>(gemini::completion::GEMINI_2_0_FLASH)
        .additional_params(
            serde_json::to_value(additional_params)
                .expect("Gemini additional params should serialize"),
        )
        .build();

    let person = extractor
        .extract(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    let first_name = person
        .first_name
        .as_deref()
        .expect("first_name should be present");
    let last_name = person
        .last_name
        .as_deref()
        .expect("last_name should be present");
    let job = person.job.as_deref().expect("job should be present");

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
}
