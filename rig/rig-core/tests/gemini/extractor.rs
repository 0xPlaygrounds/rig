//! Gemini extractor coverage, including the migrated example path.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    first_name: Option<String>,
    last_name: Option<String>,
    job: Option<String>,
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn extractor_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    let client = gemini::Client::from_env();
    let extractor = client
        .extractor::<SmokePerson>(gemini::completion::GEMINI_2_5_FLASH)
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

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn extractor_with_additional_params() {
    let params = AdditionalParameters::default().with_config(GenerationConfig::default());
    let client = gemini::Client::from_env();
    let extractor = client
        .extractor::<Person>(gemini::completion::GEMINI_2_5_FLASH)
        .additional_params(serde_json::to_value(params).expect("params should serialize"))
        .build();

    let person = extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await
        .expect("extract should succeed");

    assert_eq!(person.first_name.as_deref(), Some("John"));
    assert_eq!(person.last_name.as_deref(), Some("Doe"));
    assert_nonempty_response(person.job.as_deref().unwrap_or_default());
}
