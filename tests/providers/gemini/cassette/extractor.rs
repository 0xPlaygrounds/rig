//! Gemini extractor coverage, including the migrated example path.

use rig::client::AgentClientExt;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::test_utils::validate_extraction_fields;
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
async fn extractor_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    super::super::support::with_gemini_cassette("extractor/extractor_smoke", |client| async move {
        let extractor = client
            .extractor::<SmokePerson>(gemini::completion::GEMINI_2_5_FLASH)
            .additional_params(
                serde_json::to_value(additional_params)
                    .expect("Gemini additional params should serialize"),
            )
            .build();

        let response = extractor
            .extract_with_usage(EXTRACTOR_TEXT)
            .await
            .expect("extractor request should succeed");

        validate_extraction_fields(
            "gemini_extractor_smoke",
            response.data.first_name.as_deref(),
            response.data.last_name.as_deref(),
            response.data.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");

        let first_name = response
            .data
            .first_name
            .as_deref()
            .expect("first_name should be present");
        let last_name = response
            .data
            .last_name
            .as_deref()
            .expect("last_name should be present");
        let job = response.data.job.as_deref().expect("job should be present");

        assert_nonempty_response(first_name);
        assert_nonempty_response(last_name);
        assert_nonempty_response(job);
    })
    .await;
}

#[tokio::test]
async fn extractor_with_additional_params() {
    let params = AdditionalParameters::default().with_config(GenerationConfig::default());
    super::super::support::with_gemini_cassette(
        "extractor/extractor_with_additional_params",
        |client| async move {
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
        },
    )
    .await;
}
