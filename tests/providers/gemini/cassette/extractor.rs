//! Gemini extractor coverage, including the migrated example path.

use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig_agent::test_utils::validate_extraction_fields;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    first_name: Option<String>,
    last_name: Option<String>,
    job: Option<String>,
}

fn serialize_params(params: AdditionalParameters) -> serde_json::Value {
    serde_json::to_value(params).expect("Gemini additional params should serialize")
}

#[tokio::test]
async fn extractor_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    super::super::support::with_gemini_cassette("extractor/extractor_smoke", |client| async move {
        let response = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .additional_params(serialize_params(additional_params))
            .build()
            .extractor(EXTRACTOR_TEXT)
            .classic()
            .run_with_usage::<SmokePerson>()
            .await
            .expect("extractor request should succeed");

        validate_extraction_fields(
            "gemini_extractor_smoke",
            response.value.first_name.as_deref(),
            response.value.last_name.as_deref(),
            response.value.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");

        let first_name = response
            .value
            .first_name
            .as_deref()
            .expect("first_name should be present");
        let last_name = response
            .value
            .last_name
            .as_deref()
            .expect("last_name should be present");
        let job = response
            .value
            .job
            .as_deref()
            .expect("job should be present");

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
            let person = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .additional_params(serialize_params(params))
                .build()
                .extractor("Hello my name is John Doe! I am a software engineer.")
                .classic()
                .run::<Person>()
                .await
                .expect("extract should succeed");

            assert_eq!(person.first_name.as_deref(), Some("John"));
            assert_eq!(person.last_name.as_deref(), Some("Doe"));
            assert_nonempty_response(person.job.as_deref().unwrap_or_default());
        },
    )
    .await;
}
