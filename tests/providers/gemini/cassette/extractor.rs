//! Gemini extractor coverage, including the migrated example path.

use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::provider::Runtime;
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

/// The classic `ExtractorBuilder::additional_params` knob is an
/// [`AgentConfig`] field now.
fn config_with_additional_params(params: AdditionalParameters) -> AgentConfig {
    let mut config = AgentConfig::new();
    config.additional_params =
        Some(serde_json::to_value(params).expect("Gemini additional params should serialize"));
    config
}

#[tokio::test]
async fn extractor_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    super::super::support::with_gemini_cassette("extractor/extractor_smoke", |client| async move {
        let response = extract_with_options::<SmokePerson>(
            config_with_additional_params(additional_params),
            client.provider_config(gemini::completion::GEMINI_2_5_FLASH),
            Arc::new(Runtime::new()),
            EXTRACTOR_TEXT,
            ExtractOptions::classic_extractor(),
        )
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
            let person = extract_with_options::<Person>(
                config_with_additional_params(params),
                client.provider_config(gemini::completion::GEMINI_2_5_FLASH),
                Arc::new(Runtime::new()),
                "Hello my name is John Doe! I am a software engineer.",
                ExtractOptions::classic_extractor(),
            )
            .await
            .expect("extract should succeed")
            .value;

            assert_eq!(person.first_name.as_deref(), Some("John"));
            assert_eq!(person.last_name.as_deref(), Some("Doe"));
            assert_nonempty_response(person.job.as_deref().unwrap_or_default());
        },
    )
    .await;
}
