//! Migrated from `examples/extractor.rs`.

use rig::client::ProviderClient;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    #[schemars(required)]
    first_name: Option<String>,
    #[schemars(required)]
    last_name: Option<String>,
    #[schemars(required)]
    job: Option<String>,
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn extractor_roundtrip_and_usage() {
    let client = openai::Client::from_env();
    let extractor = client.extractor::<Person>(openai::GPT_4).build();

    let person = extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await
        .expect("extract should succeed");
    assert_eq!(person.first_name.as_deref(), Some("John"));
    assert_eq!(person.last_name.as_deref(), Some("Doe"));
    assert_nonempty_response(person.job.as_deref().unwrap_or_default());

    let response = extractor
        .extract_with_usage("Jane Smith is a data scientist.")
        .await
        .expect("extract_with_usage should succeed");
    assert_eq!(response.data.first_name.as_deref(), Some("Jane"));
    assert_eq!(response.data.last_name.as_deref(), Some("Smith"));
    assert_nonempty_response(response.data.job.as_deref().unwrap_or_default());
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}
