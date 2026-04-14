//! Integration tests for llama.cpp extractor usage tracking.

use anyhow::Result;
use rig::extractor::ExtractionResponse;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::support;

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct Person {
    name: Option<String>,
    age: Option<u8>,
    profession: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct Address {
    street: Option<String>,
    city: Option<String>,
    state: Option<String>,
    zip_code: Option<String>,
}

fn assert_compatible_professions(left: Option<&str>, right: Option<&str>) {
    let left = left
        .expect("profession should be present")
        .trim()
        .to_ascii_lowercase();
    let right = right
        .expect("profession should be present")
        .trim()
        .to_ascii_lowercase();

    assert!(
        left == right || left.contains(&right) || right.contains(&left),
        "expected compatible professions, got {left:?} and {right:?}"
    );
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn extract_backward_compatibility() -> Result<()> {
    let model = support::model_name();
    let client = support::completions_client();
    let extractor = client.extractor::<Person>(model).build();

    let person = extractor
        .extract("John Doe is a 30 year old software engineer.")
        .await?;

    assert_eq!(person.name, Some("John Doe".to_string()));
    assert_eq!(person.age, Some(30));
    assert_eq!(person.profession, Some("software engineer".to_string()));

    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let model = support::model_name();
    let client = support::completions_client();
    let extractor = client.extractor::<Person>(model).build();

    let response: ExtractionResponse<Person> = extractor
        .extract_with_usage("Jane Smith is a 45 year old data scientist.")
        .await?;

    assert_eq!(response.data.name, Some("Jane Smith".to_string()));
    assert_eq!(response.data.age, Some(45));
    assert_eq!(response.data.profession, Some("data scientist".to_string()));
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
    assert!(response.usage.total_tokens > 0);

    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    use rig::message::Message;

    let model = support::model_name();
    let client = support::completions_client();
    let extractor = client.extractor::<Address>(model).build();

    let chat_history = vec![Message::user(
        "I'm looking at a property that might be interesting.",
    )];

    let response: ExtractionResponse<Address> = extractor
        .extract_with_chat_history_with_usage(
            "The address is 123 Main St in Springfield, IL 62701.",
            chat_history,
        )
        .await?;

    assert_eq!(response.data.street, Some("123 Main St".to_string()));
    assert_eq!(response.data.city, Some("Springfield".to_string()));
    assert_eq!(response.data.state, Some("IL".to_string()));
    assert_eq!(response.data.zip_code, Some("62701".to_string()));
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.total_tokens > 0);

    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let model = support::model_name();
    let client = support::completions_client();
    let extractor = client.extractor::<Person>(model).build();

    let text = "Bob Johnson is a 55 year old retired teacher.";
    let person = extractor.extract(text).await?;
    let response = extractor.extract_with_usage(text).await?;

    assert_eq!(person.name, Some("Bob Johnson".to_string()));
    assert_eq!(response.data.name, Some("Bob Johnson".to_string()));
    assert_eq!(person.age, Some(55));
    assert_eq!(response.data.age, Some(55));
    assert_compatible_professions(
        person.profession.as_deref(),
        response.data.profession.as_deref(),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");

    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let model = support::model_name();
    let client = support::completions_client();

    let person_extractor = client.extractor::<Person>(model.clone()).build();
    let person_response = person_extractor
        .extract_with_usage("Alice is a 25 year old developer.")
        .await?;

    assert!(person_response.usage.total_tokens > 0);

    let address_extractor = client.extractor::<Address>(model).build();
    let address_response = address_extractor
        .extract_with_usage("456 Oak Avenue, Cambridge, MA 02139")
        .await?;

    assert!(address_response.usage.total_tokens > 0);

    Ok(())
}
