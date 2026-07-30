//! AWS Bedrock extractor smoke tests inspired by the provider extractor tests.

use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::message::Message;
use rig::provider::Runtime;

use super::{
    BEDROCK_COMPLETION_MODEL, bedrock_config,
    support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response},
};

fn assert_smoke_person(person: &SmokePerson) {
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
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn extractor_smoke() {
    let response = extract_with_options::<SmokePerson>(
        AgentConfig::new(),
        bedrock_config(BEDROCK_COMPLETION_MODEL),
        Arc::new(Runtime::new()),
        EXTRACTOR_TEXT,
        ExtractOptions::classic_extractor(),
    )
    .await
    .expect("extractor request should succeed");

    assert_smoke_person(&response.value);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn extractor_with_chat_history_smoke() {
    let response = extract_with_options::<SmokePerson>(
        AgentConfig::new(),
        bedrock_config(BEDROCK_COMPLETION_MODEL),
        Arc::new(Runtime::new()),
        "The text is about Ada Lovelace, a mathematician.",
        ExtractOptions::classic_extractor().with_history(vec![Message::user(
            "Extract the person's name and job from the next message.",
        )]),
    )
    .await
    .expect("extractor request with chat history should succeed");

    assert_smoke_person(&response.value);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}
