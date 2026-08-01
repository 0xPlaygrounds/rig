//! AWS Bedrock extractor smoke tests inspired by the provider extractor tests.

use rig::message::Message;

use super::{
    BEDROCK_COMPLETION_MODEL, agent,
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
    let agent = agent(BEDROCK_COMPLETION_MODEL).build();
    let response = agent
        .extractor(EXTRACTOR_TEXT)
        .classic()
        .run_with_usage::<SmokePerson>()
        .await
        .expect("extractor request should succeed");

    assert_smoke_person(&response.value);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn extractor_with_chat_history_smoke() {
    let agent = agent(BEDROCK_COMPLETION_MODEL).build();
    let response = agent
        .extractor("The text is about Ada Lovelace, a mathematician.")
        .classic()
        .history([Message::user(
            "Extract the person's name and job from the next message.",
        )])
        .run_with_usage::<SmokePerson>()
        .await
        .expect("extractor request with chat history should succeed");

    assert_smoke_person(&response.value);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}
