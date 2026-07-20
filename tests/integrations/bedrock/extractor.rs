//! AWS Bedrock extractor smoke tests inspired by the provider extractor tests.

use rig::client::CompletionClient;
use rig::message::Message;

use super::{
    BEDROCK_COMPLETION_MODEL, client,
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
    let extractor = client()
        .extractor::<SmokePerson>(BEDROCK_COMPLETION_MODEL)
        .build();

    let response = extractor
        .extract_with_usage(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    assert_smoke_person(&response.data);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn extractor_with_chat_history_smoke() {
    let extractor = client()
        .extractor::<SmokePerson>(BEDROCK_COMPLETION_MODEL)
        .build();

    let response = extractor
        .extract_with_chat_history_with_usage(
            "The text is about Ada Lovelace, a mathematician.",
            vec![Message::user(
                "Extract the person's name and job from the next message.",
            )],
        )
        .await
        .expect("extractor request with chat history should succeed");

    assert_smoke_person(&response.data);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}
