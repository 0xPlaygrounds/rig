//! DeepSeek extractor smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::deepseek;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extractor_smoke() {
    let client = deepseek::Client::from_env();
    let extractor = client
        .extractor::<SmokePerson>(deepseek::DEEPSEEK_CHAT)
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
