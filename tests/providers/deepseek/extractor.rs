//! DeepSeek extractor smoke test.

use rig::client::CompletionClient;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
async fn extractor_smoke() {
    with_deepseek_cassette("extractor/extractor_smoke", |client| async move {
        let extractor = client
            .extractor::<SmokePerson>(deepseek::DEEPSEEK_V4_FLASH)
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
    })
    .await;
}
