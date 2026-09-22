//! Groq extractor smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::EXTRACTOR_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn extractor_smoke() {
    let groq = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let extractor = groq.extractor::<SmokePerson>(EXTRACTOR_MODEL).build();

    let response = extractor
        .extract(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    let first_name = response
        .output
        .first_name
        .as_deref()
        .expect("first_name should be present");
    let last_name = response
        .output
        .last_name
        .as_deref()
        .expect("last_name should be present");
    let job = response
        .output
        .job
        .as_deref()
        .expect("job should be present");

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    assert!(
        response.usage.total_tokens.is_some_and(|n| n > 0),
        "usage should be populated"
    );
}
