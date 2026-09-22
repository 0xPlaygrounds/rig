//! Groq context smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

use super::CONTEXT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn context_smoke() {
    let groq = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(groq.agent(CONTEXT_MODEL), |builder, doc| {
            builder.context(doc)
        })
        .build();

    let response = agent
        .prompt(CONTEXT_PROMPT)
        .await
        .expect("context prompt should succeed");

    assert_contains_any_case_insensitive(
        &response.output,
        &[
            "ancient tool",
            "farming tool",
            "farm the land",
            "used by the ancestors",
        ],
    );
}
