//! xAI context smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::xai;

use crate::support::assert_contains_any_case_insensitive;

const XAI_CONTEXT_DOCS: [&str; 3] = [
    "Definition of flurbo: A flurbo is a green alien that lives on cold planets.",
    "Definition of glarb-glarb: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of linglingdong: A term used by inhabitants of the far side of the moon to describe humans.",
];

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn context_smoke() {
    let client = xai::Client::from_env().expect("client should build");
    let agent = XAI_CONTEXT_DOCS
        .iter()
        .copied()
        .fold(client.agent(xai::completion::GROK_4), |builder, doc| {
            builder.context(doc)
        })
        .preamble(
            "Use only the provided context snippets. \
             One snippet explicitly defines glarb-glarb. \
             If that definition says it is an ancient tool, reply with exactly: ancient tool. \
             Otherwise reply with exactly: not found.",
        )
        .build();

    let response = agent
        .prompt(
            "What is glarb-glarb according to the provided context? \
             Answer with exactly `ancient tool` or `not found`.",
        )
        .await
        .expect("context prompt should succeed");

    assert_contains_any_case_insensitive(&response, &["ancient tool"]);
}
