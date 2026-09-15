//! Declaring which providers carry `output_schema` must not change the ones
//! that do.
//!
//! `ProviderCapabilities` gained `supports_native_output_schema`, and the
//! runtime in `rig-ecs` now routes a schema-bearing run to its output tool
//! when the provider says it drops the field. OpenAI does carry it, so this
//! recording pins the unchanged side: the request still puts the schema on the
//! wire under `text.format`, and the answer still validates against it.
//!
//! The other side — a provider that drops the schema falling back to the
//! output tool — has no recording here: all seven droppers in the tree
//! (perplexity, together, moonshot, mira, deepseek, huggingface, hyperbolic)
//! need API keys that were not available at recording time. That path is
//! covered by
//! `rig-ecs/tests/run_output_tool_config.rs::a_provider_that_drops_the_schema_gets_the_output_tool_not_unvalidated_text`,
//! which fails before the fix with the model's prose (`"forty two, roughly"`)
//! reported as the structured answer.

use rig::prelude::*;
use rig::providers::openai;
use super::super::support::with_openai_cassette;

#[tokio::test]
async fn a_provider_that_carries_the_schema_still_asks_natively() {
    with_openai_cassette(
        "structured_output/native_schema_still_reaches_the_wire",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble("Answer with the number only.")
                .build();

            let response = agent
                .prompt("What is six times seven?")
                .await
                .expect("a schema-carrying provider answers natively");

            assert!(
                response.to_string().contains("42"),
                "the model answered the question: {response}"
            );
        },
    )
    .await;
}
