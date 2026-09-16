//! Migrated from `examples/huggingface_subproviders.rs`.

use rig::prelude::*;
use rig::providers::openai::wire::{HUGGINGFACE, OpenAI, SubRoute};

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tool_prompt_across_subproviders() {
    let cases = [
        ("deepseek-ai/DeepSeek-V3", SubRoute::Together),
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            SubRoute::HFInference,
        ),
        ("Meta-Llama-3.1-8B-Instruct", SubRoute::SambaNova),
    ];

    for (model, sub_route) in cases {
        let provider = OpenAI::from_env_with(&HUGGINGFACE)
            .expect("config should build from env")
            .with_sub_route(sub_route)
            .bound()
            .expect("transport should build");
        let agent = provider
            .agent(model)
            .preamble(
                "You are a calculator here to help the user perform arithmetic operations. \
                 Use the provided tools to answer the user's question.",
            )
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let response = agent
            .prompt("Calculate 2 - 5")
            .await
            .expect("prompt should succeed");
        assert_mentions_expected_number(&response.output, -3);
    }
}
