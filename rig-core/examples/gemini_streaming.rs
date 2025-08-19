use rig::agent::stream_to_stdout;
use rig::prelude::*;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig,
};
use rig::{
    providers::gemini::{self},
    streaming::StreamingPrompt,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    let gen_cfg = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            include_thoughts: Some(true),
            thinking_budget: 2048,
        }),
        ..Default::default()
    };
    let cfg = AdditionalParameters::default().with_config(gen_cfg);
    // Create streaming agent with a single context prompt
    let agent = gemini::Client::from_env()
        .agent("gemini-2.5-flash")
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .additional_params(serde_json::to_value(cfg).unwrap())
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let res = stream_to_stdout(&mut stream).await?;

    println!("Token usage response: {usage:?}", usage = res.usage());
    println!("Final text response: {message:?}", message = res.response());

    Ok(())
}
