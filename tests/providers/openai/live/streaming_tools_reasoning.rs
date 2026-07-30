use futures::StreamExt;
use rig::AgentBuilder;
use rig::provider::ProviderConfig;
use rig::{completion::Message, providers::openai};
use rig_agent::test_utils::MockExampleTool;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY environment variable"]
async fn test_openai_streaming_tools_reasoning() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var should exist");
    let cfg = openai::responses_api::functions::Config::new("gpt-5.2").with_api_key(api_key);
    let agent = AgentBuilder::new(ProviderConfig::OpenAiResponses(cfg))
        .max_tokens(8192)
        .tool(MockExampleTool)
        .additional_params(serde_json::json!({
            "reasoning": {"effort": "high"}
        }))
        .build();

    let chat_history: Vec<Message> = Vec::new();
    let mut stream = Box::pin(
        agent
            .runner("Call my example tool")
            .history(&chat_history)
            .max_turns(5)
            .stream_run(),
    );

    while let Some(item) = stream.next().await {
        println!("Got item: {item:?}");
    }
}
