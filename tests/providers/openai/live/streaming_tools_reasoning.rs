use futures::StreamExt;
use rig::{
    client::AgentClientExt, completion::Message, providers::openai, streaming::StreamingChat,
};
use rig_agent::test_utils::MockExampleTool;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY environment variable"]
async fn test_openai_streaming_tools_reasoning() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var should exist");
    let client = openai::Client::new(&api_key).expect("Failed to build client");
    let agent = client
        .agent("gpt-5.2")
        .max_tokens(8192)
        .tool(MockExampleTool)
        .additional_params(serde_json::json!({
            "reasoning": {"effort": "high"}
        }))
        .build();

    let chat_history: Vec<Message> = Vec::new();
    let mut stream = agent
        .stream_chat("Call my example tool", &chat_history)
        .max_turns(5)
        .await;

    while let Some(item) = stream.next().await {
        println!("Got item: {item:?}");
    }
}
