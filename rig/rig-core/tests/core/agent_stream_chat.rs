//! Migrated from `examples/agent_stream_chat.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::message::Message;
use rig::providers::openai;
use rig::streaming::StreamingChat;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn stream_chat_with_history() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    let history = vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ];

    let mut stream = agent.stream_chat("Entertain me!", &history).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("stream chat should succeed");

    assert_nonempty_response(&response);
}
