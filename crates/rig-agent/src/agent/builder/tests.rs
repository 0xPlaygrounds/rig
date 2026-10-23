use super::*;
use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool};

/// Tools added through the builder join a shared registry at build, after
/// the ones it already holds.
#[test]
fn builder_tools_join_a_shared_registry() {
    let shared = ToolServer::new().tool(MockSubtractTool).run();
    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockAddTool)
        .tool_server_handle(shared.clone())
        .build();
    assert_eq!(
        shared.snapshot().names().collect::<Vec<_>>(),
        ["subtract", "add"]
    );
    assert_eq!(agent.tool_server_handle().snapshot().names().count(), 2);
}

/// Hosted tools reach every request of a tool round trip, unary and
/// streaming, and encode into one `tools` array after the function tools.
#[tokio::test]
async fn provider_tools_join_function_tools_on_every_request() {
    use futures::StreamExt;
    use rig_core::completion::ProviderToolDefinition;
    use rig_core::providers::openai::OpenAIConfig;
    use rig_core::test_utils::{MockStreamEvent, MockTurn};
    use rig_core::wire::{Body, Mode, Wire};

    let unary = MockCompletionModel::from_turns([
        MockTurn::tool_call("c1", "add", serde_json::json!({"x": 1, "y": 2})),
        MockTurn::text("3"),
    ]);
    let streaming = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "add", serde_json::json!({"x": 1, "y": 2})),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let agent = |model: MockCompletionModel| {
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .additional_params(serde_json::json!({"top_p": 0.5}))
            .provider_tool(ProviderToolDefinition::new("web_search"))
            .build()
    };
    agent(unary.clone())
        .prompt("add")
        .max_turns(2)
        .await
        .expect("the unary run answers");
    let mut stream = agent(streaming.clone()).prompt("add").max_turns(2).stream();
    while let Some(item) = stream.next().await {
        item.expect("the streaming run answers");
    }

    let wire = OpenAIConfig::new("key").chat("gpt-4o-mini");
    let requests = [
        unary
            .requests()
            .into_iter()
            .map(|r| (r, Mode::Unary))
            .collect::<Vec<_>>(),
        streaming
            .requests()
            .into_iter()
            .map(|r| (r, Mode::Streaming))
            .collect(),
    ]
    .concat();
    assert_eq!(requests.len(), 4);
    for (request, mode) in requests {
        let encoded = wire.encode(request, mode).expect("encodes").request;
        let Body::Bytes(bytes) = encoded.body() else {
            panic!("a JSON body");
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        let tools = body.get("tools").and_then(|tools| tools.as_array());
        let kinds: Vec<_> = tools
            .into_iter()
            .flatten()
            .map(|tool| tool.get("type").and_then(|kind| kind.as_str()))
            .collect();
        assert_eq!(kinds, [Some("function"), Some("web_search")]);
        assert_eq!(body.get("top_p"), Some(&serde_json::json!(0.5)));
    }
}
