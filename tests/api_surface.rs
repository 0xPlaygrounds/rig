//! The API a user sees, as the quickstart spells it: a model from a provider
//! client or from a `"provider:model"` string, called with plain values,
//! continued with its own reply, erased, and streamed as parts. Runs with no
//! network: the OpenAI client reads its base URL from the environment and
//! reaches a local server that answers with a recorded reply, and the
//! Anthropic client sends through a scripted HTTP client.

#![cfg(feature = "bedrock")]

use futures::StreamExt;
use rig::DynModel;
use rig::bedrock::client::BedrockRuntime;
use rig::bedrock::completion::AMAZON_NOVA_LITE;
use rig::completion::{CompletionRequest, Message};
use rig::operation::Completion;
use rig::providers::anthropic::{self, Anthropic};
use rig::providers::deepseek;
use rig::providers::openai::{self, OpenAI};
use rig::providers::registry::ProviderRef;
use rig::streaming::Update;
use rig_core::test_utils::MockStreamingClient;

/// OpenAI's reply to "Reply with exactly: identity probe", from
/// `crates/rig-cassette/fixtures/cassettes/openai/response_identity/responses_nonstreaming_carries_identity.yaml`.
const RESPONSES_REPLY: &str = r#"{"access_programs":null,"background":false,"billing":{"payer":"developer"},"completed_at":0,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"resp_0bc89794b805f805006ab396d844a087d09ca8275b6bcf0672","incomplete_details":null,"instructions":null,"max_output_tokens":null,"max_tool_calls":null,"metadata":{},"model":"gpt-4o-2024-08-06","moderation":null,"object":"response","output":[{"content":[{"annotations":[],"logprobs":[],"text":"identity probe","type":"output_text"}],"id":"msg_0bc89794b805f805006ab396d8d03c87d0b27327fdbc5dc326","role":"assistant","status":"completed","type":"message"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"prompt_cache_retention":"in_memory","reasoning":{"context":null,"effort":null,"summary":null},"safety_identifier":null,"service_tier":"default","status":"completed","store":true,"temperature":1.0,"text":{"format":{"type":"text"},"verbosity":"medium"},"tool_choice":"auto","tool_usage":{"image_gen":{"input_tokens":0,"input_tokens_details":{"image_tokens":0,"text_tokens":0},"output_tokens":0,"output_tokens_details":{"image_tokens":0,"text_tokens":0},"total_tokens":0},"web_search":{"num_requests":0}},"tools":[],"top_logprobs":0,"top_p":1.0,"truncation":"disabled","usage":{"input_tokens":13,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":16},"user":null}"#;

/// An Anthropic stream that answers "Paris." and ends its turn.
const ANTHROPIC_STREAM: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Par\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"is.\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

/// Every model built from the environment, in one test: the variables are
/// process-wide.
#[tokio::test]
async fn models_from_clients_and_strings_hold_a_conversation() -> anyhow::Result<()> {
    let server = httpmock::MockServer::start_async().await;
    let responses = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/responses");
            then.status(200)
                .header("content-type", "application/json")
                .body(RESPONSES_REPLY);
        })
        .await;
    // SAFETY: this is the only test in this binary that reads or writes the
    // environment.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test-key");
        std::env::set_var("OPENAI_BASE_URL", server.url("/v1"));
        std::env::set_var("DEEPSEEK_API_KEY", "test-key");
    }

    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let res = model
        .call(CompletionRequest::new("Capital of France?").temperature(0.2))
        .await?;
    println!("{}", res.text());
    anyhow::ensure!(res.text() == "identity probe");

    let mut history = vec![Message::user("Capital of France?")];
    history.push(model.call(history.clone()).await?.into());
    anyhow::ensure!(history.len() == 2);
    anyhow::ensure!(matches!(&history[1], Message::Assistant { .. }));
    responses.assert_calls_async(2).await;

    let deepseek = deepseek::from_env()?.chat(deepseek::DEEPSEEK_V4_FLASH);
    anyhow::ensure!(deepseek.id() == Some(deepseek::DEEPSEEK_V4_FLASH));
    let any = ProviderRef::parse("deepseek:deepseek-chat")?.completion_model()?;
    anyhow::ensure!(any.name() == "deepseek" && any.id() == Some("deepseek-chat"));

    let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
    let width = model.capabilities().ndims;
    anyhow::ensure!(width == 1536);

    let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
    anyhow::ensure!(model.id() == Some(AMAZON_NOVA_LITE));
    Ok(())
}

/// A client on another HTTP client, erased, and streamed as parts.
#[tokio::test]
async fn an_erased_model_streams_its_parts() -> anyhow::Result<()> {
    let key = "test-key";
    let http = MockStreamingClient {
        sse_bytes: bytes::Bytes::from_static(ANTHROPIC_STREAM.as_bytes()),
    };

    let model = Anthropic::new(key)
        .with_http(http)
        .completion(anthropic::CLAUDE_SONNET_4_6);
    let erased: DynModel<Completion> = model.into();

    let mut stream = erased.stream("Capital of France?")?;
    let mut updates = stream.updates();
    let mut text = String::new();
    let mut done = None;
    while let Some(update) = updates.next().await {
        match update? {
            Update::Delta { text: delta, .. } => text.push_str(&delta),
            Update::Done(response) => done = Some(response),
            _ => {}
        }
    }
    anyhow::ensure!(text == "Paris.");
    anyhow::ensure!(done.is_some_and(|response| response.text() == "Paris."));
    Ok(())
}
