//! The API a user sees, as the quickstart spells it: a model from a provider
//! client or from a `"provider:model"` string, called with plain values,
//! continued with its own reply, erased, and streamed as parts, with every
//! failure one `RigError`. Runs with no network: the OpenAI client reads its
//! base URL from the environment and reaches a local server that answers with
//! a recorded reply, and the Anthropic client sends through a scripted HTTP
//! client. The Bedrock line needs the `bedrock` feature.

use futures::StreamExt;
use rig::DynModel;
#[cfg(feature = "bedrock")]
use rig::bedrock::{client::BedrockRuntime, completion::AMAZON_NOVA_LITE};
use rig::completion::{CompletionRequest, Message, PromptError};
use rig::error::{ErrorDetail, ErrorKind};
use rig::loaders::FileLoader;
use rig::operation::Completion;
use rig::prelude::*;
use rig::providers::anthropic::{self, Anthropic};
use rig::providers::deepseek;
use rig::providers::openai::{self, OpenAI};
use rig::providers::registry::ProviderRef;
use rig::streaming::{CompletionStream, StreamEvents, Update};
use rig::{Agent, RigError};
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

/// The same answer, cut after its text: no `message_delta`, no
/// `message_stop`.
const ANTHROPIC_CUT_STREAM: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Par\"}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"is.\"}}\n\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
);

/// Fails the test with `what` unless `condition` holds.
fn ensure(condition: bool, what: &str) -> Result<(), RigError> {
    if condition {
        Ok(())
    } else {
        Err(RigError::new(ErrorKind::Other, what))
    }
}

/// Stands in for signing in again.
fn relogin() {}

/// Stands in for waiting before a retry.
fn back_off() {}

/// A failed call, routed by what failed.
async fn say_hi(model: &DynModel<Completion>) -> Result<(), RigError> {
    match model.call("Hi").await {
        Err(error) if matches!(error.detail, Some(ErrorDetail::InvalidAuthentication)) => relogin(),
        Err(error) if error.http_status == Some(429) => back_off(),
        Err(error) => return Err(error),
        Ok(response) => println!("{}", response.text()),
    }
    Ok(())
}

/// A failed prompt, direct or relayed, in one shape.
async fn greet(agent: &Agent) -> Result<(), RigError> {
    match agent.prompt("Hi").await {
        Err(PromptError::Failed(error)) => return Err(error), // direct or relayed, one shape
        Err(other) => return Err(other.into()),
        Ok(response) => println!("{}", response.output),
    }
    Ok(())
}

/// An application function that mixes setup, model and foreign errors in
/// one `Result`.
async fn run(agent: &Agent) -> Result<(), RigError> {
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let any = ProviderRef::parse("deepseek:deepseek-chat")?;
    let docs = FileLoader::with_glob("docs/*.md")?;
    let answer = agent.prompt("Hi").await?;
    let key = std::env::var("APP_TOKEN").map_err(RigError::other)?;
    ensure(model.id() == Some(openai::GPT_5_2), "the model")?;
    ensure(
        any.completion_model()?.id() == Some("deepseek-chat"),
        "the reference",
    )?;
    ensure(
        docs.read().into_iter().count() == 0,
        "no docs/*.md under the test's directory",
    )?;
    ensure(answer.output == "identity probe", "the answer")?;
    ensure(key == "app-token", "the foreign value")
}

/// Every model built from the environment, in one test: the variables are
/// process-wide.
#[tokio::test]
async fn models_from_clients_and_strings_hold_a_conversation() -> Result<(), RigError> {
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
        std::env::set_var("APP_TOKEN", "app-token");
    }

    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let res = model
        .call(CompletionRequest::new("Capital of France?").temperature(0.2))
        .await?;
    println!("{}", res.text());
    ensure(res.text() == "identity probe", "the recorded answer")?;

    let mut history = vec![Message::user("Capital of France?")];
    history.push(model.call(history.clone()).await?.into());
    ensure(history.len() == 2, "the reply continues the history")?;
    ensure(
        matches!(history.last(), Some(Message::Assistant { .. })),
        "the reply is the assistant's message",
    )?;
    responses.assert_calls_async(2).await;

    say_hi(&model.clone().into()).await?;
    ensure(
        call_with_one_retry(&model.clone().into()).await? == "identity probe",
        "the answer, with no retry needed",
    )?;
    let agent = AgentBuilder::new(model).build();
    greet(&agent).await?;
    run(&agent).await?;
    responses.assert_calls_async(6).await;

    let deepseek = deepseek::from_env()?.chat(deepseek::DEEPSEEK_V4_FLASH);
    ensure(
        deepseek.id() == Some(deepseek::DEEPSEEK_V4_FLASH),
        "the DeepSeek model id",
    )?;
    let any = ProviderRef::parse("deepseek:deepseek-chat")?.completion_model()?;
    ensure(
        any.name() == "deepseek" && any.id() == Some("deepseek-chat"),
        "the model the reference names",
    )?;

    let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
    let width = model.capabilities().ndims;
    ensure(width == 1536, "the embedding width")?;

    #[cfg(feature = "bedrock")]
    {
        let model = BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE);
        ensure(model.id() == Some(AMAZON_NOVA_LITE), "the Bedrock model id")?;
    }
    Ok(())
}

/// A client on another HTTP client, erased, and streamed as parts.
#[tokio::test]
async fn an_erased_model_streams_its_parts() -> Result<(), RigError> {
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
        match update {
            Update::Delta { text: delta, .. } => text.push_str(&delta),
            Update::Done(response) => done = Some(response),
            Update::Failed { error, .. } => return Err(error),
            _ => {}
        }
    }
    ensure(text == "Paris.", "the streamed text")?;
    ensure(
        done.is_some_and(|response| response.text() == "Paris."),
        "the final response",
    )?;
    Ok(())
}

/// Streams a story, keeping what arrived when the stream fails.
async fn tell_a_story(
    model: &DynModel<Completion>,
    history: &mut Vec<Message>,
) -> Result<(), RigError> {
    let mut stream = model.stream("Tell me a story.")?;
    let mut updates = stream.updates();
    while let Some(update) = updates.next().await {
        match update {
            Update::Delta { text, .. } => print!("{text}"),
            Update::Done(response) => history.push(response.into()),
            Update::Failed { error, partial } => {
                history.push(partial.into());
                return Err(error);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Reads a stream relayed over the bus: its failure is the last update,
/// with what arrived before it.
async fn read_relayed(events: StreamEvents, kept: &mut Vec<Message>) -> Result<(), RigError> {
    let mut stream = CompletionStream::relay("model", events);
    let mut updates = stream.updates();
    while let Some(update) = updates.next().await {
        if let Update::Failed { error, partial } = update {
            kept.push(partial.into());
            return Err(error);
        }
    }
    Ok(())
}

/// Calls once more when the transport failed and the failure is worth
/// retrying: the report's kind and verdict replace a downcast.
async fn call_with_one_retry(model: &DynModel<Completion>) -> Result<String, RigError> {
    match model.call("Hi").await {
        Err(error) if error.kind == ErrorKind::Http && error.retryable => {
            Ok(model.call("Hi").await?.text())
        }
        result => Ok(result?.text()),
    }
}

/// A stream cut before its end fails with what arrived: the caller keeps
/// the partial answer in its history and returns the error.
#[tokio::test]
async fn a_failed_stream_ends_with_its_partial_response() -> Result<(), RigError> {
    let http = MockStreamingClient {
        sse_bytes: bytes::Bytes::from_static(ANTHROPIC_CUT_STREAM.as_bytes()),
    };
    let model: DynModel<Completion> = Anthropic::new("test-key")
        .with_http(http)
        .completion(anthropic::CLAUDE_SONNET_4_6)
        .into();

    let mut kept = Vec::new();
    let relayed = read_relayed(Box::pin(model.stream("Tell me a story.")?), &mut kept).await;
    ensure(
        relayed.is_err_and(|error| error.kind == ErrorKind::Response),
        "the relayed truncation",
    )?;
    ensure(kept.len() == 1, "the relayed partial answer is kept")?;

    let mut history = Vec::new();
    let Err(error) = tell_a_story(&model, &mut history).await else {
        return Err(RigError::new(ErrorKind::Other, "the cut stream fails"));
    };
    ensure(error.kind == ErrorKind::Response, "a truncation")?;
    ensure(
        error.message.contains("without a terminal record"),
        "the truncation error",
    )?;
    ensure(history.len() == 1, "the partial answer is kept")?;
    ensure(
        matches!(
            history.last(),
            Some(Message::Assistant { content, .. }) if content.len() == 1
        ),
        "the partial answer holds the text that ended",
    )
}
