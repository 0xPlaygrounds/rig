use bytes::Bytes;

use crate::completion::{CompletionRequest, Message};
use crate::message::{AssistantContent, Reasoning};
use crate::providers::anthropic::wire::Anthropic;
use crate::providers::gemini::Gemini;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::openai::wire::{DEEPSEEK, OPENROUTER, OpenAI};
use crate::test_utils::RecordingHttpClient;
use crate::wire::{HasCompletion, Wire};

const OWN: &str = "own-opaque-reasoning-state";
const FOREIGN: &str = "foreign-opaque-reasoning-state";
const UNKNOWN: &str = "unknown-opaque-reasoning-state";

/// One of each reasoning representation, signed or encrypted with `data`.
/// Each carries an item id, which the Responses encoder requires to replay.
fn reasoning(data: &str) -> Vec<Reasoning> {
    [
        Reasoning::new_with_signature(&format!("thought {data}"), Some(data.to_owned())),
        Reasoning::encrypted(format!("{data}-encrypted")),
        Reasoning::redacted(format!("{data}-redacted")),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, mut reasoning)| {
        reasoning.id = Some(format!("rs_{data}_{index}"));
        reasoning
    })
    .collect()
}

fn history(own: &str) -> CompletionRequest {
    let content = [
        reasoning(OWN)
            .into_iter()
            .map(|reasoning| reasoning.with_provider(own))
            .collect::<Vec<_>>(),
        reasoning(FOREIGN)
            .into_iter()
            .map(|reasoning| reasoning.with_provider("another.provider"))
            .collect(),
        reasoning(UNKNOWN),
    ]
    .concat()
    .into_iter()
    .map(AssistantContent::Reasoning)
    .chain([AssistantContent::text("The answer is 4.")])
    .collect();
    CompletionRequest {
        model: None,
        chat_history: vec![
            Message::user("What is 2 + 2?"),
            Message::Assistant { id: None, content },
            Message::user("And 3 + 3?"),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// The body `wire` sends for a history holding its own, another
/// provider's, and unattributed reasoning.
async fn sent<W>(wire: W) -> String
where
    W: Wire<Op = super::Completion>,
{
    let own = wire.name().to_owned();
    let http = RecordingHttpClient::new(Bytes::from_static(b"{}"));
    // The canned reply does not decode; only the request matters here.
    let _ = crate::driver::call(&wire, &http, history(&own), None).await;
    let requests = http.requests();
    let request = requests.first().expect("the wire sent its request");
    String::from_utf8_lossy(&request.body).into_owned()
}

fn assert_scoped(body: &str, wire: &str, replays_opaque: bool) {
    assert!(
        !body.contains(FOREIGN),
        "{wire} must not replay another provider's reasoning: {body}"
    );
    if replays_opaque {
        assert!(
            body.contains(OWN),
            "{wire} replays its own reasoning: {body}"
        );
        assert!(
            body.contains(UNKNOWN),
            "{wire} replays reasoning of unknown provenance: {body}"
        );
    }
}

#[tokio::test]
async fn anthropic_replays_only_its_own_reasoning() {
    let body = sent(Anthropic::new("test-key").completion("claude-sonnet-4-6")).await;
    assert_scoped(&body, "anthropic", true);
}

#[tokio::test]
async fn gemini_replays_only_its_own_reasoning() {
    let body = sent(GenerateContent::new(
        Gemini::new("test-key"),
        "gemini-2.5-flash",
    ))
    .await;
    assert_scoped(&body, "gemini", true);
}

#[tokio::test]
async fn openai_responses_replays_only_its_own_reasoning() {
    let body = sent(OpenAI::new("test-key").responses("gpt-5.2")).await;
    assert_scoped(&body, "openai responses", true);
}

#[tokio::test]
async fn openrouter_replays_only_its_own_reasoning() {
    let body = sent(OpenAI::with_key(&OPENROUTER, "test-key").chat("openai/gpt-5.2")).await;
    assert_scoped(&body, "openrouter", true);
}

#[tokio::test]
async fn deepseek_replays_only_its_own_reasoning() {
    // DeepSeek replays reasoning as plain text, so only the foreign text is
    // checked.
    let body = sent(OpenAI::with_key(&DEEPSEEK, "test-key").chat("deepseek-v4-flash")).await;
    assert!(
        !body.contains(FOREIGN),
        "deepseek must not replay foreign reasoning: {body}"
    );
}

#[test]
fn provenance_round_trips_and_older_histories_load() {
    let signed =
        Reasoning::new_with_signature("thought", Some("sig".to_owned())).with_provider("anthropic");
    let json = serde_json::to_value(&signed).expect("serializes");
    assert_eq!(json["provider"], "anthropic");
    let back: Reasoning = serde_json::from_value(json.clone()).expect("loads");
    assert_eq!(back, signed);

    let mut legacy = json;
    legacy
        .as_object_mut()
        .expect("an object")
        .remove("provider");
    let loaded: Reasoning = serde_json::from_value(legacy).expect("a pre-provenance value loads");
    assert_eq!(loaded.provider, None);
    assert!(loaded.replayable_to("anthropic") && loaded.replayable_to("gcp.gemini"));
    assert!(signed.replayable_to("anthropic") && !signed.replayable_to("gcp.gemini"));
}

#[test]
fn a_turn_that_held_only_foreign_reasoning_is_omitted() {
    use crate::wire::Operation;

    let mut request = history("anthropic");
    request.chat_history.insert(
        1,
        Message::Assistant {
            id: None,
            content: reasoning(FOREIGN)
                .into_iter()
                .map(|reasoning| AssistantContent::Reasoning(reasoning.with_provider("another")))
                .collect(),
        },
    );
    assert_eq!(request.chat_history.len(), 4);
    super::Completion::scope_to_wire(&mut request, "anthropic");
    assert_eq!(request.chat_history.len(), 3, "{:?}", request.chat_history);
    assert!(request.chat_history.iter().all(|message| match message {
        Message::Assistant { content, .. } => !content.is_empty(),
        _ => true,
    }));
}

#[test]
fn a_stream_stamps_its_reasoning_with_the_terminal_issuer() {
    use crate::message::AssistantContent;
    use crate::streaming::{StreamFinal, stamp_reasoning};

    let terminal = StreamFinal::new("aws_bedrock", Default::default(), serde_json::Value::Null)
        .with_reasoning_issuer("anthropic");
    assert_eq!(terminal.issuer(), "anthropic");
    let bare = StreamFinal::new("openai", Default::default(), serde_json::Value::Null);
    assert_eq!(bare.issuer(), "openai");

    let choice = stamp_reasoning(
        vec![
            AssistantContent::Reasoning(Reasoning::new("unstamped")),
            AssistantContent::Reasoning(Reasoning::new("stamped").with_provider("gcp.gemini")),
        ],
        terminal.issuer(),
    );
    let issuers: Vec<_> = choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => reasoning.provider.as_deref(),
            _ => None,
        })
        .collect();
    assert_eq!(issuers, ["anthropic", "gcp.gemini"]);

    let json = serde_json::to_value(&terminal).expect("serializes");
    let back: StreamFinal = serde_json::from_value(json).expect("loads");
    assert_eq!(
        back.issuer(),
        "anthropic",
        "the issuer crosses the effect bus"
    );
}

#[tokio::test]
async fn a_stream_names_its_reasoning_issuer_only_when_it_knows_it() {
    use crate::streaming::{StreamEvent, StreamFinal, StreamingCompletionResponse};
    use futures::StreamExt;

    type Items = Vec<Result<StreamEvent, crate::completion::CompletionError>>;
    let terminal = || StreamFinal::new("aws_bedrock", Default::default(), serde_json::Value::Null);

    // A provider that opens its own stream knows the issuer up front.
    let mut stream = StreamingCompletionResponse::stream(
        "aws_bedrock",
        Box::pin(futures::stream::iter(vec![Ok(StreamEvent::Final(
            terminal().with_reasoning_issuer("anthropic"),
        ))] as Items)),
    )
    .with_reasoning_issuer("anthropic");
    assert_eq!(
        stream.reasoning_issuer(),
        Some("anthropic"),
        "before the terminal"
    );
    while stream.next().await.is_some() {}
    assert_eq!(stream.reasoning_issuer(), Some("anthropic"), "after it");

    let plain = StreamingCompletionResponse::stream(
        "openai",
        Box::pin(futures::stream::iter(Items::new())),
    );
    assert_eq!(plain.reasoning_issuer(), Some("openai"));

    // A stream rebuilt from bus events is opened under a handler label: the
    // issuer is unknown until its terminal names it.
    let events: crate::streaming::StreamEvents = Box::pin(futures::stream::iter(vec![Ok(
        StreamEvent::Final(terminal().with_reasoning_issuer("anthropic")),
    )]));
    let mut rebuilt = StreamingCompletionResponse::from_events("default", events);
    assert_eq!(rebuilt.reasoning_issuer(), None);
    while rebuilt.next().await.is_some() {}
    assert_eq!(rebuilt.reasoning_issuer(), Some("anthropic"));
}
