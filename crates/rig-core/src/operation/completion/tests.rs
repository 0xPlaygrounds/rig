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
    super::Completion::scope_to_wire(&mut request, &["anthropic"]);
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

    type Items = Vec<Result<StreamEvent, crate::error::ProviderError>>;
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

#[test]
fn a_gateway_attributes_reasoning_to_the_upstream_family() {
    use crate::providers::openai::wire::upstream_reasoning_issuer;
    assert_eq!(
        upstream_reasoning_issuer("openrouter", "anthropic/claude-haiku-4.5"),
        "anthropic"
    );
    assert_eq!(
        upstream_reasoning_issuer("openrouter", "~anthropic/claude-sonnet"),
        "anthropic"
    );
    assert_eq!(
        upstream_reasoning_issuer("openrouter", "openai/gpt-5-mini"),
        "openrouter/openai"
    );
    assert_eq!(
        upstream_reasoning_issuer("openrouter", "google/gemini-3-flash-preview"),
        "openrouter/google"
    );
    assert_eq!(
        upstream_reasoning_issuer("openrouter", "openrouter/auto"),
        "openrouter/openrouter"
    );
}

/// A request to OpenRouter replays only the requested model family's
/// reasoning, plus reasoning stamped with the gateway alone before issuers
/// were upstream-scoped, and reasoning of unknown provenance.
#[tokio::test]
async fn openrouter_replays_only_the_requested_familys_reasoning() {
    let issuers = [
        "anthropic",
        "openrouter/openai",
        "openrouter/google",
        "openrouter",
        "aws_bedrock",
    ];
    let content: Vec<AssistantContent> = issuers
        .iter()
        .map(|issuer| {
            AssistantContent::Reasoning(
                Reasoning::new_with_signature(issuer, Some(format!("sig-{issuer}")))
                    .with_provider(*issuer),
            )
        })
        .chain([AssistantContent::Reasoning(Reasoning::new_with_signature(
            "unknown",
            Some("sig-unknown".to_owned()),
        ))])
        .chain([AssistantContent::text("4")])
        .collect();
    // Through the chat wire itself and through the route wrapper
    // `completion` returns, which must delegate its issuers.
    async fn send<W: Wire<Op = super::Completion>>(
        wire: W,
        content: Vec<AssistantContent>,
    ) -> String {
        let mut request = history("unused");
        request.chat_history[1] = Message::Assistant { id: None, content };
        let http = RecordingHttpClient::new(Bytes::from_static(b"{}"));
        let _ = crate::driver::call(&wire, &http, request, None).await;
        let requests = http.requests();
        String::from_utf8_lossy(&requests[0].body).into_owned()
    }
    let body_for = |model: &'static str| {
        let content: Vec<AssistantContent> = content.clone();
        async move {
            let chat = send(
                OpenAI::with_key(&OPENROUTER, "test-key").chat(model),
                content.clone(),
            )
            .await;
            let routed = send(
                OpenAI::with_key(&OPENROUTER, "test-key").completion(model),
                content,
            )
            .await;
            assert_eq!(chat, routed, "the route wrapper scopes like the chat wire");
            chat
        }
    };
    let kept = |body: &str| {
        issuers
            .iter()
            .chain(["unknown"].iter())
            .filter(|issuer| body.contains(&format!("sig-{issuer}")))
            .copied()
            .collect::<Vec<_>>()
    };
    assert_eq!(
        kept(&body_for("google/gemini-3-flash-preview").await),
        ["openrouter/google", "openrouter", "unknown"]
    );
    assert_eq!(
        kept(&body_for("anthropic/claude-haiku-4.5").await),
        ["anthropic", "openrouter", "unknown"]
    );
    // A router or preset names no family, so every relayed family replays;
    // direct-provider reasoning of another service still does not.
    for router in ["openrouter/auto", "@preset/fast"] {
        assert_eq!(
            kept(&body_for(router).await),
            [
                "anthropic",
                "openrouter/openai",
                "openrouter/google",
                "openrouter",
                "unknown"
            ],
            "{router}"
        );
    }
}

/// OpenRouter's Responses route scopes replay and records reasoning issuers
/// the way its chat route does.
#[tokio::test]
async fn openrouter_responses_route_scopes_reasoning_by_family() {
    use crate::completion::CompletionModel as _;

    let issuers = [
        "anthropic",
        "openrouter/openai",
        "openrouter/google",
        "openrouter",
        "aws_bedrock",
    ];
    let content: Vec<AssistantContent> = issuers
        .iter()
        .map(|issuer| {
            AssistantContent::Reasoning(
                Reasoning::new(issuer)
                    .with_id(format!("rs-{issuer}"))
                    .with_provider(*issuer),
            )
        })
        .chain([
            AssistantContent::Reasoning(Reasoning::new("unknown").with_id("rs-unknown".to_owned())),
            AssistantContent::text("4"),
        ])
        .collect();
    let kept = |model: &'static str| {
        let content = content.clone();
        async move {
            let mut request = history("unused");
            request.chat_history[1] = Message::Assistant { id: None, content };
            let http = RecordingHttpClient::new(Bytes::from_static(b"{}"));
            let wire = OpenAI::with_key(&OPENROUTER, "test-key").responses(model);
            let _ = crate::driver::call(&wire, &http, request, None).await;
            let body = String::from_utf8_lossy(&http.requests()[0].body).into_owned();
            issuers
                .iter()
                .chain(["unknown"].iter())
                .filter(|issuer| body.contains(&format!("\"rs-{issuer}\"")))
                .copied()
                .collect::<Vec<_>>()
        }
    };
    assert_eq!(
        kept("openai/gpt-5-mini").await,
        ["openrouter/openai", "openrouter", "unknown"]
    );
    assert_eq!(
        kept("anthropic/claude-haiku-4.5").await,
        ["anthropic", "openrouter", "unknown"]
    );

    let reply = serde_json::json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "model": "openai/gpt-5-mini",
        "output": [
            { "type": "reasoning", "id": "rs_1", "summary": [{ "type": "summary_text", "text": "thinking" }], "encrypted_content": "ciphertext" },
            { "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
              "content": [{ "type": "output_text", "text": "4", "annotations": [] }] }
        ],
        "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
    });
    let response = crate::driver::Bound::new(
        OpenAI::with_key(&OPENROUTER, "test-key").responses("openai/gpt-5-mini"),
        RecordingHttpClient::new(Bytes::from(reply.to_string())),
    )
    .completion(history("unused"))
    .await
    .expect("the reply decodes");
    let recorded: Vec<_> = response
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => reasoning.provider.as_deref(),
            _ => None,
        })
        .collect();
    assert_eq!(recorded, ["openrouter/openai"]);
}

/// OpenRouter's Responses route returns Claude's thinking with a `signature`
/// beside it; the reasoning keeps it and a continuation sends it back, which
/// is what makes the thinking reach Claude again.
#[tokio::test]
async fn openrouter_responses_route_round_trips_a_claude_signature() {
    use crate::completion::CompletionModel as _;

    let model = "anthropic/claude-haiku-4.5";
    let reply = serde_json::json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "model": model,
        "output": [
            { "type": "reasoning", "id": "rs_tmp_1", "summary": [],
              "content": [{ "type": "reasoning_text", "text": "thinking it through" }],
              "signature": "claude-signature", "format": "anthropic-claude-v1" },
            { "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
              "content": [{ "type": "output_text", "text": "4", "annotations": [] }] }
        ],
        "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
    });
    let wire = OpenAI::with_key(&OPENROUTER, "test-key").responses(model);
    let response = crate::driver::Bound::new(
        wire.clone(),
        RecordingHttpClient::new(Bytes::from(reply.to_string())),
    )
    .completion(history("unused"))
    .await
    .expect("the reply decodes");
    let Some(AssistantContent::Reasoning(reasoning)) = response.choice.first() else {
        panic!("the reply opens with reasoning: {:?}", response.choice);
    };
    assert_eq!(reasoning.first_signature(), Some("claude-signature"));
    assert_eq!(reasoning.provider.as_deref(), Some("anthropic"));

    let mut request = history("unused");
    request.chat_history[1] = Message::Assistant {
        id: None,
        content: response.choice.clone(),
    };
    let http = RecordingHttpClient::new(Bytes::from_static(b"{}"));
    let _ = crate::driver::call(&wire, &http, request, None).await;
    let body: serde_json::Value =
        serde_json::from_slice(&http.requests()[0].body).expect("a JSON body");
    let item = body["input"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "reasoning"))
        .expect("the reasoning item is replayed");
    assert_eq!(item["id"], "rs_tmp_1");
    assert_eq!(item["signature"], "claude-signature");
    assert_eq!(item["content"][0]["text"], "thinking it through");
}

/// OpenRouter's reasoning records the family of the model that produced it,
/// on a unary reply and on a stream.
#[tokio::test]
async fn openrouter_reasoning_records_its_upstream_family() {
    use crate::completion::CompletionModel as _;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let reply = |model: &str| {
        serde_json::json!({
            "id": "gen-1", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{ "index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "4",
                "reasoning_details": [
                    { "type": "reasoning.text", "text": "thinking", "signature": "sig", "format": "anthropic-claude-v1", "index": 0 }
                ]
            }}],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })
    };
    let issuers = |choice: &[AssistantContent]| {
        choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(reasoning) => reasoning.provider.clone(),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    for (model, expected) in [
        ("anthropic/claude-4.5-haiku-20251001", "anthropic"),
        ("openai/gpt-5-mini", "openrouter/openai"),
    ] {
        let wire = OpenAI::with_key(&OPENROUTER, "test-key").chat(model);
        let unary = crate::driver::Bound::new(
            wire.clone(),
            RecordingHttpClient::new(Bytes::from(reply(model).to_string())),
        )
        .completion(history("unused"))
        .await
        .expect("the reply decodes");
        assert_eq!(issuers(&unary.choice), [expected], "unary {model}");

        let mut chunk = reply(model);
        chunk["object"] = "chat.completion.chunk".into();
        let message = chunk["choices"][0]["message"].take();
        chunk["choices"][0]["delta"] = message;
        let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
        let mut stream = crate::driver::Bound::new(
            wire,
            MockStreamingClient {
                sse_bytes: Bytes::from(sse),
            },
        )
        .stream(history("unused"))
        .await
        .expect("the stream opens");
        while stream.next().await.is_some() {}
        let streamed = stream.finish().expect("a terminal record");
        assert_eq!(issuers(&streamed.choice), [expected], "streamed {model}");
    }
}
