//! Citations and reported cost on the Chat wire, from recorded replies where
//! a recording carries them (Perplexity, Venice, OpenRouter) and from
//! hand-built frames for the shapes no recording carries yet: OpenAI,
//! OpenRouter and MiMo `url_citation` annotations, Mistral `reference`
//! chunks, Z.AI `web_search`, xAI's `cost_in_usd_ticks` and a gateway's
//! top-level `cost`. A streamed reply and the unary document it rebuilds
//! give the same citations and cost.

use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, Cost, Message};
use crate::message::{AssistantContent, AssistantMessage, Citation, Source, SourceLocation};
use crate::providers::openai::wire::chat::Chat;
use crate::providers::openai::wire::{
    Dialect, MISTRAL, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY, VENICE, XIAOMIMIMO, ZAI,
};
use crate::test_utils::provider_extensions::{
    chat_reply, chat_stream, encoded_body, recorded_reply, recorded_stream, reply_of,
    streamed_reply_of,
};
use crate::wire::Mode;

fn wire(dialect: &'static Dialect, model: &str) -> Chat {
    OpenAIConfig::with_key(dialect, "key").chat(model)
}

/// Each text block's citations, in order.
fn citations(response: &CompletionResponse) -> Vec<Vec<Citation>> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.citations().to_vec()),
            _ => None,
        })
        .collect()
}

/// A citation of a whole block by the page at `url`.
fn url(url: &str) -> Source {
    Source::new(SourceLocation::Url {
        url: url.to_owned(),
    })
}

/// Fold a stream, then its rebuilt unary document as a unary reply, and
/// check both give the same citations and cost.
async fn streamed_and_refolded(wire: Chat, stream: String) -> CompletionResponse {
    let streamed = streamed_reply_of(wire.clone(), stream).await;
    let refolded = reply_of(wire, streamed.raw.to_string()).await;
    assert_eq!(citations(&streamed), citations(&refolded));
    assert_eq!(streamed.usage.cost, refolded.usage.cost);
    streamed
}

/// `fixtures/cassettes/perplexity/agent/completion_smoke.yaml`: the
/// `citations` list cites the answer, one whole-block citation per URL,
/// titled from `search_results`, and `usage.cost` splits input and output
/// from a total that holds the request fee.
#[tokio::test]
async fn perplexity_cites_its_url_list_and_reports_its_cost() {
    let body = recorded_reply("perplexity", "agent/completion_smoke", 0);
    let recorded: Value = serde_json::from_str(&body).expect("the reply is JSON");
    let response = reply_of(wire(&PERPLEXITY, "sonar"), body).await;

    let [cited] = citations(&response).try_into().expect("one text block");
    let urls = recorded["citations"].as_array().expect("a URL list");
    assert_eq!(cited.len(), 18);
    assert_eq!(cited.len(), urls.len());
    for (citation, recorded) in cited.iter().zip(urls) {
        assert_eq!(citation.span, None);
        let [source] = citation.sources.as_slice() else {
            panic!("one source per URL: {citation:?}");
        };
        assert_eq!(
            source.location,
            SourceLocation::Url {
                url: recorded.as_str().unwrap_or_default().to_owned()
            }
        );
        assert!(source.title.is_some(), "titled from search_results");
    }
    assert_eq!(
        cited[0].sources[0],
        url("https://learn.microsoft.com/en-us/windows/dev-environment/rust/overview")
            .title("Get Started With Rust")
    );

    let cost = response.usage.cost.expect("Perplexity reports a cost");
    assert_eq!(cost.input, 0.00003);
    assert_eq!(cost.output, 0.00007);
    assert_eq!(cost.total, 0.0051);
    // Cost is no token arithmetic: the counters are the recorded ones.
    assert_eq!(response.usage.input_tokens, Some(27));
    assert_eq!(response.usage.output_tokens, Some(74));
}

/// `fixtures/cassettes/perplexity/raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object.yaml`:
/// a stream restating its list on every chunk cites each URL once.
#[tokio::test]
async fn a_perplexity_stream_cites_its_url_list_once() {
    let stream = recorded_stream(
        "perplexity",
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object",
        0,
    );
    let response = streamed_and_refolded(wire(&PERPLEXITY, "sonar"), stream).await;
    let [cited] = citations(&response).try_into().expect("one text block");
    assert_eq!(cited.len(), 18);
    assert_eq!(
        cited[0].sources[0].location,
        SourceLocation::Url {
            url: "https://www.dodorouter.com/docs/integrations/claude-code/".to_owned()
        }
    );
    let cost = response.usage.cost.expect("Perplexity reports a cost");
    assert_eq!(
        (cost.input, cost.output, cost.total),
        (0.00001, 0.0, 0.00501)
    );
}

/// `fixtures/cassettes/venice/venice_parameters/web_search_on.yaml`: the
/// echoed `web_search_citations` cite the answer and `cost.usd` is its cost.
#[tokio::test]
async fn venice_cites_its_web_search_and_reports_its_cost() {
    let body = recorded_reply("venice", "venice_parameters/web_search_on", 0);
    let response = reply_of(wire(&VENICE, "qwen3-5-9b"), body).await;

    let [cited] = citations(&response).try_into().expect("one text block");
    assert_eq!(cited.len(), 10);
    let first = &cited[0];
    assert_eq!(first.span, None);
    assert_eq!(
        first.sources[0].location,
        SourceLocation::Url {
            url: "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_owned()
        }
    );
    assert_eq!(
        first.sources[0].title.as_deref(),
        Some("Rust (programming language) - Wikipedia")
    );
    assert!(
        first.sources[0].cited_text.is_some(),
        "the result's content"
    );
    assert_eq!(response.usage.cost, Some(Cost::from_total(0.0105599)));
}

/// `fixtures/cassettes/venice/streaming/streaming_smoke.yaml`: the cost of
/// the last chunk.
#[tokio::test]
async fn a_venice_stream_reports_its_cost() {
    let stream = recorded_stream("venice", "streaming/streaming_smoke", 0);
    let response = streamed_and_refolded(wire(&VENICE, "qwen3-5-9b"), stream).await;
    assert_eq!(
        response.usage.cost,
        Some(Cost::from_total(0.000_239_400_000_000_000_02))
    );
    assert_eq!(citations(&response), vec![Vec::<Citation>::new()]);
}

/// `fixtures/cassettes/openrouter/agent/completion_smoke.yaml` and
/// `fixtures/cassettes/openrouter/streaming/example_streaming_prompt.yaml`:
/// `usage.cost` is the turn's cost, whatever the catalog would price it at.
#[tokio::test]
async fn openrouter_reports_its_cost_on_both_reply_shapes() {
    let body = recorded_reply("openrouter", "agent/completion_smoke", 0);
    let unary = reply_of(wire(&OPENROUTER, "openai/gpt-4o-mini"), body).await;
    assert_eq!(unary.usage.cost, Some(Cost::from_total(0.0000351)));
    assert_eq!(unary.usage.input_tokens, Some(38));
    assert_eq!(unary.usage.output_tokens, Some(49));

    let stream = recorded_stream("openrouter", "streaming/example_streaming_prompt", 0);
    let streamed = streamed_and_refolded(wire(&OPENROUTER, "openai/gpt-4o-mini"), stream).await;
    assert_eq!(streamed.usage.cost, Some(Cost::from_total(0.0000258)));
}

/// The message annotations OpenAI, OpenRouter's web plugin and MiMo send.
/// No Chat recording carries one yet, so the frames are built by hand from
/// the documented shape.
fn annotations() -> Value {
    json!([
        {"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 4,
            "url": "https://rig.rs", "title": "Rig"}},
        {"type": "url_citation", "url_citation": {"start_index": 5, "end_index": 9,
            "url": "https://docs.rig.rs", "title": "Docs", "content": "Rig docs"}},
        {"type": "file_citation", "file_citation": {"file_id": "f"}}
    ])
}

fn annotated() -> Vec<Citation> {
    vec![
        Citation::new([url("https://rig.rs").title("Rig")]),
        Citation::new([url("https://docs.rig.rs")
            .title("Docs")
            .cited_text("Rig docs")]),
    ]
}

/// Hand-built: no Chat recording carries an annotation.
#[tokio::test]
async fn url_citations_cite_the_whole_text_on_both_reply_shapes() {
    for dialect in [&OPENAI, &OPENROUTER, &XIAOMIMIMO] {
        let unary = reply_of(
            wire(dialect, "model"),
            chat_reply(json!({"choices": [{"message": {"content": "Rig is a crate",
                "annotations": annotations()}}]})),
        )
        .await;
        assert_eq!(citations(&unary), vec![annotated()], "{}", dialect.name);

        // OpenAI streams the annotations on a delta after the text.
        let stream = chat_stream(&[
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"content": "Rig is "}}]}),
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"content": "a crate"}}]}),
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"annotations": annotations()},
                    "finish_reason": "stop"}]}),
        ]);
        let streamed = streamed_and_refolded(wire(dialect, "model"), stream).await;
        assert_eq!(citations(&streamed), citations(&unary), "{}", dialect.name);
    }
}

/// Hand-built: no Mistral recording carries a `reference` chunk. Each cites
/// the text before it and stays its own replayed block.
#[tokio::test]
async fn mistral_references_cite_the_text_before_them() {
    let content = json!([
        {"type": "text", "text": "Rig is a crate"},
        {"type": "reference", "reference_ids": [1, 2]},
        {"type": "text", "text": " for agents"},
        {"type": "reference", "reference_ids": [3]}
    ]);
    let document = |ids: &[&str]| {
        Citation::new(ids.iter().map(|id| {
            Source::new(SourceLocation::Document {
                index: None,
                id: Some((*id).to_owned()),
                within: None,
            })
        }))
    };
    let expected = vec![vec![document(&["1", "2"])], vec![document(&["3"])]];

    let unary = reply_of(
        wire(&MISTRAL, "mistral-medium-latest"),
        chat_reply(json!({"choices": [{"message": {"content": content}}]})),
    )
    .await;
    assert_eq!(citations(&unary), expected);
    let opaque = unary
        .choice
        .iter()
        .filter(|content| matches!(content, AssistantContent::Opaque(_)))
        .count();
    assert_eq!(
        opaque, 2,
        "each reference stays a block: {:?}",
        unary.choice
    );

    let chunks: Vec<Value> = content
        .as_array()
        .into_iter()
        .flatten()
        .map(|part| {
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"content": [part]}}]})
        })
        .chain([
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ])
        .collect();
    let streamed = streamed_reply_of(
        wire(&MISTRAL, "mistral-medium-latest"),
        chat_stream(&chunks),
    )
    .await;
    assert_eq!(citations(&streamed), expected);
}

/// Hand-built: no Z.AI recording carries `web_search`.
#[tokio::test]
async fn zai_cites_its_web_search() {
    let response = reply_of(
        wire(&ZAI, "glm-4.6"),
        chat_reply(json!({"web_search": [
            {"title": "Rig", "link": "https://rig.rs", "content": "Rig is", "refer": "ref_1"},
            {"title": "Untitled"}
        ]})),
    )
    .await;
    assert_eq!(
        citations(&response),
        vec![vec![Citation::new([url("https://rig.rs")
            .title("Rig")
            .cited_text("Rig is")])]]
    );
}

/// Hand-built: xAI's Chat usage counts its cost in ticks, 10^10 per USD;
/// its Chat route has no recording.
#[tokio::test]
async fn xai_cost_ticks_are_dollars() {
    let response = reply_of(
        wire(&crate::providers::xai::DIALECT, "grok-4"),
        chat_reply(json!({"usage": {"cost_in_usd_ticks": 158_500_000}})),
    )
    .await;
    assert_eq!(response.usage.cost, Some(Cost::from_total(0.01585)));
    assert_eq!(response.usage.input_tokens, Some(1));
}

/// Ported from #2243: a gateway (OpenCode Go) states the cost in a chunk of
/// its own after the usage chunk, as a string. The cost is the reported
/// figure and the counters stay the reported ones.
#[tokio::test]
async fn a_gateways_cost_chunk_is_the_turns_cost() {
    let stream = chat_stream(&[
        json!({"choices": [{"delta": {"content": "Hello", "tool_calls": []}}], "usage": null}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15}}),
        json!({"choices": [], "x-opencode-type": "inference-cost", "cost": "0.00006972"}),
    ]);
    let response = streamed_reply_of(wire(&OPENAI, "gpt-4.1-nano"), stream).await;
    assert_eq!(response.usage.cost, Some(Cost::from_total(0.00006972)));
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

/// Ported from #2243: a top-level cost reads as a string or a number, and
/// a reply without one reports none.
#[test]
fn a_top_level_cost_reads_as_a_string_or_a_number() {
    let fields = |reply: Value| reply.as_object().cloned().unwrap_or_default();
    assert_eq!(
        super::cost(None, &fields(json!({"cost": "0.00006972"}))),
        Some(Cost::from_total(0.00006972))
    );
    assert_eq!(
        super::cost(None, &fields(json!({"cost": 0.001}))),
        Some(Cost::from_total(0.001))
    );
    assert_eq!(super::cost(None, &fields(json!({"usage": null}))), None);
    assert_eq!(
        super::cost(None, &fields(json!({"cost": "not a number"}))),
        None
    );
}

/// A cited turn replays as it did before citations: the encoder never reads
/// them, so the body equals the one sent for the same turn without them.
#[tokio::test]
async fn a_cited_turn_replays_unchanged() {
    for (dialect, model, reply) in [
        (
            &OPENAI,
            "gpt-4.1-nano",
            json!({"choices": [{"message": {"content": "Rig is a crate",
                "annotations": annotations()}}]}),
        ),
        (
            &MISTRAL,
            "mistral-medium-latest",
            json!({"choices": [{"message": {"content": [
                {"type": "text", "text": "Rig"},
                {"type": "reference", "reference_ids": [1]}]}}]}),
        ),
    ] {
        let wire = wire(dialect, model);
        let response = reply_of(wire.clone(), chat_reply(reply)).await;
        assert!(
            citations(&response).iter().any(|cited| !cited.is_empty()),
            "{}",
            dialect.name
        );
        let uncited: Vec<AssistantContent> = response
            .choice
            .iter()
            .cloned()
            .map(|content| match content {
                AssistantContent::Text(mut text) => {
                    text.clear_citations();
                    AssistantContent::Text(text)
                }
                other => other,
            })
            .collect();
        let body = |content: Vec<AssistantContent>| {
            let mut request = CompletionRequest::new("next");
            request.chat_history = vec![
                Message::user("q"),
                Message::Assistant(AssistantMessage {
                    content,
                    ..response.head()
                }),
            ];
            encoded_body(&wire, request, Mode::Unary)
                .unwrap_or_else(|error| panic!("the request encodes: {error}"))
        };
        assert_eq!(
            body(response.choice.clone()),
            body(uncited),
            "{}",
            dialect.name
        );
    }
}
