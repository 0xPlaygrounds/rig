//! Recorded Messages replies that cite, decoded through the Anthropic wire:
//! each cited text block carries its citations as `Text::citations`, the
//! provider's JSON stays in the block's native item, and the reply's cost
//! is priced from the catalog, because Anthropic reports none. A streamed
//! recording is decoded a second time from the unary document its stream
//! rebuilds, and both decodes cite alike.

use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionResponse, Cost};
use rig::message::{AssistantContent, DocumentRange, Source, SourceLocation};
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde_json::Value;

use crate::cassettes::recorded_interaction_bodies;

/// The recorded reply at `index` of `scenario`, and whether rig streamed it.
fn recorded(scenario: &str, index: usize) -> (String, bool) {
    let (request, response) = recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .nth(index)
        .unwrap_or_else(|| panic!("{scenario} records interaction {index}"));
    let request: Value = serde_json::from_str(&request).expect("the request is JSON");
    (response, request["stream"] == Value::Bool(true))
}

/// The response `body` decodes to on `model`, streamed or whole.
async fn decode(model: &str, body: String, streamed: bool) -> CompletionResponse {
    let reply = if streamed {
        MockHttpResponse::success_typed(body, "text/event-stream")
    } else {
        MockHttpResponse::success_typed(body, "application/json")
    };
    let model = AnthropicConfig::new("cassette")
        .connect(SequencedHttpClient::new([reply]))
        .completion(model);
    let request = CompletionRequest::new("cite").max_tokens(1024);
    if !streamed {
        return model.call(request).await.expect("the reply decodes");
    }
    let mut stream = model.stream(request).expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("the stream yields no error");
    }
    stream.finish().await.expect("the stream folds")
}

/// The one cited text block of `response`: its text, its citations' spans
/// and sources, and the citation JSON of its native item.
fn cited(response: &CompletionResponse) -> (String, Vec<(Option<usize>, Vec<Source>)>, Value) {
    let mut cited = response.choice.iter().filter_map(|content| match content {
        AssistantContent::Text(text) if !text.citations().is_empty() => Some((
            text.text.clone(),
            text.citations()
                .iter()
                .map(|citation| {
                    assert_eq!(text.cited(citation), Some(text.text.as_str()));
                    (
                        citation.span.map(|span| span.start()),
                        citation.sources.clone(),
                    )
                })
                .collect(),
            content
                .native_item()
                .and_then(|item| item.get("citations"))
                .cloned()
                .unwrap_or_default(),
        )),
        _ => None,
    });
    let first = cited.next().expect("a cited text block");
    assert!(cited.next().is_none(), "one cited text block");
    first
}

/// `expected` within a nanodollar, part by part.
fn assert_cost(actual: Option<Cost>, expected: [f64; 5]) {
    let actual = actual.expect("the catalog prices the reply");
    let parts = [
        actual.input,
        actual.output,
        actual.cache_read,
        actual.cache_write,
        actual.total,
    ];
    for (part, want) in parts.into_iter().zip(expected) {
        assert!((part - want).abs() < 1e-9, "{parts:?} != {expected:?}");
    }
}

/// Decode a recorded reply, check its cited block, and for a stream, decode
/// the unary document it rebuilds and check that it cites alike.
async fn check(
    scenario: &str,
    index: usize,
    model: &str,
    text: &str,
    sources: Vec<Source>,
    cost: [f64; 5],
) {
    let (body, streamed) = recorded(scenario, index);
    assert!(body.contains("_location"), "{scenario}#{index} cites");
    let response = decode(model, body, streamed).await;
    let (cited_text, citations, native) = cited(&response);
    assert_eq!(cited_text, text);
    assert_eq!(citations, vec![(None, sources)]);
    assert_eq!(native.as_array().map(Vec::len), Some(1));
    assert_cost(response.usage.cost, cost);
    if streamed {
        let unary = decode(model, response.raw.to_string(), false).await;
        assert_eq!(cited(&unary), (cited_text, citations, native));
        assert_eq!(unary.usage.cost, response.usage.cost);
    }
}

/// `char_location`, whole (`long_run_caching/document_60.yaml#0`).
#[tokio::test]
async fn a_char_location_cites_a_document_range() {
    check(
        "long_run_caching/document_60",
        0,
        "claude-opus-5-5",
        "requests about orders above 375 EUR need a supervisor's approval, which support \
         records on the order before replying to the customer.",
        vec![
            Source::new(SourceLocation::Document {
                index: Some(0),
                id: None,
                within: Some(DocumentRange::Chars(4998..5136)),
            })
            .title("Northwind Outfitters store policy")
            .cited_text(
                "8.2 Requests about orders above 375 EUR need a supervisor's approval, which \
                 support records on the order before replying to the customer.\n",
            ),
        ],
        // Opus 5.5: 4 uncached, 15948 written, 72 out.
        [0.000016, 0.00144, 0.0, 0.07974, 0.081196],
    )
    .await;
}

/// `char_location`, streamed as a `citations_delta`
/// (`long_run_caching/document_60.yaml#1`).
#[tokio::test]
async fn a_streamed_char_location_cites_a_document_range() {
    check(
        "long_run_caching/document_60",
        1,
        "claude-opus-5-5",
        STREAMED_CHAR_TEXT,
        vec![
            Source::new(SourceLocation::Document {
                index: Some(0),
                id: None,
                within: Some(DocumentRange::Chars(9953..10089)),
            })
            .title("Northwind Outfitters store policy")
            .cited_text(
                "15.3 The store answers every customs request within 71 business hours and \
                 confirms the outcome by email, quoting this section's number.\n",
            ),
        ],
        // Opus 5.5: 4 uncached, 15948 read, 96 written, 69 out.
        [0.000016, 0.00138, 0.0031896, 0.00048, 0.0050656],
    )
    .await;
}

/// `page_location`, streamed (`models/sonnet_5_5/session.yaml#14`).
#[tokio::test]
async fn a_streamed_page_location_cites_pages() {
    check(
        "models/sonnet_5_5/session",
        14,
        "claude-sonnet-5-5",
        STREAMED_PAGE_TEXT,
        vec![
            Source::new(SourceLocation::Document {
                index: Some(1),
                id: None,
                within: Some(DocumentRange::Pages(1..2)),
            })
            .title("Bitcoin Whitepaper")
            .cited_text(
                "Bitcoin: A Peer-to-Peer Electronic Cash System\r\nSatoshi Nakamoto\r\n\
                 satoshin@gmx.com\r\nwww.bitcoin.org\r\nAbstract. ",
            ),
        ],
        // Sonnet 5.5: 4 uncached, 5297 read, 24048 written, 95 out.
        [0.000008, 0.00095, 0.0010594, 0.06012, 0.0621374],
    )
    .await;
}

/// `char_location`, whole (`models/sonnet_5_5/session.yaml#17`).
#[tokio::test]
async fn a_char_location_on_sonnet_cites_a_document_range() {
    check(
        "models/sonnet_5_5/session",
        17,
        "claude-sonnet-5-5",
        "Rust's three goals are safety, speed, and concurrency.",
        vec![
            Source::new(SourceLocation::Document {
                index: Some(0),
                id: None,
                within: Some(DocumentRange::Chars(0..94)),
            })
            .title("Rust Goals")
            .cited_text(
                "Rust is a systems programming language focused on three goals: safety, \
                 speed, and concurrency.",
            ),
        ],
        // Sonnet 5.5: 4 uncached, 29580 read, 38 written, 41 out.
        [0.000008, 0.00041, 0.005916, 0.000095, 0.006429],
    )
    .await;
}

/// `web_search_result_location`, streamed after a server tool
/// (`models/sonnet_5_5/session.yaml#53`).
#[tokio::test]
async fn a_streamed_web_search_result_cites_its_page() {
    check(
        "models/sonnet_5_5/session",
        53,
        "claude-sonnet-5-5",
        STREAMED_WEB_TEXT,
        vec![
            Source::new(SourceLocation::Url {
                url: "https://releases.rs/".to_owned(),
            })
            .title("Rust Versions")
            .cited_text(
                "Rust Versions Stable: 1.99.0 Beta: 1.100.0 (12 November, 2026, 38 days left) \
                 Nightly: 1.101.0 (24 December, 2026, 80 days left) Ongoing Stabilization ...",
            ),
        ],
        // Sonnet 5.5: 14236 uncached, 93 out.
        [0.028472, 0.00093, 0.0, 0.0, 0.029402],
    )
    .await;
}

/// `web_search_result_location`, whole
/// (`opus_4_8/messages_preserve_system_role_after_server_tool_result.yaml#0`).
#[tokio::test]
async fn a_web_search_result_cites_its_page() {
    check(
        "opus_4_8/messages_preserve_system_role_after_server_tool_result",
        0,
        "claude-opus-4-8",
        "A clear daytime sky is blue.",
        vec![
            Source::new(SourceLocation::Url {
                url: "https://math.ucr.edu/home/baez/physics/General/BlueSky/blue_sky.html"
                    .to_owned(),
            })
            .title("Why is the sky blue?")
            .cited_text(
                "A clear cloudless day-time sky is blue because molecules in the air scatter \
                 blue light from the Sun more than they scatter red light.\n\n",
            ),
        ],
        // Opus 4.8: 14382 uncached, 86 out.
        [0.07191, 0.00215, 0.0, 0.0, 0.07406],
    )
    .await;
}

const STREAMED_CHAR_TEXT: &str = "the store answers every customs request within 71 business \
    hours and confirms the outcome by email, quoting this section's number.";
const STREAMED_PAGE_TEXT: &str =
    "The paper is titled \"Bitcoin: A Peer-to-Peer Electronic Cash System.\"";
const STREAMED_WEB_TEXT: &str = "Rust 1.99.0 is the latest stable release.";
