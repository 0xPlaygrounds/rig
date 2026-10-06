//! Hand-built GenerateContent replies: no recording carries grounding or
//! recitation metadata yet.

use serde_json::{Value, json};

use crate::completion::CompletionResponse;
use crate::message::{AssistantContent, Source, SourceLocation, Text};
use crate::test_utils::history::{assert_restated_agrees, decode};
use crate::wire::{Mode, WireFrame};

/// The answer every test cites. The dash before the claims is three bytes
/// and one character, so a span read in the wrong unit misses.
const ANSWER: &str = "Spain won \u{2014} 2\u{2013}1. Nico Williams scored first. Oyarzabal won it.";

fn wire() -> crate::providers::gemini::completion::GenerateContent {
    crate::providers::gemini::GeminiConfig::new("k").completion("gemini-2.5-flash")
}

fn frame(value: Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

/// A chunk of one candidate holding `parts`, with `extra` candidate fields.
fn chunk(parts: Value, extra: Value) -> Value {
    let mut candidate = json!({"content": {"role": "model", "parts": parts}});
    if let (Some(candidate), Value::Object(extra)) = (candidate.as_object_mut(), extra) {
        candidate.extend(extra);
    }
    json!({"candidates": [candidate]})
}

fn finished(mut chunk: Value) -> Value {
    chunk["candidates"][0]["finishReason"] = json!("STOP");
    chunk
}

fn bytes_of(text: &str) -> usize {
    ANSWER.find(text).expect("the answer holds the text")
}

/// A grounding support of `text` in part `part`, which starts at byte
/// `offset` of the answer, by the chunks at `indices`.
fn support(part: usize, offset: usize, text: &str, indices: &[u32], scores: &[f32]) -> Value {
    let start = bytes_of(text) - offset;
    json!({
        "segment": {"partIndex": part, "startIndex": start, "endIndex": start + text.len(), "text": text},
        "groundingChunkIndices": indices,
        "confidenceScores": scores,
    })
}

fn grounding(supports: Vec<Value>) -> Value {
    json!({"groundingMetadata": {
        "webSearchQueries": ["euro 2024 final"],
        "groundingChunks": [
            {"web": {"uri": "https://example.com/final", "title": "example.com"}},
            {"web": {"uri": "https://example.org/report", "title": "example.org"}},
        ],
        "groundingSupports": supports,
    }})
}

/// The text blocks of `response`.
fn texts(response: &CompletionResponse) -> Vec<&Text> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .collect()
}

/// Each citation of `text` as the text it covers and its sources.
fn cited(text: &Text) -> Vec<(Option<String>, Vec<Source>)> {
    text.citations()
        .iter()
        .map(|citation| {
            (
                citation
                    .span
                    .as_ref()
                    .map(|_| text.cited(citation).map(str::to_owned).unwrap_or_default()),
                citation.sources.clone(),
            )
        })
        .collect()
}

fn web(url: &str, title: &str, confidence: f32) -> Source {
    Source::new(SourceLocation::Url {
        url: url.to_owned(),
    })
    .title(title)
    .confidence(confidence)
}

fn expected_grounding() -> Vec<(Option<String>, Vec<Source>)> {
    vec![
        (
            Some("Nico Williams scored first.".to_owned()),
            vec![web("https://example.com/final", "example.com", 0.75)],
        ),
        (
            Some("Oyarzabal won it.".to_owned()),
            vec![
                web("https://example.com/final", "example.com", 0.5),
                web("https://example.org/report", "example.org", 0.25),
            ],
        ),
    ]
}

fn supports(offset: usize) -> Vec<Value> {
    vec![
        support(0, offset, "Nico Williams scored first.", &[0], &[0.75]),
        support(0, offset, "Oyarzabal won it.", &[0, 1], &[0.5, 0.25]),
    ]
}

/// A unary reply's grounding segments resolve to bytes of their part, past
/// a multi-byte dash, with one web source per chunk index and its score.
#[test]
fn a_unary_reply_cites_its_grounding_segments_in_bytes() {
    let reply = finished(chunk(json!([{"text": ANSWER}]), grounding(supports(0))));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    let [text] = texts(&response)[..] else {
        panic!("one text block: {:?}", response.choice);
    };
    assert_eq!(cited(text), expected_grounding());
    let span = text.citations()[0].span.as_ref().expect("a span");
    assert_eq!(span.range(), bytes_of("Nico")..bytes_of("Nico") + 27);
}

/// A segment of the second part counts from that part's start, which is
/// past the first part's bytes in the block the parts share.
#[test]
fn a_segment_of_a_later_part_counts_from_that_part() {
    let split = bytes_of("Nico");
    let parts = json!([{"text": &ANSWER[..split]}, {"text": &ANSWER[split..]}]);
    let supports = vec![
        support(1, split, "Nico Williams scored first.", &[0], &[0.75]),
        support(1, split, "Oyarzabal won it.", &[0, 1], &[0.5, 0.25]),
    ];
    let reply = finished(chunk(parts, grounding(supports)));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    let [text] = texts(&response)[..] else {
        panic!("one text block: {:?}", response.choice);
    };
    assert_eq!(cited(text), expected_grounding());
}

/// A stream sends the grounding with its last chunk, its segments counted
/// over the whole answer though their part index names the chunk's own
/// part. It cites what the unary reply cites, and both fold into the same
/// turn.
#[test]
fn a_streamed_reply_cites_what_the_unary_reply_cites() {
    let unary = finished(chunk(json!([{"text": ANSWER}]), grounding(supports(0))));
    let (first, rest) = ANSWER.split_at(bytes_of("Nico"));
    let (second, last) = rest.split_at(10);
    let streamed = [
        chunk(json!([{"text": first}]), json!({})),
        chunk(json!([{"text": second}]), json!({})),
        finished(chunk(json!([{"text": last}]), grounding(supports(0)))),
    ];
    let response =
        decode(&wire(), Mode::Streaming, streamed.clone().map(frame)).expect("the stream decodes");
    let [text] = texts(&response)[..] else {
        panic!("one text block: {:?}", response.choice);
    };
    assert_eq!(cited(text), expected_grounding());
    assert_restated_agrees(&wire(), [frame(unary)], streamed.map(frame));
}

/// A grounding chunk from retrieval without a URI cites its document, with
/// the retrieved text; a map place cites its URI; a chunk index past the
/// list and an unknown chunk cite nothing.
#[test]
fn every_grounding_chunk_kind_names_its_source() {
    let text = "Nico Williams scored first.";
    let metadata = json!({"groundingMetadata": {
        "groundingChunks": [
            {"retrievedContext": {"documentName": "corpora/c/documents/d", "title": "Report", "text": "Williams scored in the 47th minute."}},
            {"maps": {"uri": "https://maps.example/p", "title": "Olympiastadion", "placeId": "places/p"}},
            {"x_rig_invented": {"uri": "https://example.net"}},
        ],
        "groundingSupports": [support(0, 0, text, &[0, 1, 2, 9], &[])],
    }});
    let reply = finished(chunk(json!([{"text": ANSWER}]), metadata));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    let [block] = texts(&response)[..] else {
        panic!("one text block: {:?}", response.choice);
    };
    let document = Source::new(SourceLocation::Document {
        index: None,
        id: Some("corpora/c/documents/d".to_owned()),
        within: None,
    })
    .title("Report")
    .cited_text("Williams scored in the 47th minute.");
    let place = Source::new(SourceLocation::Url {
        url: "https://maps.example/p".to_owned(),
    })
    .title("Olympiastadion");
    assert_eq!(
        cited(block),
        vec![(Some(text.to_owned()), vec![document, place])]
    );
}

/// A segment whose text the answer does not hold, and one with no source,
/// cite nothing, and the reply still decodes.
#[test]
fn a_segment_that_does_not_place_is_dropped() {
    let mut wrong = support(0, 0, "Oyarzabal won it.", &[0], &[0.5]);
    wrong["segment"]["text"] = json!("Morata won it.");
    let sourceless = support(0, 0, "Nico Williams scored first.", &[], &[]);
    let reply = finished(chunk(
        json!([{"text": ANSWER}]),
        grounding(vec![wrong, sourceless]),
    ));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    assert!(texts(&response)[0].citations().is_empty());
}

/// Thought text is not answer text: a segment counts bytes of the answer
/// part after it, and cites the answer block.
#[test]
fn thought_text_is_not_cited() {
    let parts = json!([{"text": "Spain won, I recall.", "thought": true}, {"text": ANSWER}]);
    let supports = vec![
        support(1, 0, "Nico Williams scored first.", &[0], &[0.75]),
        support(1, 0, "Oyarzabal won it.", &[0, 1], &[0.5, 0.25]),
    ];
    let reply = finished(chunk(parts, grounding(supports)));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    let [text] = texts(&response)[..] else {
        panic!("one text block: {:?}", response.choice);
    };
    assert_eq!(cited(text), expected_grounding());
}

/// The Gemini API's recitation sources count bytes of the answer text,
/// across the blocks a call splits it into; Vertex AI's state no unit and
/// cite the first text block whole.
#[test]
fn recitation_sources_cite_the_answer() {
    let split = bytes_of("Nico");
    let parts = json!([
        {"text": &ANSWER[..split]},
        {"functionCall": {"name": "lookup", "args": {}}},
        {"text": &ANSWER[split..]},
    ]);
    let quoted = "Oyarzabal won it.";
    let metadata = json!({"citationMetadata": {
        "citationSources": [{"startIndex": bytes_of(quoted), "endIndex": bytes_of(quoted) + quoted.len(), "uri": "https://example.com/final", "license": "mit"}],
        "citations": [{"startIndex": 3, "endIndex": 9, "uri": "https://example.org/report", "title": "Report"}],
    }});
    let reply = finished(chunk(parts, metadata));
    let response = decode(&wire(), Mode::Unary, [frame(reply)]).expect("the reply decodes");
    let [head, tail] = texts(&response)[..] else {
        panic!("two text blocks: {:?}", response.choice);
    };
    let report = Source::new(SourceLocation::Url {
        url: "https://example.org/report".to_owned(),
    })
    .title("Report");
    assert_eq!(cited(head), vec![(None, vec![report])]);
    let final_ = Source::new(SourceLocation::Url {
        url: "https://example.com/final".to_owned(),
    });
    assert_eq!(cited(tail), vec![(Some(quoted.to_owned()), vec![final_])]);
}

/// The request body the REST wire sends to continue `response`.
fn replayed(response: &CompletionResponse) -> Value {
    use crate::wire::{Operation, Wire};
    let history = vec![
        crate::message::Message::user("q"),
        response.message().expect("a turn"),
        crate::message::Message::user("n"),
    ];
    let request = crate::operation::Completion::prepare(
        crate::completion::CompletionRequest::from(history),
        &wire().describe(),
    )
    .expect("the history prepares");
    let encoded = wire()
        .encode(request, Mode::Unary)
        .expect("the history encodes");
    crate::test_utils::json_body(&encoded.request)
}

/// Citations never reach the request: a cited turn replays the parts
/// Gemini sent, as the same turn without grounding does.
#[test]
fn a_cited_turn_replays_its_parts_unchanged() {
    let parts = json!([{"text": ANSWER, "thoughtSignature": "c2ln"}]);
    let cited = finished(chunk(parts.clone(), grounding(supports(0))));
    let plain = finished(chunk(parts.clone(), json!({})));
    let cited = decode(&wire(), Mode::Unary, [frame(cited)]).expect("the reply decodes");
    let plain = decode(&wire(), Mode::Unary, [frame(plain)]).expect("the reply decodes");
    assert_eq!(texts(&cited)[0].citations().len(), 2);
    assert_eq!(replayed(&cited), replayed(&plain));
    assert_eq!(replayed(&cited)["contents"][1]["parts"], parts);
}

/// A segment and a recitation source the answer cannot hold: a reversed
/// range, and offsets so large their sums overflow.
fn malformed_ranges(text: &str) -> Vec<(Value, Value)> {
    let huge = u64::MAX - 1;
    vec![
        (
            json!({"partIndex": 0, "startIndex": 10, "endIndex": 5, "text": text}),
            json!({"startIndex": 10, "endIndex": 5, "uri": "https://example.com/final"}),
        ),
        (
            json!({"partIndex": 0, "startIndex": huge, "endIndex": u64::MAX, "text": text}),
            json!({"startIndex": 3, "endIndex": u64::MAX, "uri": "https://example.com/final"}),
        ),
        (
            json!({"partIndex": 0, "startIndex": 3, "endIndex": u64::MAX, "text": text}),
            json!({"startIndex": 0, "endIndex": u64::MAX, "uri": "https://example.com/final"}),
        ),
    ]
}

/// A reversed range or one whose offsets overflow drops its citation; the
/// reply decodes and the rest of the answer stays uncited by it.
#[test]
fn a_reversed_or_huge_range_is_dropped_without_panicking() {
    for (segment, recitation) in malformed_ranges("Nico") {
        for quoted in [true, false] {
            let mut segment = segment.clone();
            if !quoted {
                segment
                    .as_object_mut()
                    .expect("a segment object")
                    .shift_remove("text");
            }
            let metadata = json!({
                "groundingMetadata": {
                    "groundingChunks": [{"web": {"uri": "https://example.com/final"}}],
                    "groundingSupports": [{"segment": segment, "groundingChunkIndices": [0]}],
                },
                "citationMetadata": {"citationSources": [recitation]},
            });
            for mode in [Mode::Unary, Mode::Streaming] {
                let reply = finished(chunk(json!([{"text": ANSWER}]), metadata.clone()));
                let response = decode(&wire(), mode, [frame(reply)]).expect("the reply decodes");
                for citation in texts(&response)[0].citations() {
                    let span = citation.span.as_ref().expect("a span");
                    assert!(span.range().end <= ANSWER.len(), "{segment} {span:?}");
                }
            }
        }
    }
}
