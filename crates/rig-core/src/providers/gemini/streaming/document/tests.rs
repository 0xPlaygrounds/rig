//! The GenerateContent fold on hand-built chunks. Unit tests: each pins a
//! joining rule, several of which no recorded pair exercises; the recorded
//! pairs are checked whole in `test_utils::raw_parity`.

use serde_json::{Value, json};

use super::GenerateContentResponse;
use crate::wire::WireFrame;
use crate::wire::document::Reassemble;

/// The document `chunks` add up to.
fn folded(chunks: impl IntoIterator<Item = Value>) -> Value {
    let mut document = GenerateContentResponse::default();
    for chunk in chunks {
        document.absorb(&WireFrame::Text(chunk.to_string()));
    }
    document.finish()
}

/// A chunk of the first candidate holding `parts`.
fn parts(parts: Value) -> Value {
    json!({ "candidates": [{ "content": { "parts": parts, "role": "model" }, "index": 0 }] })
}

/// The parts of the first candidate of `document`.
fn first_parts(document: &Value) -> &Value {
    &document["candidates"][0]["content"]["parts"]
}

#[test]
fn a_text_run_is_one_part_and_its_trailing_signature_joins_it() {
    let document = folded([
        parts(json!([{ "text": "Hel" }])),
        parts(json!([{ "text": "lo" }])),
        parts(json!([{ "text": "", "thoughtSignature": "sig" }])),
    ]);
    assert_eq!(
        *first_parts(&document),
        json!([{ "text": "Hello", "thoughtSignature": "sig" }])
    );
}

#[test]
fn thought_and_answer_are_separate_parts() {
    let document = folded([
        parts(json!([{ "text": "Let me ", "thought": true }])),
        parts(json!([{ "text": "think.", "thought": true }])),
        parts(json!([{ "text": "Red", "thoughtSignature": "sig" }])),
        parts(json!([{ "text": "." }])),
    ]);
    assert_eq!(
        *first_parts(&document),
        json!([
            { "text": "Let me think.", "thought": true },
            { "text": "Red.", "thoughtSignature": "sig" },
        ])
    );
}

#[test]
fn a_second_signature_starts_a_part_of_its_own() {
    let document = folded([
        parts(json!([{ "text": "a", "thoughtSignature": "one" }])),
        parts(json!([{ "text": "b", "thoughtSignature": "two" }])),
    ]);
    assert_eq!(
        *first_parts(&document),
        json!([
            { "text": "a", "thoughtSignature": "one" },
            { "text": "b", "thoughtSignature": "two" },
        ])
    );
}

#[test]
fn an_empty_unsigned_text_part_carries_nothing() {
    let call =
        json!({ "functionCall": { "name": "add", "args": { "x": 1 } }, "thoughtSignature": "sig" });
    let mut last = parts(json!([{ "text": "" }]));
    last["candidates"][0]["finishReason"] = json!("STOP");
    let document = folded([parts(json!([call.clone()])), last]);
    assert_eq!(*first_parts(&document), json!([call]));
    assert_eq!(document["candidates"][0]["finishReason"], "STOP");
}

#[test]
fn a_bare_signature_joins_the_part_before_it_or_the_next_one() {
    let call = json!({ "functionCall": { "name": "add" } });
    let after = folded([
        parts(json!([call.clone()])),
        parts(json!([{ "thoughtSignature": "sig" }])),
    ]);
    assert_eq!(
        *first_parts(&after),
        json!([{ "functionCall": { "name": "add" }, "thoughtSignature": "sig" }])
    );
    let before = folded([
        parts(json!([{ "thoughtSignature": "sig" }])),
        parts(json!([call])),
    ]);
    assert_eq!(*first_parts(&after), *first_parts(&before));
}

#[test]
fn candidates_are_kept_by_index() {
    let chunk = |index: u64, text: &str| json!({ "candidates": [{ "content": { "parts": [{ "text": text }] }, "index": index }] });
    let document = folded([chunk(1, "b"), chunk(0, "a"), chunk(1, "c")]);
    assert_eq!(
        document["candidates"],
        json!([
            { "content": { "parts": [{ "text": "bc" }] }, "index": 1 },
            { "content": { "parts": [{ "text": "a" }] }, "index": 0 },
        ])
    );
}

#[test]
fn citations_append_and_metadata_keeps_its_last_value() {
    let chunk = |source: &str, tokens: u64| {
        json!({
            "candidates": [{
                "citationMetadata": { "citationSources": [{ "uri": source }] },
                "finishMessage": null,
                "index": 0,
            }],
            "usageMetadata": { "totalTokenCount": tokens },
            "modelVersion": null,
        })
    };
    let mut first = chunk("https://a.example", 3);
    first["modelVersion"] = json!("gemini-2.5-flash");
    first["promptFeedback"] = json!({ "safetyRatings": [] });
    let mut second = chunk("https://b.example", 5);
    second["promptFeedback"] = json!({ "blockReason": "OTHER" });
    let document = folded([first, second]);
    assert_eq!(
        document,
        json!({
            "candidates": [{
                "citationMetadata": { "citationSources": [
                    { "uri": "https://a.example" },
                    { "uri": "https://b.example" },
                ] },
                "finishMessage": null,
                "index": 0,
            }],
            "usageMetadata": { "totalTokenCount": 5 },
            "modelVersion": "gemini-2.5-flash",
            "promptFeedback": { "safetyRatings": [] },
        })
    );
}

#[test]
fn a_reply_with_no_chunk_has_no_document() {
    assert_eq!(folded([]), Value::Null);
    let mut document = GenerateContentResponse::default();
    document.absorb(&WireFrame::Text("not json".to_owned()));
    assert_eq!(document.finish(), Value::Null);
}
