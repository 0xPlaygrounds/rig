//! Every recorded whole `generateContent` reply, restated as a stream,
//! folds into the turn the whole reply folds into: one decoder writes both.
//! The restatement sends each part in its own chunk, splits each text part
//! in two, and ends with a chunk holding the candidate's metadata.

use rig::providers::gemini::GeminiConfig;
use rig::providers::gemini::completion::GenerateContent;
use rig::wire::WireFrame;
use rig_core::test_utils::history::{assert_restated_agrees, decode};
use rig_core::wire::Mode;
use serde_json::{Map, Value, json};

use crate::cassettes::{cassette_root, recorded_request_paths, recorded_statuses_and_bodies};

/// `reply` as the stream that would have carried it.
fn restated(reply: &Value) -> Vec<WireFrame> {
    let mut candidate = reply
        .pointer("/candidates/0")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let parts = candidate
        .shift_remove("content")
        .and_then(|content| content.get("parts").cloned())
        .and_then(|parts| parts.as_array().cloned())
        .unwrap_or_default();
    let mut chunks: Vec<Map<String, Value>> = Vec::new();
    for part in parts {
        let Value::Object(mut part) = part else {
            continue;
        };
        if let Some(Value::String(text)) = part.get("text").cloned() {
            let at = text
                .char_indices()
                .nth(text.chars().count() / 2)
                .map_or(text.len(), |(at, _)| at);
            if at > 0 {
                let mut head = Map::new();
                head.insert("text".to_owned(), json!(&text[..at]));
                if let Some(thought) = part.get("thought") {
                    head.insert("thought".to_owned(), thought.clone());
                }
                chunks.push(head);
                part.insert("text".to_owned(), json!(&text[at..]));
            }
        }
        chunks.push(part);
    }
    let envelope = |candidate: Value| {
        let mut chunk = reply.as_object().cloned().unwrap_or_default();
        chunk.shift_remove("usageMetadata");
        chunk.insert("candidates".to_owned(), json!([candidate]));
        chunk
    };
    let mut frames: Vec<Value> = chunks
        .into_iter()
        .map(|part| {
            Value::Object(envelope(json!({
                "content": { "parts": [part], "role": "model" },
            })))
        })
        .collect();
    let mut last = envelope(Value::Object(candidate));
    if let Some(usage) = reply.get("usageMetadata") {
        last.insert("usageMetadata".to_owned(), usage.clone());
    }
    frames.push(Value::Object(last));
    frames
        .iter()
        .map(|frame| WireFrame::Text(frame.to_string()))
        .collect()
}

/// The scenarios under `directory`, relative to the Gemini cassette root.
fn scenarios(directory: &std::path::Path, root: &std::path::Path, found: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scenarios(&path, root, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            let relative = path.strip_prefix(root).expect("under the root");
            found.push(relative.with_extension("").to_string_lossy().into_owned());
        }
    }
}

/// `turn` with each rig-issued call id replaced by its position, and the
/// fingerprints that cover those ids left out.
fn numbered(mut turn: Value) -> Value {
    fn walk(value: &mut Value, next: &mut usize) {
        match value {
            Value::Object(fields) => {
                fields.shift_remove("fingerprint");
                if let Some(local) = fields.get_mut("local") {
                    *local = json!(*next);
                    *next += 1;
                }
                fields.values_mut().for_each(|value| walk(value, next));
            }
            Value::Array(values) => values.iter_mut().for_each(|value| walk(value, next)),
            _ => {}
        }
    }
    walk(&mut turn, &mut 0);
    turn
}

#[test]
fn every_recorded_whole_reply_agrees_with_its_restatement() {
    let root = cassette_root().join("gemini");
    let mut found = Vec::new();
    scenarios(&root, &root, &mut found);
    found.sort();
    let mut restated_replies = 0;
    for scenario in found {
        let paths = recorded_request_paths("gemini", &scenario);
        let replies = recorded_statuses_and_bodies("gemini", &scenario);
        for (path, (status, body)) in paths.iter().zip(replies) {
            let Some(model) = path
                .strip_prefix("/v1beta/models/")
                .and_then(|rest| rest.strip_suffix(":generateContent"))
            else {
                continue;
            };
            let Ok(reply) = serde_json::from_str::<Value>(&body) else {
                continue;
            };
            let wire = GenerateContent::new(GeminiConfig::new("replay"), model);
            let whole = vec![WireFrame::Text(body.clone())];
            let Some(unary) = decode(&wire, Mode::Unary, whole.clone())
                .ok()
                .filter(|_| status == 200)
            else {
                continue;
            };
            if unary.tool_calls().any(|call| call.id.is_local()) {
                // Each decode issues its own ids for id-less calls, and the
                // shared harness compares them; compare with them numbered.
                let streamed = decode(&wire, Mode::Streaming, restated(&reply))
                    .unwrap_or_else(|error| panic!("{scenario}: the restatement decodes: {error}"));
                assert_eq!(
                    numbered(json!(unary.message())),
                    numbered(json!(streamed.message())),
                    "{scenario}"
                );
            } else {
                assert_restated_agrees(&wire, whole, restated(&reply));
            }
            restated_replies += 1;
        }
    }
    assert!(
        restated_replies > 100,
        "the sweep restated {restated_replies} replies"
    );
}
