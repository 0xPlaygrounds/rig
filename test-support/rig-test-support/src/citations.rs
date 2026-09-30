//! Assertions on Responses citations: the `output_text` extras a recorded
//! message item states, and the text blocks a route delivered for it.
//!
//! The message item's `phase` is left out of every comparison: the unary
//! route stamps it on the text block, the streamed route does not.

use futures::StreamExt;
use rig_core::message::{AssistantContent, Text};
use rig_core::streaming::{CompletionStream, Item, StreamEvent};
use serde_json::{Map, Value};

/// The key the Responses wire files its own text extras under.
const EXTRAS_KEY: &str = "openai_responses";

/// What a drained stream delivered.
pub struct Drained {
    /// The text blocks its parts ended with, in order.
    pub texts: Vec<Text>,
    /// The `type` of every unmodeled payload it delivered, in order.
    pub unknown_types: Vec<String>,
}

/// Drain `stream`, check its event grammar, and keep its text blocks and
/// unmodeled payload types.
pub async fn drain(mut stream: CompletionStream) -> Drained {
    let mut items = Vec::new();
    let mut texts = Vec::new();
    let mut unknown_types = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("the stream item is ok");
        items.push(Ok(item.clone()));
        match item {
            Item::Event(StreamEvent::End {
                content: AssistantContent::Text(text),
                ..
            }) => texts.push(text),
            Item::Unknown(payload) => unknown_types.push(
                serde_json::to_value(&payload).expect("the payload serializes")["type"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            ),
            _ => {}
        }
    }
    let response = stream.finish().await.expect("the stream ends");
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(
        &items,
        &response.choice,
    );
    Drained {
        texts,
        unknown_types,
    }
}

/// The text blocks of a reply's choice, in order.
pub fn choice_texts(choice: &[AssistantContent]) -> Vec<Text> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// The Responses-owned extras on a text block, without `phase`.
pub fn content_extras(text: &Text) -> Option<Map<String, Value>> {
    let mut extras = text
        .additional_params
        .as_ref()
        .and_then(|params| params.wire_extras(EXTRAS_KEY))
        .cloned()?;
    extras.remove("phase");
    (!extras.is_empty()).then_some(extras)
}

/// The extras a recorded message item's content parts state, in part
/// order: every non-empty sibling of `text` and `type`, arrays concatenated
/// across parts.
pub fn recorded_extras(item: &Value) -> Option<Map<String, Value>> {
    let mut extras = Map::new();
    for part in item["content"].as_array().into_iter().flatten() {
        for (key, value) in part.as_object().into_iter().flatten() {
            let empty = value.is_null()
                || value.as_array().is_some_and(Vec::is_empty)
                || value.as_object().is_some_and(Map::is_empty);
            if key == "text" || key == "type" || empty {
                continue;
            }
            match (extras.get_mut(key), value) {
                (Some(Value::Array(existing)), Value::Array(more)) => {
                    existing.extend(more.iter().cloned());
                }
                _ => {
                    extras.insert(key.clone(), value.clone());
                }
            }
        }
    }
    (!extras.is_empty()).then_some(extras)
}

/// A recorded message item's visible text, its parts joined.
pub fn recorded_text(item: &Value) -> String {
    item["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect()
}

/// The annotations a recorded message item states.
pub fn annotation_count(item: &Value) -> usize {
    item["content"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|part| part["annotations"].as_array().map_or(0, Vec::len))
        .sum()
}

/// The message items among `items` that state text.
pub fn messages<'a>(items: impl IntoIterator<Item = &'a Value>) -> Vec<Value> {
    items
        .into_iter()
        .filter(|item| item["type"] == "message" && !recorded_text(item).is_empty())
        .cloned()
        .collect()
}

/// The message items of a recorded stream's `output_item.done` frames.
pub fn item_snapshots(frames: &[Value]) -> Vec<Value> {
    let items: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["type"] == "response.output_item.done")
        .map(|frame| &frame["item"])
        .collect();
    messages(items)
}

/// The recorded frames of type `kind`.
pub fn frames_of<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
    frames
        .iter()
        .filter(|frame| frame["type"] == kind)
        .collect()
}

/// Each text block with text ends with the text and extras of its recorded
/// message item, in order: one part per item, its extras stated once.
pub fn assert_texts_match_items(route: &str, texts: &[Text], items: &[Value]) {
    let texts: Vec<&Text> = texts.iter().filter(|text| !text.text.is_empty()).collect();
    assert_eq!(
        texts.len(),
        items.len(),
        "{route}: one text part per message item"
    );
    for (text, item) in texts.into_iter().zip(items) {
        assert_eq!(text.text, recorded_text(item), "{route}: the item's text");
        assert_eq!(
            content_extras(text),
            recorded_extras(item),
            "{route}: the item's extras, once"
        );
    }
}
