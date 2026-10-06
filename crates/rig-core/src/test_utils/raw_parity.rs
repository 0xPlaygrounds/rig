//! Whole-document parity of a reply's `raw`: the unary body of a turn and
//! the document a stream of the same turn rebuilds, decoded as live calls
//! decode them and compared after [`comparable`].
//!
//! ```no_run
//! # async fn example() -> Result<(), rig_core::error::ProviderError> {
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::test_utils::raw_parity::assert_raw_parity;
//!
//! let wire = OpenAIConfig::new("sk-test").chat("gpt-4.1-nano");
//! let unary = r#"{"id":"a","object":"chat.completion","choices":[]}"#;
//! let streamed = "data: {\"id\":\"b\",\"object\":\"chat.completion.chunk\",\"choices\":[]}\n\n";
//! assert_raw_parity(wire, unary, streamed, &[]).await?;
//! # Ok(())
//! # }
//! ```

use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::completion::CompletionRequest;
use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::Wire;

/// Object keys every recording mints for itself, at any depth: reply, item
/// and call ids, and timestamps.
pub const MINTED_KEYS: &[&str] = &["id", "created", "created_at", "completed_at"];

/// `document` as parity compares it: `null`s, empty lists and empty
/// objects, which a unary body states and a stream may omit, are dropped,
/// as are [`MINTED_KEYS`] and the values at the JSON pointers in `minted`,
/// which the two recordings of a turn cannot share.
pub fn comparable(document: &Value, minted: &[&str]) -> Value {
    let mut document = document.clone();
    for pointer in minted {
        remove(&mut document, pointer);
    }
    pruned(&document).unwrap_or(Value::Null)
}

/// Remove the value at `pointer`, when there is one.
fn remove(document: &mut Value, pointer: &str) {
    let Some((parent, key)) = pointer.rsplit_once('/') else {
        return;
    };
    let key = key.replace("~1", "/").replace("~0", "~");
    match document.pointer_mut(parent) {
        Some(Value::Object(map)) => {
            map.shift_remove(&key);
        }
        Some(Value::Array(items)) => {
            if let Ok(index) = key.parse::<usize>()
                && index < items.len()
            {
                items.remove(index);
            }
        }
        _ => {}
    }
}

/// `value` without empties and minted keys; `None` when nothing is left.
fn pruned(value: &Value) -> Option<Value> {
    match value {
        Value::Null => None,
        Value::Object(map) => {
            let map: serde_json::Map<String, Value> = map
                .iter()
                .filter(|(key, _)| !MINTED_KEYS.contains(&key.as_str()))
                .filter_map(|(key, value)| Some((key.clone(), pruned(value)?)))
                .collect();
            (!map.is_empty()).then_some(Value::Object(map))
        }
        Value::Array(items) => {
            let items: Vec<Value> = items
                .iter()
                .map(|item| pruned(item).unwrap_or(Value::Null))
                .collect();
            (!items.is_empty()).then_some(Value::Array(items))
        }
        other => Some(other.clone()),
    }
}

/// The `raw` of the unary reply `unary` and of the streamed reply
/// `streamed` on `wire`: each body served as the provider sent it, through
/// the driver, the transport and the wire's decoder and reassembler.
///
/// # Errors
///
/// When either reply fails to decode or does not end.
pub async fn raw_pair<W>(
    wire: W,
    unary: impl Into<Bytes>,
    streamed: impl Into<Bytes>,
) -> Result<(Value, Value), ProviderError>
where
    W: Wire<Op = Completion>,
    RecordingHttpClient: Transport<W>,
    MockStreamingClient: Transport<W>,
{
    let streamed = MockStreamingClient {
        sse_bytes: streamed.into(),
    };
    raw_pair_over(wire, RecordingHttpClient::new(unary.into()), streamed).await
}

/// [`raw_pair`] for a wire whose transport is not HTTP: the `raw` of one
/// unary reply over `unary` and one streamed reply over `streamed`, each a
/// transport that serves the recorded reply.
///
/// # Errors
///
/// When either reply fails to decode or does not end.
pub async fn raw_pair_over<W, U, S>(
    wire: W,
    unary: U,
    streamed: S,
) -> Result<(Value, Value), ProviderError>
where
    W: Wire<Op = Completion>,
    U: Transport<W>,
    S: Transport<W>,
{
    let request = || CompletionRequest::new("parity");
    let unary = Model::new(wire.clone(), unary).call(request()).await?.raw;
    let mut stream = Model::new(wire, streamed).stream(request())?;
    while stream.next().await.is_some() {}
    let streamed = stream.finish().await?.raw;
    Ok((unary, streamed))
}

/// The reply body of interaction `interaction` (from 1) of a cassette whose
/// YAML text is `cassette`: a single-quoted or double-quoted scalar, a
/// literal block for an event stream, or base64 when the cassette says so.
/// Read without a YAML dependency, for the shapes the recorder writes.
pub fn recorded_reply(cassette: &str, interaction: usize) -> Option<Bytes> {
    let recorded = interactions(cassette).nth(interaction.checked_sub(1)?)?;
    let (_, reply) = recorded.split_once("\nthen:\n")?;
    let body = scalar(reply, "body")?;
    let base64 = reply
        .lines()
        .any(|line| line.trim() == "body_encoding: base64");
    if base64 {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(body.trim())
            .ok()?;
        return Some(Bytes::from(bytes));
    }
    Some(Bytes::from(body))
}

/// Each interaction of a cassette's YAML text, every line, its last
/// included, ending in a newline.
pub(crate) fn interactions(cassette: &str) -> impl Iterator<Item = &str> {
    let mut rest = Some(cassette);
    std::iter::from_fn(move || {
        let text = rest?;
        match text.find("\n---\n") {
            Some(at) => {
                rest = text.get(at + 5..);
                text.get(..=at)
            }
            None => {
                rest = None;
                Some(text)
            }
        }
    })
}

/// The scalar at `key` among the two-space-indented fields of one section
/// of a recorded interaction.
pub(crate) fn scalar(section: &str, key: &str) -> Option<String> {
    let prefix = format!("  {key}:");
    let mut lines = section.lines();
    let value = lines
        .by_ref()
        .find_map(|line| line.strip_prefix(prefix.as_str()))?
        .trim();
    if let Some(quoted) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
        return Some(quoted.replace("''", "'"));
    }
    // A double-quoted scalar escapes as JSON does for what the recorder
    // writes.
    if value.starts_with('"') {
        return serde_json::from_str(value).ok();
    }
    if !value.starts_with('|') {
        return Some(value.to_owned());
    }
    let mut block = String::new();
    for line in lines {
        if !line.is_empty() && !line.starts_with("    ") {
            break;
        }
        block.push_str(line.get(4..).unwrap_or_default());
        block.push('\n');
    }
    Some(block)
}

/// Check that a streamed reply's `raw` is the unary document of the same
/// turn: [`raw_pair`], each side [`comparable`] with `minted`.
///
/// # Errors
///
/// When either reply fails to decode.
///
/// # Panics
///
/// When the two documents differ, naming both.
pub async fn assert_raw_parity<W>(
    wire: W,
    unary: impl Into<Bytes>,
    streamed: impl Into<Bytes>,
    minted: &[&str],
) -> Result<(), ProviderError>
where
    W: Wire<Op = Completion>,
    RecordingHttpClient: Transport<W>,
    MockStreamingClient: Transport<W>,
{
    let (unary, streamed) = raw_pair(wire, unary, streamed).await?;
    let (unary, streamed) = (comparable(&unary, minted), comparable(&streamed, minted));
    assert_eq!(
        unary, streamed,
        "a streamed reply's raw is the unary document of the same turn"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
