//! Where an assistant turn came from and the provider items it was decoded
//! from. An [`Origin`] names the wire, provider and requested model of a
//! turn; a [`Native`] holds one provider item verbatim beside the canonical
//! block it decoded to, with a [`Fingerprint`] of that block so an edited
//! block stops replaying its stale item.
//!
//! ```
//! use rig_core::message::{AssistantContent, Text};
//!
//! let block = AssistantContent::Text(Text::new("hi"))
//!     .with_native(serde_json::json!({"type": "text", "text": "hi", "citations": []}));
//! assert!(block.native_item().is_some());
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// The wire format a turn was produced by, for example
/// `"anthropic.messages"`, `"openai.responses"` or `"openai.chat"`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Api(Cow<'static, str>);

impl Api {
    /// The API named `name`, in a const context.
    pub const fn from_static(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    /// The API's name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&'static str> for Api {
    fn from(name: &'static str) -> Self {
        Self::from_static(name)
    }
}

impl From<String> for Api {
    fn from(name: String) -> Self {
        Self(Cow::Owned(name))
    }
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which wire, provider and model produced an assistant turn.
///
/// `model` is the model the request named. Replay compares it, with `api`
/// and `provider`, against the target: only an exact match replays the
/// turn's provider items.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// The wire format.
    pub api: Api,
    /// The provider descriptor name (`"anthropic"`).
    pub provider: String,
    /// The model the request named.
    pub model: String,
    /// The model the provider reported, when it reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// The provider's response id, when it sent one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The fingerprint of the request's tools and system prompt. A wire
    /// that binds its items to them ([`ReplayTarget::binds_context`])
    /// replays a turn made under another context as if from another model.
    ///
    /// [`ReplayTarget::binds_context`]: crate::completion::ReplayTarget::binds_context
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Fingerprint>,
}

impl Origin {
    /// A turn from `model` on `provider` over `api`, with no response
    /// metadata.
    pub fn new(api: impl Into<Api>, provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            response_model: None,
            response_id: None,
            context: None,
        }
    }

    /// Whether this turn came from exactly the wire, provider and model
    /// `target` names. The one sameness rule replay uses.
    pub fn same_model(&self, api: &Api, provider: &str, model: &str) -> bool {
        &self.api == api && self.provider == provider && self.model == model
    }
}

/// How an assistant turn ended.
///
/// A turn that failed ([`Self::fails`]) is kept in history but never
/// replayed to a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished.
    Stop,
    /// The output-token limit cut the turn.
    Length,
    /// The model stopped to call tools.
    ToolUse,
    /// The provider failed the turn or refused it, with its explanation.
    Error(String),
    /// The caller cancelled the turn.
    Aborted(String),
}

impl StopReason {
    /// Whether the turn is incomplete and must not be replayed.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Error(_) | Self::Aborted(_))
    }

    /// Whether a turn that ended so failed: it is never replayed, and what
    /// it answered is still owed. That is a failure ([`Self::is_failure`]),
    /// or the output limit cutting a turn that `answered` nothing.
    pub fn fails(&self, answered: bool) -> bool {
        self.is_failure() || (*self == Self::Length && !answered)
    }
}

/// A 64-bit FNV-1a hash of a value's JSON serialization with every object's
/// keys in sorted order and every whole number written as an integer, so a
/// store that reorders keys (Postgres `jsonb`, sorted-key dumps) or writes
/// `20.0` as `20` never changes it. It is stored as 16 hex digits, since a
/// store that reads JSON numbers as doubles would round a 64-bit number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint(u64);

impl Serialize for Fingerprint {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:016x}", self.0))
    }
}

impl<'de> Deserialize<'de> for Fingerprint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = std::borrow::Cow::<'de, str>::deserialize(deserializer)?;
        u64::from_str_radix(&text, 16)
            .map(Self)
            .map_err(|_| serde::de::Error::custom("a fingerprint is 16 hex digits"))
    }
}

impl Fingerprint {
    /// The fingerprint of `value`'s JSON bytes, keys sorted and whole
    /// numbers as integers.
    pub fn of(value: &impl Serialize) -> Self {
        let mut hash = Fnv(0xcbf2_9ce4_8422_2325);
        // Message types always serialize; a failure would only leave a
        // fingerprint no native item matches, which replays canonically.
        if let Ok(value) = serde_json::to_value(value) {
            let _ = serde_json::to_writer(&mut hash, &Canonical(&value));
        }
        Self(hash.0)
    }

    /// The fingerprint of `value` hashed exactly as it serializes: every
    /// JSON object inside must already be canonical (wrapped in
    /// [`Canonical`], or with sorted keys and no floats).
    pub(crate) fn of_canonical(value: &impl Serialize) -> Self {
        let mut hash = Fnv(0xcbf2_9ce4_8422_2325);
        let _ = serde_json::to_writer(&mut hash, value);
        Self(hash.0)
    }
}

/// FNV-1a over the bytes written to it.
struct Fnv(u64);

impl std::io::Write for Fnv {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A JSON value (or object) that serializes canonically: object keys
/// sorted, and whole floats as integers (through `format!("{float:.0}")`
/// parsed back, so `1e20` and `-0.0` come out as that parse writes them).
pub(crate) struct Canonical<'a, T: ?Sized>(pub(crate) &'a T);

impl Serialize for Canonical<'_, serde_json::Map<String, serde_json::Value>> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut fields: Vec<_> = self.0.iter().collect();
        fields.sort_by_key(|(left, _)| *left);
        serializer.collect_map(
            fields
                .into_iter()
                .map(|(key, value)| (key, Canonical(value))),
        )
    }
}

impl Serialize for Canonical<'_, serde_json::Value> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::Object(fields) => Canonical(fields).serialize(serializer),
            serde_json::Value::Array(values) => {
                serializer.collect_seq(values.iter().map(Canonical))
            }
            serde_json::Value::Number(number) => match number
                .as_f64()
                .filter(|float| number.is_f64() && float.fract() == 0.0)
                .and_then(|float| format!("{float:.0}").parse::<serde_json::Number>().ok())
            {
                Some(whole) => whole.serialize(serializer),
                None => number.serialize(serializer),
            },
            value => value.serialize(serializer),
        }
    }
}

/// One provider item, verbatim in its API's JSON shape, beside the
/// canonical block (or message) it was decoded to.
///
/// `fingerprint` is the canonical form's at decode time. When the block is
/// edited its fingerprint changes, the item is stale, and encoders rebuild
/// the wire item from the canonical fields instead.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Native {
    /// The provider item.
    pub item: serde_json::Value,
    /// The fingerprint of the canonical form the item was decoded to.
    pub fingerprint: Fingerprint,
}

/// Where each wire keeps the opaque part of a reasoning item (a signature
/// or an encrypted copy of the reasoning), as a JSON pointer into the item,
/// by the [`Api`] that made it: Anthropic's `thinking.signature` and
/// `redacted_thinking.data`, Bedrock Converse's `reasoningText.signature`
/// and `redactedContent`, Responses' `reasoning.encrypted_content`, and
/// Gemini's `thoughtSignature` (on any part, a call's included).
const REASONING_BLOBS: [(&str, &str); 7] = [
    ("anthropic.messages", "/signature"),
    ("anthropic.messages", "/data"),
    (
        "bedrock.converse",
        "/reasoningContent/reasoningText/signature",
    ),
    ("bedrock.converse", "/reasoningContent/redactedContent"),
    ("openai.responses", "/encrypted_content"),
    ("gemini.generate_content", "/thoughtSignature"),
    ("vertexai.generate_content", "/thoughtSignature"),
];

/// The opaque part of a provider's reasoning item, as the wire that made it
/// spells it ([`Native::reasoning_blob`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blob<'a> {
    /// Where it sits in the item, as a JSON pointer such as `/signature`.
    pub key: &'static str,
    /// The blob, as the provider sent it.
    pub value: &'a str,
}

impl Native {
    /// The opaque part of this item when `api`, the wire that made it, keeps
    /// one there: a reasoning signature or encrypted reasoning, which only
    /// that provider can read and which it may refuse once anything around
    /// it changed. `None` for an item with none, or a wire rig does not know
    /// blobs of.
    ///
    /// ```
    /// use rig_core::message::{Api, Fingerprint, Native};
    ///
    /// let native = Native {
    ///     item: serde_json::json!({"type": "thinking", "thinking": "plan", "signature": "c2ln"}),
    ///     fingerprint: Fingerprint::of(&"plan"),
    /// };
    /// let api = Api::from_static("anthropic.messages");
    /// assert_eq!(native.reasoning_blob(&api).map(|blob| blob.value), Some("c2ln"));
    /// ```
    pub fn reasoning_blob(&self, api: &Api) -> Option<Blob<'_>> {
        REASONING_BLOBS
            .iter()
            .filter(|(wire, _)| *wire == api.as_str())
            .find_map(|(_, key)| {
                let value = self.item.pointer(key)?.as_str()?;
                (!value.is_empty()).then_some(Blob { key, value })
            })
    }
}

/// A block's stored `native`, or `None` when it cannot be read, such as one
/// fingerprinted by an earlier projection: the block then loads with its
/// canonical fields and replays from them.
pub(crate) fn lenient<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Native>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| Native::deserialize(value).ok()))
}

/// A provider item with no canonical meaning, such as a hosted-tool step or
/// a compaction record. Only the API that produced it reads it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Opaque {
    /// The provider item.
    pub item: serde_json::Value,
    /// Whether the item goes back to the model that produced it. A
    /// client-executed call nothing answers is kept but never sent.
    pub replay: bool,
}

impl Opaque {
    /// The item's `type` field, when it is an object that has one.
    pub fn kind(&self) -> Option<&str> {
        self.item.get("type").and_then(serde_json::Value::as_str)
    }
}

#[cfg(test)]
mod tests;
