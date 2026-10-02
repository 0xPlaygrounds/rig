//! Shared frame classification and tool-call id spelling. Companion providers
//! use these helpers from their [`Decoder`](crate::wire::Decoder)s and
//! encoders.
//!
//! ```
//! use rig_core::providers::internal::wire::classify_untyped_line;
//! use rig_core::wire::WireEvent;
//!
//! let event: WireEvent<serde_json::Value> = classify_untyped_line(br#"{"done":true}"#);
//! assert!(matches!(event, WireEvent::Known(_)));
//! ```

pub(crate) mod auth;
#[cfg(not(target_family = "wasm"))]
pub(crate) mod device_auth;
pub(crate) mod openai_chat_completions_compatible;
pub(crate) mod schema;
/// The debug-mode sequence-law validator the completion fold checks a
/// decoder's output against; its checks run under `debug_assertions`.
pub mod wire;
pub mod wire_ids;

/// A rig logging target for [`trace_json`]. An enum (not a `&str`) because
/// `tracing` targets must be literals, so the dispatch is total by
/// construction.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub enum LogTarget {
    Completions,
    Streaming,
}

/// Trace-log `value` as pretty-printed JSON under one of rig's logging
/// targets. Infallible: does nothing when TRACE is disabled for the target or
/// the value fails to serialize.
#[doc(hidden)]
pub fn trace_json(target: LogTarget, label: &str, value: &impl serde::Serialize) {
    macro_rules! emit {
        ($target:literal) => {
            if tracing::enabled!(target: $target, tracing::Level::TRACE) {
                if let Ok(json) = serde_json::to_string_pretty(value) {
                    tracing::trace!(target: $target, "{label}: {json}");
                }
            }
        };
    }
    match target {
        LogTarget::Streaming => emit!("rig::streaming"),
        LogTarget::Completions => emit!("rig::completions"),
    }
}

/// Serde for a dialect that is a registered `const`: the name is the whole
/// wire format, so a host storing a wire cannot reconstitute a dialect
/// with somebody else's base URL, and an unknown name is an error rather
/// than a silent default.
pub(crate) mod named_dialect {
    /// Write the dialect's name, refusing a value that is not the registered
    /// constant of that name: writing only its name would silently lose
    /// its payload.
    pub(crate) fn serialize<S: serde::Serializer>(
        serializer: S,
        family: &str,
        name: &str,
        registered: bool,
    ) -> Result<S::Ok, S::Error> {
        if !registered {
            return Err(serde::ser::Error::custom(format!(
                "an unregistered or modified {family} dialect cannot be persisted by name; \
                 use configuration overrides"
            )));
        }
        serializer.serialize_str(name)
    }

    /// Read a name and look the dialect up among the family's constants.
    pub(crate) fn deserialize<'de, D, T>(
        deserializer: D,
        family: &str,
        by_name: impl FnOnce(&str) -> Option<T>,
    ) -> Result<T, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = <String as serde::Deserialize>::deserialize(deserializer)?;
        by_name(&name).ok_or_else(|| {
            serde::de::Error::custom(format!("`{name}` is not a registered {family} dialect"))
        })
    }
}

/// Append `pairs` to `path` as a percent-encoded query string.
/// Encoding preserves cursor delimiters and prevents query-parameter injection.
pub(crate) fn with_query_pairs(path: &str, pairs: &[(&str, &str)]) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (name, value) in pairs {
        serializer.append_pair(name, value);
    }
    format!("{path}?{}", serializer.finish())
}

/// The decoder and encode error of a completion wire whose family has not
/// moved to item-shaped history yet: every request and reply fails. The
/// family migration replaces it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unmigrated;

impl Unmigrated {
    const MESSAGE: &'static str = "this provider has not moved to item-shaped history yet";

    /// The error every request to an unmigrated wire fails with.
    pub fn encode_error() -> crate::error::EncodeError {
        crate::error::EncodeError::request(Self::MESSAGE)
    }
}

impl<'id, F> crate::wire::Decoder<'id, crate::operation::Completion, F> for Unmigrated {
    type Event = ();

    fn classify(&self, _frame: F) -> crate::wire::WireEvent<()> {
        crate::wire::WireEvent::Known(())
    }

    fn decode(
        &mut self,
        _event: (),
        _out: crate::wire::Out<'id, crate::operation::Completion>,
    ) -> Result<crate::wire::Flow, crate::error::ProviderError> {
        Err(crate::error::ProviderError::Response(
            Self::MESSAGE.to_owned(),
        ))
    }
}
