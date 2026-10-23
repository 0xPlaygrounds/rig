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
pub(crate) mod schema;
/// The frame classifiers every JSON wire's decoder delegates to.
pub mod wire;
pub mod wire_ids;

/// A JSON object written as `json!` writes one, with each value moved in
/// through `Value::from`: `json!` copies every value it interpolates, so a
/// request body built around owned values is built with this instead.
macro_rules! object {
    ($($key:literal: $value:expr),* $(,)?) => {
        serde_json::Value::Object(serde_json::Map::from_iter([
            $(($key.to_owned(), serde_json::Value::from($value))),*
        ]))
    };
}
pub(crate) use object;

/// An object of the `fields` that are present, in order.
pub(crate) fn set_fields<const N: usize>(
    fields: [(&str, Option<serde_json::Value>); N],
) -> serde_json::Map<String, serde_json::Value> {
    fields
        .into_iter()
        .filter_map(|(key, value)| Some((key.to_owned(), value?)))
        .collect()
}

/// Every option refused for `reason`.
pub(crate) fn refuse_all(
    fields: crate::completion::options::OptionFields<'_>,
    reason: &str,
) -> crate::completion::options::OptionMap {
    use crate::completion::options::{Mapping, OptionMap};
    let refuse = |set: bool| match set {
        true => Mapping::unsupported(reason),
        false => Mapping::Nothing,
    };
    OptionMap {
        reasoning: refuse(fields.reasoning.is_some()),
        cache: refuse(fields.cache.is_some()),
        service_tier: refuse(fields.service_tier.is_some()),
        verbosity: refuse(fields.verbosity.is_some()),
        parallel_tool_calls: refuse(fields.parallel_tool_calls.is_some()),
        top_p: refuse(fields.top_p.is_some()),
        seed: refuse(fields.seed.is_some()),
        stop: refuse(!fields.stop.is_empty()),
        cache_key: Mapping::unrouted(fields.cache_key),
    }
}

/// The session id a dialect's session header carries: the request's cache
/// key, so one conversation's requests share it, or a fresh id without one.
pub(crate) fn session_id(cache_key: Option<&str>) -> String {
    cache_key.map_or_else(crate::id::generate, str::to_owned)
}

/// [`session_id`] for `request`. The cache key is a routing hint, which a
/// header carries outside the option mapping.
pub(crate) fn request_session_id(request: &crate::completion::CompletionRequest) -> String {
    session_id(request.options.cache_key.as_deref())
}

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
