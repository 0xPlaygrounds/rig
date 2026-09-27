//! Shared frame classification, tool-call identity, and reasoning lifecycle helpers.
//! Companion providers use these helpers from [`Decoder`](crate::wire::Decoder)
//! implementations that emit into [`AdapterOutput`](crate::operation::AdapterOutput).
//!
//! ```
//! use rig_core::providers::internal::resolve_empty_tool_result_names;
//! let mut history = Vec::new();
//! resolve_empty_tool_result_names(&mut history);
//! ```

pub(crate) mod auth;
pub mod chunk_lifecycle;
#[cfg(not(target_family = "wasm"))]
pub(crate) mod device_auth;
pub(crate) mod openai_chat_completions_compatible;
pub(crate) mod schema;
/// The debug-mode sequence-law validator. Public only because it is the
/// completion sink's `Laws` type; its checks run under `debug_assertions`.
#[doc(hidden)]
pub mod sequence_law;
pub mod tool_call_bridge;
pub mod tool_call_ids;
pub mod wire;

/// Fill empty tool-result names from preceding unmatched calls.
/// Match local correlation handles before provider identifiers. Existing names
/// disambiguate multiple matches; ambiguous or missing matches remain unchanged.
/// This helper is available to companion serializers but is not a stable public API.
pub fn resolve_empty_tool_result_names(history: &mut [crate::message::Message]) {
    use crate::message::{AssistantContent, Message, ToolCall, UserContent};

    // IDs are completion-local. Resolve only against preceding outstanding
    // calls, never a future turn that happens to reuse the same generated key.
    let mut pending: Vec<ToolCall> = Vec::new();
    for message in history {
        match message {
            Message::Assistant { content, .. } => {
                pending.extend(content.iter().filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                }));
            }
            Message::User { content } => {
                for item in content {
                    let UserContent::ToolResult(result) = item else {
                        continue;
                    };
                    let local: Vec<_> = pending
                        .iter()
                        .enumerate()
                        .filter(|(_, call)| call.id == result.call)
                        .map(|(index, _)| index)
                        .collect();
                    let mut candidates = if local.is_empty() {
                        // Provider aliases are their own namespace, not strings
                        // inserted alongside generated correlation keys.
                        pending
                            .iter()
                            .enumerate()
                            .filter(|(_, call)| match (&call.provider, &result.provider) {
                                (Some(call), Some(result)) => {
                                    call.call_id == result.call_id
                                        || call.item_id.as_ref().is_some_and(|id| {
                                            id == &result.call_id
                                                || result.item_id.as_ref() == Some(id)
                                        })
                                        || result.item_id.as_ref() == Some(&call.call_id)
                                }
                                _ => false,
                            })
                            .map(|(index, _)| index)
                            .collect()
                    } else {
                        local
                    };
                    if candidates.len() > 1 && !result.name.is_empty() {
                        candidates.retain(|index| {
                            pending
                                .get(*index)
                                .is_some_and(|call| call.function.name == result.name)
                        });
                    }
                    if let [index] = candidates.as_slice() {
                        let call = pending.remove(*index);
                        if result.name.is_empty() {
                            result.name = call.function.name;
                        }
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tool_call_id_tests;

/// Reads the provider's transport request id off a response's headers, when
/// the provider names such a header and the response carries a non-empty
/// value. `None` is the documented "not reported" outcome.
pub(crate) fn request_id_from_headers(
    headers: &http::HeaderMap,
    request_id_header: Option<&str>,
) -> Option<String> {
    request_id_header.and_then(|header| {
        headers
            .get(header)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
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
