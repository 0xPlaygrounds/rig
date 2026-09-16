//! Shared provider infrastructure: the decode-then-validate classify layer
//! and the assembly helpers every completion decoder shares.
//!
//! [`wire`], [`tool_call_bridge`], and [`chunk_lifecycle`] are public so
//! out-of-tree providers implement [`Decoder`](crate::wire::Decoder) and
//! inherit the shared fold ([`WireDriver`](crate::driver::WireDriver)),
//! frame-triage policy, index→identity tool-call bridging, and the
//! boundary-less reasoning lifecycle derivation instead of hand-rolling
//! per-provider assemblers; the remaining helpers are crate-private. The
//! output buffer those decoders write through is the completion operation's
//! own ([`AdapterOutput`](crate::operation::AdapterOutput)).

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

/// Fill empty [`ToolResult::name`](crate::message::ToolResult::name)s from
/// the calls they answer, for wires that key the replay on the tool name
/// (Gemini `functionResponse.name`, Ollama tool messages, Vertex AI,
/// gemini-grpc, Interactions).
///
/// `ToolResult::name` is required data, but rig's own inbound converters
/// cannot supply it: Anthropic, OpenAI-chat, Cohere, and Bedrock tool
/// messages carry no name on their wires, so a cross-provider ingested
/// transcript arrives with `name: ""`. The name lives on the paired
/// assistant call in the same history — match by rig's correlation handle
/// first, then by provider identifiers. A result matching no call keeps
/// its empty name: the transcript genuinely lacks the data, and the wire's
/// own rejection is the honest failure.
///
/// `pub` (not `pub(crate)`) because sibling serializer crates that speak a
/// name-keyed tool-result wire (rig-vertexai, rig-gemini-grpc) carry the
/// same contract; it is not part of rig-core's stable public API.
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

/// `path` with `pairs` appended as a percent-encoded query string.
///
/// Shared by the wires that put a provider-reported cursor or page size in a
/// URI (Anthropic's `Models`, Gemini's `models` and `cachedContents`):
/// concatenating a cursor raw would let a `+`, `&`, `=` or `/` inside it
/// truncate the cursor or inject a query parameter, silently dropping pages.
pub(crate) fn with_query_pairs(path: &str, pairs: &[(&str, &str)]) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (name, value) in pairs {
        serializer.append_pair(name, value);
    }
    format!("{path}?{}", serializer.finish())
}
