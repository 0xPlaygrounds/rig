//! Shared provider infrastructure: the wire-adapter contract, its
//! single-policy-site driver, and the decode-then-validate classify layer.
//!
//! [`adapter`], [`wire`], [`tool_call_bridge`], and [`chunk_lifecycle`] are
//! public so out-of-tree providers implement [`adapter::WireAdapter`] and
//! inherit the shared driver, frame-triage policy, index→identity tool-call
//! bridging, and the boundary-less reasoning lifecycle derivation instead of
//! hand-rolling per-provider assemblers; the remaining helpers are
//! crate-private.

pub mod adapter;
pub(crate) mod anthropic_compatible;
#[cfg(feature = "audio")]
pub(crate) mod audio_generation;
pub(crate) mod auth;
pub mod chunk_lifecycle;
pub(crate) mod completion_send;
#[cfg(not(target_family = "wasm"))]
pub(crate) mod device_auth;
pub(crate) mod envelope;
#[cfg(feature = "image")]
pub(crate) mod image_generation;
pub(crate) mod model_listing;
pub(crate) mod openai_chat_completions_compatible;
pub(crate) mod schema;
#[cfg(any(test, debug_assertions))]
pub(crate) mod sequence_law;
pub(crate) mod sse_transport;
pub mod tool_call_bridge;
pub(crate) mod transcription;
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
    use std::collections::HashMap;

    let mut names_by_id: HashMap<String, String> = HashMap::new();
    for message in history.iter() {
        let crate::message::Message::Assistant { content, .. } = message else {
            continue;
        };
        for item in content.iter() {
            let crate::message::AssistantContent::ToolCall(call) = item else {
                continue;
            };
            names_by_id.insert(call.id.as_str().to_owned(), call.function.name.clone());
            if let Some(provider) = &call.provider {
                names_by_id.insert(provider.call_id.clone(), call.function.name.clone());
                if let Some(item_id) = &provider.item_id {
                    names_by_id.insert(item_id.clone(), call.function.name.clone());
                }
            }
        }
    }
    if names_by_id.is_empty() {
        return;
    }

    for message in history.iter_mut() {
        let crate::message::Message::User { content } = message else {
            continue;
        };
        for item in content.iter_mut() {
            let crate::message::UserContent::ToolResult(result) = item else {
                continue;
            };
            if !result.name.is_empty() {
                continue;
            }
            let resolved = names_by_id.get(result.call.as_str()).or_else(|| {
                result.provider.as_ref().and_then(|provider| {
                    names_by_id.get(&provider.call_id).or_else(|| {
                        provider
                            .item_id
                            .as_ref()
                            .and_then(|item_id| names_by_id.get(item_id))
                    })
                })
            });
            if let Some(name) = resolved {
                result.name = name.clone();
            }
        }
    }
}

/// A rig logging target for [`trace_json`]. An enum (not a `&str`) because
/// `tracing` targets must be literals, so the dispatch is total by
/// construction.
#[derive(Clone, Copy)]
pub(crate) enum LogTarget {
    Completions,
    Streaming,
}

/// Trace-log `value` as pretty-printed JSON under one of rig's logging
/// targets. Infallible: does nothing when TRACE is disabled for the target or
/// the value fails to serialize.
pub(crate) fn trace_json(target: LogTarget, label: &str, value: &impl serde::Serialize) {
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

pub(crate) fn completion_usage(
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cached_input_tokens: u64,
) -> crate::completion::Usage {
    crate::completion::Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        cache_creation_input_tokens: 0,
        tool_use_prompt_tokens: 0,
        reasoning_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use crate::message::{
        AssistantContent, Message, ToolCall, ToolFunction, ToolResultContent, UserContent,
    };

    fn call(wire_id: &str, name: &str) -> Message {
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
                wire_id,
                ToolFunction {
                    name: name.to_owned(),
                    arguments: serde_json::json!({}),
                },
            ))],
        }
    }

    fn nameless_result(wire_id: &str) -> Message {
        Message::User {
            content: vec![UserContent::tool_result_from_wire(
                wire_id,
                "",
                vec![ToolResultContent::text("out")],
            )],
        }
    }

    fn result_names(history: &[Message]) -> Vec<String> {
        history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => content.iter().next().and_then(|item| match item {
                    UserContent::ToolResult(result) => Some(result.name.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .collect()
    }

    /// An ingested cross-provider transcript (converters stamp `name: ""`)
    /// resolves each result's name from its paired call — by provider id
    /// here, since `from_wire` on both sides shares it.
    #[test]
    fn empty_names_resolve_from_the_paired_call() {
        let mut history = vec![
            call("toolu_1", "get_weather"),
            nameless_result("toolu_1"),
            call("toolu_2", "get_time"),
            nameless_result("toolu_2"),
        ];
        super::resolve_empty_tool_result_names(&mut history);
        assert_eq!(result_names(&history), ["get_weather", "get_time"]);
    }

    /// A result no call in the history answers keeps its empty name: the
    /// transcript genuinely lacks the data, and inventing one would ship a
    /// fabricated name to a name-keyed wire.
    #[test]
    fn an_unmatched_result_keeps_its_empty_name() {
        let mut history = vec![call("toolu_1", "get_weather"), nameless_result("toolu_9")];
        super::resolve_empty_tool_result_names(&mut history);
        assert_eq!(result_names(&history), [""]);
    }

    /// An established name is data, never overwritten — a repair hook may
    /// have renamed the executed tool relative to the model's call.
    #[test]
    fn an_established_name_is_never_overwritten() {
        let mut history = vec![
            call("toolu_1", "add"),
            Message::User {
                content: vec![UserContent::tool_result_from_wire(
                    "toolu_1",
                    "sum",
                    vec![ToolResultContent::text("3")],
                )],
            },
        ];
        super::resolve_empty_tool_result_names(&mut history);
        assert_eq!(result_names(&history), ["sum"]);
    }

    /// Matching falls through the identifier tiers: rig's correlation
    /// handle first (a driver-built result answering an id-less call),
    /// then the provider identifiers.
    #[test]
    fn a_handle_only_result_resolves_from_an_id_less_call() {
        let id_less = ToolCall::new(
            crate::message::ToolCallId::mint(),
            ToolFunction {
                name: "lookup".to_owned(),
                arguments: serde_json::json!({}),
            },
        );
        let handle = id_less.id.as_str().to_owned();
        let mut history = vec![
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::ToolCall(id_less)],
            },
            Message::User {
                content: vec![UserContent::tool_result(
                    handle,
                    "",
                    vec![ToolResultContent::text("out")],
                )],
            },
        ];
        super::resolve_empty_tool_result_names(&mut history);
        assert_eq!(result_names(&history), ["lookup"]);
    }
}
