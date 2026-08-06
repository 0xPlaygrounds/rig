//! Shared provider infrastructure: the wire-adapter contract, its
//! single-policy-site driver, and the decode-then-validate classify layer.
//!
//! [`adapter`], [`wire`], and [`tool_call_bridge`] are public so out-of-tree
//! providers implement [`adapter::WireAdapter`] and inherit the shared driver,
//! frame-triage policy, and index→identity tool-call bridging instead of
//! hand-rolling per-provider assemblers; the remaining helpers are
//! crate-private.

pub mod adapter;
pub(crate) mod auth;
pub(crate) mod openai_chat_completions_compatible;
pub mod tool_call_bridge;
pub mod wire;

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

/// Resolve each tool result's function name from its paired assistant call.
///
/// Some wires require the *function name* on a replayed tool result
/// (Gemini's `functionResponse.name`, Ollama's tool-message name), which
/// rig's durable `ToolResult` does not carry as a field. Gemini's
/// `functionResponse.name` is the *function name*, which rig's
/// durable `ToolResult` does not carry as a field — and rig no longer
/// smuggles it through the tool-call id (two calls to the same tool must
/// stay distinct; identity is not a name). The pairing is recovered from the
/// history itself: a result matches a pending call by a non-empty
/// `call_id`/`id`, else the oldest unanswered call in order — the wire's own
/// correlation model. A result with no pending call keeps its `id` as the
/// name, which is what legacy histories stored there.
///
/// The resolved name is written into `ToolResult::id`, the field the
/// `FunctionResponse` conversion reads as the name.
pub(crate) fn resolve_tool_result_names(history: &mut [crate::message::Message]) {
    let mut pending: std::collections::VecDeque<(String, String)> =
        std::collections::VecDeque::new();
    for msg in history {
        match msg {
            crate::message::Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let crate::message::AssistantContent::ToolCall(call) = item {
                        let identity = call
                            .call_id
                            .clone()
                            .filter(|id| !id.is_empty())
                            .or_else(|| Some(call.id.clone()).filter(|id| !id.is_empty()))
                            .unwrap_or_default();
                        pending.push_back((identity, call.function.name.clone()));
                    }
                }
            }
            crate::message::Message::User { content } => {
                for item in content.iter_mut() {
                    let crate::message::UserContent::ToolResult(result) = item else {
                        continue;
                    };
                    // What does this result's `id` hold? Three shapes exist
                    // in the wild:
                    //   1. Empty — the id-less wire; rig fabricates nothing.
                    //      The name comes from the paired call: by non-empty
                    //      `call_id`, else the oldest unanswered call (the
                    //      wire's own in-order correlation).
                    //   2. Equal to `call_id` — a wire identifier copied onto
                    //      both fields. The name comes from the call bearing
                    //      that identifier.
                    //   3. Anything else — the id *is* the function name
                    //      (legacy name-in-id histories; results whose
                    //      executed tool differs from the model's call, e.g.
                    //      a repair hook renaming it). Kept verbatim.
                    let call_id = result.call_id.as_deref().filter(|id| !id.is_empty());
                    let id_is_identifier = call_id.is_some_and(|call_id| result.id == call_id);
                    if !result.id.is_empty() && !id_is_identifier {
                        // Shape 3: advance the in-order bookkeeping past the
                        // answered call so later results keep pairing.
                        if let Some(index) = pending.iter().position(|(call_identity, _)| {
                            Some(call_identity.as_str()) == call_id || call_identity == &result.id
                        }) {
                            pending.remove(index);
                        } else {
                            pending.pop_front();
                        }
                        continue;
                    }
                    // Shapes 1 and 2: resolve the name from the paired call.
                    let matched = match call_id {
                        Some(identity) => pending
                            .iter()
                            .position(|(call_identity, _)| call_identity == identity),
                        // A non-matching identifier is a veto, never a
                        // license to pair positionally.
                        None if result.id.is_empty() => (!pending.is_empty()).then_some(0),
                        None => None,
                    };
                    if let Some(index) = matched
                        && let Some((_, name)) = pending.remove(index)
                    {
                        result.id = name;
                    }
                }
            }
            crate::message::Message::System { .. } => {}
        }
    }
}
