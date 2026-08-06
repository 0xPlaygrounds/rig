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

/// Back-compat shim: resolve a tool result's function name from its paired
/// assistant call, for histories that predate [`ToolResult::name`].
///
/// Some wires require the *function name* on a replayed tool result
/// (Gemini's `functionResponse.name`, Ollama's tool-message name). Since
/// review 84a43e9e the durable [`ToolResult`](crate::message::ToolResult)
/// carries it as data (`name`), populated by the agent drivers at
/// construction — a result with `name: Some(..)` is left untouched here and
/// only advances the pairing bookkeeping. The heuristic below runs only for
/// `name: None` results (persisted pre-field histories, hand-built
/// results): pair by identifier when one exists, by wire order when none
/// does, and treat a non-empty id that matches no call as the name itself
/// (the legacy name-in-id encoding). The resolved name is written into
/// `ToolResult::name`; serializers read `name` first and fall back to `id`.
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
                    let call_id = result.call_id.as_deref().filter(|id| !id.is_empty());
                    // A result that already carries its name needs no
                    // heuristic — advance the pairing bookkeeping and move
                    // on, so later name-less results keep pairing in order.
                    if result.name.is_some() {
                        if let Some(index) = pending.iter().position(|(call_identity, _)| {
                            Some(call_identity.as_str()) == call_id
                                || (!result.id.is_empty() && call_identity == &result.id)
                        }) {
                            pending.remove(index);
                        } else {
                            pending.pop_front();
                        }
                        continue;
                    }
                    // Legacy shapes, decided by what `id` holds:
                    //   1. Empty — pair by non-empty `call_id`, else the
                    //      oldest unanswered call (the wire's own in-order
                    //      correlation).
                    //   2. Matching a pending call's identity (directly or
                    //      via `call_id`) — the id is an *identifier*, and
                    //      the match is proof: take that call's name
                    //      (84a43e9e #5 — an OpenAI-Chat `call_abc` replayed
                    //      cross-provider must never become the name).
                    //   3. Non-empty and matching nothing — the id *is* the
                    //      name (the legacy name-in-id encoding, and results
                    //      whose executed tool differs from the model's
                    //      call, e.g. a repair hook renaming it).
                    let identity_match = pending.iter().position(|(call_identity, _)| {
                        Some(call_identity.as_str()) == call_id
                            || (!result.id.is_empty() && call_identity == &result.id)
                    });
                    let matched = identity_match.or_else(|| {
                        // A non-matching identifier is a veto, never a
                        // license to pair positionally.
                        (result.id.is_empty() && call_id.is_none() && !pending.is_empty())
                            .then_some(0)
                    });
                    match matched {
                        Some(index) => {
                            if let Some((_, name)) = pending.remove(index) {
                                result.name = Some(name);
                            }
                        }
                        None if !result.id.is_empty() => {
                            // Shape 3: the id is the name.
                            result.name = Some(result.id.clone());
                            pending.pop_front();
                        }
                        None => {}
                    }
                }
            }
            crate::message::Message::System { .. } => {}
        }
    }
}

#[cfg(test)]
mod resolve_tool_result_names_tests {
    use super::resolve_tool_result_names;
    use crate::message::{
        AssistantContent, Message, ToolCall, ToolFunction, ToolResult, ToolResultContent,
        UserContent,
    };
    use crate::one_or_many::OneOrMany;

    fn call(id: &str, call_id: Option<&str>, name: &str) -> Message {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall {
                id: id.to_owned(),
                call_id: call_id.map(str::to_owned),
                function: ToolFunction {
                    name: name.to_owned(),
                    arguments: serde_json::json!({}),
                },
                signature: None,
                additional_params: None,
            })),
        }
    }

    fn result(id: &str, call_id: Option<&str>, name: Option<&str>) -> Message {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: id.to_owned(),
                call_id: call_id.map(str::to_owned),
                name: name.map(str::to_owned),
                content: OneOrMany::one(ToolResultContent::text("out")),
            })),
        }
    }

    fn resolved_name(history: &[Message], index: usize) -> Option<String> {
        let Message::User { content } = &history[index] else {
            panic!("expected user message");
        };
        let UserContent::ToolResult(result) = content.first() else {
            panic!("expected tool result");
        };
        result.name.clone()
    }

    /// A result that already carries its name is data, not a candidate for
    /// the heuristic: the executed tool's name wins, even when the paired
    /// call's name differs (a repair hook renamed the call).
    #[test]
    fn an_existing_name_is_never_overwritten() {
        let mut history = vec![
            call("call_1", None, "model_named_tool"),
            result("call_1", None, Some("executed_tool")),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("executed_tool".into()));
    }

    /// 84a43e9e finding #5, pinned: an OpenAI-Chat-shaped history
    /// (`ToolResult { id: "call_abc" }`) replayed cross-provider. The id
    /// matches the call's identity — proof it is an identifier — so the
    /// resolved name is the call's function name, never `call_abc`.
    #[test]
    fn an_identifier_matching_a_call_resolves_to_that_calls_name() {
        let mut history = vec![
            call("call_abc", None, "get_weather"),
            result("call_abc", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("get_weather".into()));
    }

    /// The legacy name-in-id encoding still resolves: a non-empty id
    /// matching no pending call is the name itself.
    #[test]
    fn a_legacy_name_in_id_history_keeps_the_name() {
        let mut history = vec![
            call("", None, "get_weather"),
            result("get_weather", None, None),
        ];
        resolve_tool_result_names(&mut history);
        // "get_weather" matches the... call identity is "" here, so the id
        // matches nothing and is kept as the name.
        assert_eq!(resolved_name(&history, 1), Some("get_weather".into()));
    }

    /// Id-less results pair with unanswered calls in wire order.
    #[test]
    fn id_less_results_pair_in_wire_order() {
        let mut history = vec![
            call("", None, "first_tool"),
            call("", None, "second_tool"),
            result("", None, None),
            result("", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 2), Some("first_tool".into()));
        assert_eq!(resolved_name(&history, 3), Some("second_tool".into()));
    }

    /// A named result still advances the in-order bookkeeping, so a later
    /// name-less result pairs with the RIGHT remaining call.
    #[test]
    fn named_results_advance_the_pairing_bookkeeping() {
        let mut history = vec![
            call("", None, "first_tool"),
            call("", None, "second_tool"),
            result("", None, Some("first_tool_executed")),
            result("", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 3), Some("second_tool".into()));
    }
}
