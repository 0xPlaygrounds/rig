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
pub(crate) mod chunk_lifecycle;
pub(crate) mod openai_chat_completions_compatible;
#[cfg(any(test, debug_assertions))]
pub(crate) mod sequence_law;
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
/// assistant call, for histories that predate [`ToolResult::name`](crate::message::ToolResult::name).
///
/// Some wires require the *function name* on a replayed tool result
/// (Gemini's `functionResponse.name`, Ollama's tool-message name). Since
/// review 84a43e9e the durable [`ToolResult`](crate::message::ToolResult)
/// carries it as data (`name`), populated by the agent drivers at
/// construction — a result with `name: Some(..)` is left untouched here and
/// only advances the pairing bookkeeping. The heuristic below runs only for
/// `name: None` results (persisted pre-field histories, hand-built
/// results): pair by identifier when one exists, else by tool name, else
/// by wire order, and treat a non-empty id that matches no call as the
/// name itself (the legacy name-in-id encoding). The name tier is
/// legacy-encoding support, not general matching policy — id-keyed SDKs
/// never fall back to names, and neither does any result that carries a
/// usable identifier. The resolved name is written into
/// `ToolResult::name`; serializers read `name` first and fall back to `id`.
///
/// `pub` (not `pub(crate)`) because sibling serializer crates that speak a
/// name-keyed tool-result wire (rig-vertexai's `functionResponse.name`)
/// carry the same contract; it is not part of rig-core's stable public API.
pub fn resolve_tool_result_names(history: &mut [crate::message::Message]) {
    /// A call's identifiers are a *set*: wires like OpenAI Responses issue
    /// both an item id (`fc_…`, [`ToolCall::id`](crate::message::ToolCall::id))
    /// and a `call_id` (`call_…`), and a legacy result may mirror either or
    /// both. Any id-vs-name decision below must test against the whole set —
    /// an id that equals *either* identifier is an identifier, never a name.
    struct PendingCall {
        call_id: Option<String>,
        item_id: Option<String>,
        name: String,
    }
    impl PendingCall {
        fn matches(&self, candidate: &str) -> bool {
            self.call_id.as_deref() == Some(candidate) || self.item_id.as_deref() == Some(candidate)
        }
    }
    let mut pending: std::collections::VecDeque<PendingCall> = std::collections::VecDeque::new();
    for msg in history {
        match msg {
            crate::message::Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let crate::message::AssistantContent::ToolCall(call) = item {
                        pending.push_back(PendingCall {
                            call_id: call.call_id.clone().filter(|id| !id.is_empty()),
                            item_id: Some(call.id.clone()).filter(|id| !id.is_empty()),
                            name: call.function.name.clone(),
                        });
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
                    if let Some(name) = result.name.as_deref() {
                        if let Some(index) = pending.iter().position(|call| {
                            call_id.is_some_and(|id| call.matches(id))
                                || (!result.id.is_empty() && call.matches(&result.id))
                        }) {
                            pending.remove(index);
                        } else if let Some(index) =
                            pending.iter().position(|call| call.name == name)
                        {
                            // No identifier matched, but the known name
                            // does: the result answers *that* call, which
                            // need not be the oldest — front-popping here
                            // would discard an unrelated call and shift
                            // every later positional pairing. Duplicate
                            // names take the oldest, preserving wire order.
                            // (Legacy-shim tier only: id-keyed wires never
                            // reach it.)
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
                    //      name (the legacy name-in-id encoding).
                    //   4. `call_id` matched a call, but `id` is a distinct
                    //      non-empty value that is *neither of that call's
                    //      identifiers* nor its name — `call_id` already
                    //      carries the association and the id belongs to no
                    //      identifier slot, so it is the *executed* name
                    //      (the legacy repair-hook-rename encoding: the tool
                    //      that ran differs from the tool the model called).
                    //      Dual-identifier wires (OpenAI Responses: item id
                    //      `fc_…` + `call_id` `call_…`) mirror both onto the
                    //      result; the mirrored item id matches the call's
                    //      identifier set, so it stays an identifier.
                    let identity_match = pending.iter().position(|call| {
                        call_id.is_some_and(|id| call.matches(id))
                            || (!result.id.is_empty() && call.matches(&result.id))
                    });
                    let matched = identity_match.or_else(|| {
                        // A non-matching identifier is a veto, never a
                        // license to pair positionally.
                        (result.id.is_empty() && call_id.is_none() && !pending.is_empty())
                            .then_some(0)
                    });
                    match matched {
                        Some(index) => {
                            if let Some(call) = pending.remove(index) {
                                let id_is_divergent_name = !result.id.is_empty()
                                    && !call.matches(&result.id)
                                    && result.id != call.name;
                                result.name = if id_is_divergent_name {
                                    // Shape 4: the association came from
                                    // `call_id`; the id carries the name of
                                    // the tool that actually executed.
                                    Some(result.id.clone())
                                } else {
                                    Some(call.name)
                                };
                            }
                        }
                        None if !result.id.is_empty() => {
                            // Shape 3: the id is the name — of the answered
                            // call, which need not be the oldest pending
                            // one. Consume the oldest call *bearing that
                            // name* when one exists so an out-of-order
                            // legacy result doesn't discard an unrelated
                            // call and shift every later positional
                            // pairing; only a name matching nothing falls
                            // back to wire order.
                            result.name = Some(result.id.clone());
                            match pending.iter().position(|call| call.name == result.id) {
                                Some(index) => {
                                    pending.remove(index);
                                }
                                None => {
                                    pending.pop_front();
                                }
                            }
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

    /// Shape 4: the legacy repair-hook-rename encoding. The result's
    /// `call_id` carries the association; its `id` holds the *executed*
    /// tool's name, which differs from the paired call's name. The executed
    /// name wins — the association is not a license to rename what ran.
    #[test]
    fn a_call_id_paired_result_with_a_divergent_name_in_id_keeps_the_executed_name() {
        let mut history = vec![
            call("call_1", Some("call_1"), "get_weather"),
            result("executed_tool", Some("call_1"), None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("executed_tool".into()));
    }

    /// The shape-4 carve-out never fires when the id IS the matched
    /// identifier (finding #5 stays fixed) or when it equals the call's
    /// name — both resolve to the call's function name.
    #[test]
    fn a_call_id_paired_result_whose_id_matches_identity_or_name_takes_the_calls_name() {
        let mut history = vec![
            call("call_abc", Some("call_abc"), "get_weather"),
            result("call_abc", Some("call_abc"), None),
            call("call_2", Some("call_2"), "add"),
            result("add", Some("call_2"), None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("get_weather".into()));
        assert_eq!(resolved_name(&history, 3), Some("add".into()));
    }

    /// Dual-identifier wires (OpenAI Responses) issue an item id (`fc_…`)
    /// AND a `call_id` (`call_…`); a legacy result mirrors both. The
    /// mirrored item id matches the call's identifier *set*, so it is an
    /// identifier — the resolved name is the call's function name, never
    /// `fc_1`.
    #[test]
    fn a_dual_identifier_result_resolves_to_the_calls_name() {
        let mut history = vec![
            call("fc_1", Some("call_1"), "get_weather"),
            result("fc_1", Some("call_1"), None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("get_weather".into()));
    }

    /// Shape 4 still wins on a dual-identifier call when the result's id
    /// matches *neither* identifier nor the called name: it is the executed
    /// tool's name.
    #[test]
    fn a_dual_identifier_call_still_honors_a_genuine_repair_rename() {
        let mut history = vec![
            call("fc_1", Some("call_1"), "get_weather"),
            result("executed_tool", Some("call_1"), None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("executed_tool".into()));
    }

    /// A result carrying only the item id (no `call_id`) still pairs with
    /// its call — either identifier slot associates.
    #[test]
    fn an_item_id_only_result_pairs_with_its_call() {
        let mut history = vec![
            call("fc_1", Some("call_1"), "get_weather"),
            result("fc_1", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 1), Some("get_weather".into()));
    }

    /// Parallel dual-identifier calls with results replayed in swapped
    /// order: each result pairs by identifier, not position, and resolves
    /// its own call's name.
    #[test]
    fn parallel_dual_identifier_results_resolve_out_of_order() {
        let mut history = vec![
            call("fc_1", Some("call_1"), "get_weather"),
            call("fc_2", Some("call_2"), "get_time"),
            result("fc_2", Some("call_2"), None),
            result("fc_1", Some("call_1"), None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 2), Some("get_time".into()));
        assert_eq!(resolved_name(&history, 3), Some("get_weather".into()));
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

    /// A legacy name-in-id result answering a LATER call consumes the call
    /// it names, not the oldest pending one — front-popping would discard
    /// an unrelated call and shift every subsequent pairing.
    #[test]
    fn an_out_of_order_name_in_id_result_consumes_the_call_it_names() {
        let mut history = vec![
            call("1", None, "alpha"),
            call("2", None, "beta"),
            result("beta", None, None),
            result("1", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(resolved_name(&history, 2), Some("beta".into()));
        assert_eq!(
            resolved_name(&history, 3),
            Some("alpha".into()),
            "the beta result must not have consumed alpha's pending slot"
        );
    }

    /// Same discipline for a named result on an id-less history: with no
    /// identifier to match, the known name selects the pending call to
    /// consume before wire order does.
    #[test]
    fn an_out_of_order_named_result_consumes_the_call_it_names() {
        let mut history = vec![
            call("", None, "alpha"),
            call("", None, "beta"),
            result("", None, Some("beta")),
            result("", None, None),
        ];
        resolve_tool_result_names(&mut history);
        assert_eq!(
            resolved_name(&history, 3),
            Some("alpha".into()),
            "the named beta result must not have consumed alpha's pending slot"
        );
    }
}
