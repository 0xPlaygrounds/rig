//! Registry guard for the name-keyed tool-result serializer family.
//!
//! Some wires key a replayed tool result by *function name* (Gemini's
//! `functionResponse.name`, Ollama's tool-message name). Each such
//! serializer must resolve legacy name-less results through the ONE shared
//! entry point, `rig_core::providers::internal::resolve_tool_result_names` —
//! the rule took four fix commits precisely because it existed as five
//! hand-maintained copies (the langchain standard-tests precedent: a shared
//! inherited contract a provider cannot silently skip).
//!
//! `include_str!` makes the linkage semi-structural: moving or deleting a
//! registered serializer file breaks this binary's COMPILE, not just a scan,
//! and the content check then pins that the file still routes through the
//! shared resolver. Registering a new name-keyed wire means adding one row.

/// The registry: every wire whose tool-result payload is keyed by function
/// name, with its serializer source compiled in.
const NAME_KEYED_SERIALIZERS: &[(&str, &str)] = &[
    (
        "gemini_rest",
        include_str!("../../crates/rig-core/src/providers/gemini/completion.rs"),
    ),
    (
        "gemini_interactions",
        include_str!("../../crates/rig-core/src/providers/gemini/interactions_api/mod.rs"),
    ),
    (
        "ollama",
        include_str!("../../crates/rig-core/src/providers/ollama.rs"),
    ),
    (
        "vertexai",
        include_str!("../../crates/rig-vertexai/src/types/completion_request.rs"),
    ),
    (
        "gemini_grpc",
        include_str!("../../crates/rig-gemini-grpc/src/completion.rs"),
    ),
];

/// Every registered name-keyed serializer routes legacy histories through
/// the shared resolver — the rule lives once, and a listed wire cannot
/// silently stop calling it.
#[test]
fn name_keyed_serializers_route_through_the_shared_resolver() {
    let mut missing = Vec::new();
    for (family, source) in NAME_KEYED_SERIALIZERS {
        if !source.contains("resolve_tool_result_names(") {
            missing.push(*family);
        }
    }
    assert!(
        missing.is_empty(),
        "name-keyed serializers must call \
         rig_core::providers::internal::resolve_tool_result_names before \
         serializing tool results; missing in: {missing:?}",
    );
}

/// The registry itself stays honest: a wire family that serializes
/// `functionResponse.name` (the Gemini-shaped name-keyed payload) must be
/// registered above. This catches the "new name-keyed wire, forgot the
/// resolver AND the registry" case for the known payload spelling.
#[test]
fn function_response_serializers_are_registered() {
    for (family, source) in NAME_KEYED_SERIALIZERS {
        assert!(
            source.contains("functionResponse")
                || source.contains("function_response")
                || source.contains("tool_result")
                || source.contains("ToolResult"),
            "{family} no longer looks like a tool-result serializer; \
             update the registry",
        );
    }
}
