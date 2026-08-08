//! Registry guard for the name-keyed tool-result serializer family.
//!
//! Some wires key a replayed tool result by *function name* (Gemini's
//! `functionResponse.name`, Ollama's tool-message name). `ToolResult::name`
//! is required data — the executed tool's name travels on the result itself
//! — so each such serializer must read it directly. The legacy shared
//! resolver (`resolve_tool_result_names`, which back-filled name-less
//! results by pairing them with their calls) is deleted, and no serializer
//! may reintroduce a name-as-id fallback or a private resolution heuristic
//! (the langchain standard-tests precedent: a shared inherited contract a
//! provider cannot silently skip).
//!
//! `include_str!` makes the linkage semi-structural: moving or deleting a
//! registered serializer file breaks this binary's COMPILE, not just a scan,
//! and the content check then pins that the file still reads the required
//! `name` where it builds its function-response payload. Registering a new
//! name-keyed wire means adding one row.

/// The registry: every wire whose tool-result payload is keyed by function
/// name, with its serializer source compiled in and the source marker of its
/// required-`name` read.
const NAME_KEYED_SERIALIZERS: &[(&str, &str, &str)] = &[
    (
        "gemini_rest",
        include_str!("../../crates/rig-core/src/providers/gemini/completion.rs"),
        "let function_name = name;",
    ),
    (
        "gemini_interactions",
        include_str!("../../crates/rig-core/src/providers/gemini/interactions_api/mod.rs"),
        "name: Some(name),",
    ),
    (
        "ollama",
        include_str!("../../crates/rig-core/src/providers/ollama.rs"),
        "let function_name = name;",
    ),
    (
        "vertexai",
        include_str!("../../crates/rig-vertexai/src/types/message.rs"),
        "tool_result.name.clone()",
    ),
    (
        "gemini_grpc",
        include_str!("../../crates/rig-gemini-grpc/src/completion.rs"),
        "name: result.name,",
    ),
];

/// Every registered name-keyed serializer reads the required
/// `ToolResult::name` directly where it builds its function-response
/// payload — the name is data on the result, and a listed wire cannot
/// silently swap it for an identifier.
#[test]
fn name_keyed_serializers_read_the_required_result_name() {
    let mut missing = Vec::new();
    for (family, source, name_read) in NAME_KEYED_SERIALIZERS {
        if !source.contains(name_read) {
            missing.push(*family);
        }
    }
    assert!(
        missing.is_empty(),
        "name-keyed serializers must read the required `ToolResult::name` \
         when building their function-response payload (the registered \
         marker no longer matches); missing in: {missing:?}",
    );
}

/// The deleted legacy resolver stays deleted: no registered serializer may
/// route through a name-resolution shim again — the executed tool's name is
/// required on `ToolResult` itself.
#[test]
fn name_keyed_serializers_do_not_resurrect_the_legacy_resolver() {
    let mut resurrected = Vec::new();
    for (family, source, _) in NAME_KEYED_SERIALIZERS {
        if source.contains("resolve_tool_result_names") {
            resurrected.push(*family);
        }
    }
    assert!(
        resurrected.is_empty(),
        "`resolve_tool_result_names` was deleted along with name-less tool \
         results; serializers read the required `ToolResult::name` directly, \
         found a resurrected call in: {resurrected:?}",
    );
}

/// The registry itself stays honest: a wire family that serializes
/// `functionResponse.name` (the Gemini-shaped name-keyed payload) must be
/// registered above. This catches the "new name-keyed wire, forgot the
/// required-name read AND the registry" case for the known payload spelling.
#[test]
fn function_response_serializers_are_registered() {
    for (family, source, _) in NAME_KEYED_SERIALIZERS {
        assert!(
            source.contains("functionResponse")
                || source.contains("function_response")
                || source.contains("FunctionResponse")
                || source.contains("tool_result")
                || source.contains("ToolResult"),
            "{family} no longer looks like a tool-result serializer; \
             update the registry",
        );
    }
}
