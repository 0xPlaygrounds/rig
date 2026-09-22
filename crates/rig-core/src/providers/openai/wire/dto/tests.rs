use super::{StreamingFunction, StreamingToolCall};
use crate::providers::internal::tool_call_bridge::ToolCallBridge;

/// One streamed tool-call fragment, as a chunk at wire index 0.
fn fragment(id: Option<&str>, name: Option<&str>, arguments: Option<&str>) -> StreamingToolCall {
    StreamingToolCall {
        index: 0,
        id: id.map(ToOwned::to_owned),
        function: StreamingFunction {
            name: name.map(ToOwned::to_owned),
            arguments: arguments.map(ToOwned::to_owned),
        },
    }
}

/// Whether the `(id, name)` call open at index 0 must be evicted to make
/// room for `incoming`.
fn evicts(existing: (&str, &str), incoming: &StreamingToolCall) -> bool {
    let mut bridge = ToolCallBridge::<usize>::new();
    let slot = bridge.open(
        0,
        (!existing.0.is_empty()).then_some(existing.0),
        (!existing.1.is_empty()).then_some(existing.1),
    );
    incoming.evicts(slot)
}

/// Some gateways stream two *distinct* calls under the same `index`, which
/// is the only evidence that one call ended and another began. Getting this
/// wrong concatenates two calls' arguments into one unparseable blob, so the
/// rule is stated per input rather than as one happy path: a new id plus
/// either a different name or an argument-less opening fragment is a second
/// call; anything else is a continuation of the call already open.
#[test]
fn a_second_call_at_one_index_is_told_apart_by_id_and_opening_shape() {
    assert!(
        evicts(
            ("call_a", "get_weather"),
            &fragment(Some("call_b"), Some("get_time"), None)
        ),
        "a new id under a different name is a different call"
    );
    assert!(
        evicts(
            ("call_a", "get_weather"),
            &fragment(Some("call_b"), Some("get_weather"), None)
        ),
        "a new id re-announcing the same name with no arguments opens a second call"
    );
    assert!(
        !evicts(
            ("call_a", "get_weather"),
            &fragment(Some("call_b"), Some("get_weather"), Some(r#"{"city":"#)),
        ),
        "a same-named fragment carrying arguments continues the open call"
    );
    assert!(
        !evicts(
            ("call_a", "get_weather"),
            &fragment(Some("call_a"), Some("get_time"), None)
        ),
        "the same id is the same call, whatever the name says"
    );
    assert!(
        !evicts(
            ("call_a", "get_weather"),
            &fragment(None, Some("get_time"), None)
        ),
        "an id-less fragment carries no evidence of a second call"
    );
    assert!(
        !evicts(("", ""), &fragment(Some("call_b"), Some("get_time"), None)),
        "an id-less slot has no id to differ from"
    );
}
