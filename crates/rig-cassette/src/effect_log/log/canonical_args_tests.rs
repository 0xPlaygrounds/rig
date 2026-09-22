use super::canonical_tool_args;

/// Tool-call arguments that differ only in key order are the same call:
/// which order a build emits depends on serde_json feature unification.
#[test]
fn tool_arguments_compare_in_canonical_key_order() {
    assert_eq!(
        canonical_tool_args(r#"{"y":25,"x":17}"#),
        canonical_tool_args(r#"{"x":17,"y":25}"#)
    );
    assert_eq!(
        canonical_tool_args(r#"{"y":25,"x":17}"#),
        r#"{"x":17,"y":25}"#
    );
    assert_ne!(
        canonical_tool_args(r#"{"x":17,"y":25}"#),
        canonical_tool_args(r#"{"x":17,"y":26}"#)
    );
    assert_eq!(canonical_tool_args("not json"), "not json");
}
