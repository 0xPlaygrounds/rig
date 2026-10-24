use super::*;
use crate::message::ToolFunction;

#[test]
fn argument_feedback_obeys_the_schema_and_reports_bounded_problems() {
    let name = ToolName::new("add");
    assert!(name.is_ok());
    let Ok(name) = name else { return };
    let extra =
        "`add` has no argument `c`. Its arguments are: `a`, `b`. Call it again with only those.";
    let typed = "`add`'s arguments do not fit its schema: /limit: expected an integer, got a string; /path: missing. Call it again with arguments that fit.";
    let many = "`add`'s arguments do not fit its schema: /a: missing; /b: missing; /c: missing; /d: missing; /e: missing; and 2 more. Call it again with arguments that fit.";
    let invalid = invalid_arguments_feedback("add", "[1, 2]");
    for (schema, raw, expected) in [
        (
            serde_json::json!({"properties": {"a": {}, "b": {}}}),
            r#"{"a":1,"c":2}"#,
            Some(extra),
        ),
        (
            serde_json::json!({"properties": {"a": {}, "b": {}}}),
            "{}",
            None,
        ),
        (
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}),
            r#"{"limit":"ten"}"#,
            Some(typed),
        ),
        (
            serde_json::json!({"type": "object", "properties": {}, "required": ["a", "b", "c", "d", "e", "f", "g"]}),
            "{}",
            Some(many),
        ),
        (
            serde_json::json!({"properties": {}, "required": ["a"]}),
            "[1, 2]",
            Some(invalid.as_str()),
        ),
    ] {
        let call = ToolCall::from_wire("call", ToolFunction::parse(name.clone(), raw));
        assert_eq!(
            arguments_refusal(&schema, &call).as_deref(),
            expected,
            "{schema}, {raw}"
        );
    }
    let call = ToolCall::from_wire(
        "call",
        ToolFunction::new(name, serde_json::json!({"path": "x"})),
    );
    for open in [
        serde_json::json!({}),
        serde_json::json!({"type": "object"}),
        serde_json::json!({"type": "object", "properties": {}, "additionalProperties": true}),
        serde_json::json!({"properties": {"a": {}}, "additionalProperties": {"type": "string"}}),
        serde_json::json!({"properties": {"a": {}}, "patternProperties": {"^p": {}}}),
    ] {
        assert_eq!(arguments_refusal(&open, &call), None, "{open}");
    }
    let closed = serde_json::json!({"properties": {}, "additionalProperties": false});
    assert!(arguments_refusal(&closed, &call).is_some());
}

#[test]
fn a_call_cut_off_by_the_output_limit_is_told_so_with_its_start_echoed() {
    let raw = format!(
        "{{\"path\": \"big.rs\", \"content\": \"{}",
        "x".repeat(50_000)
    );
    let text = output_limit_feedback("write", &raw);
    assert!(text.starts_with("Not run: your reply reached the output limit before `write`"));
    assert!(text.contains("big.rs"), "{text}");
    assert!(text.len() < 1_000, "{}", text.len());
    assert!(output_limit_feedback("write", "{}").ends_with("began: {}"));
}

#[test]
fn a_call_that_did_not_finish_says_whether_it_ran_and_what_it_may_have_changed() {
    use crate::tool::{Scope, Touches};
    let why = "the user stopped the turn.";
    let reads = Touches::Reads(Scope::Anything);
    let writes = Touches::Writes(Scope::Anything);
    let said = [
        not_finished(why, false, &writes),
        not_finished(why, true, &reads),
        not_finished(why, true, &writes),
        not_finished(why, true, &Touches::Independent),
    ];
    let partly = "Stopped: the user stopped the turn. It may have partly run; check before \
                  running it again.";
    assert_eq!(
        said,
        [
            "Not run: the user stopped the turn.",
            "Stopped: the user stopped the turn. It only read; run it again if you still need it.",
            partly,
            partly,
        ]
    );
}

#[test]
fn unknown_tool_feedback_lists_exact_names_or_no_tools() {
    assert_eq!(
        unknown_tool_feedback("bash", ["write", "read", "edit", "shell"]),
        "There is no tool named `bash`. Your tools are: `edit`, `read`, `shell`, `write`. Call one by its exact name."
    );
    assert_eq!(
        unknown_tool_feedback("bash", []),
        "There is no tool named `bash`. You have no tools available. Answer without calling tools."
    );
}
