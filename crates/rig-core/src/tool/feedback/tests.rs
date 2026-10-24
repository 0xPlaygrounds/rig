use super::*;
use crate::message::ToolFunction;

fn tool_call(id: &str) -> ToolCall {
    ToolCall {
        id: CallId::from_wire(id),
        function: ToolFunction::new(
            ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
        native: None,
    }
}

#[test]
fn arguments_the_schema_does_not_declare_are_refused() {
    let schema = serde_json::json!({"properties": {"a": {}, "b": {}}});
    let mut extra = tool_call("c1");
    extra.function = ToolFunction::new(
        crate::message::ToolName::new("add").expect("tool name"),
        serde_json::json!({"a": 1, "c": 2}),
    );
    assert_eq!(
        arguments_refusal(&schema, &extra).as_deref(),
        Some(
            "`add` has no argument `c`. Its arguments are: `a`, `b`. Call it again with only those."
        )
    );
    assert_eq!(arguments_refusal(&schema, &tool_call("c2")), None);
}

#[test]
fn ill_typed_arguments_are_refused() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    });
    let mut ill_typed = tool_call("c1");
    ill_typed.function = ToolFunction::new(
        crate::message::ToolName::new("read").expect("tool name"),
        serde_json::json!({"limit": "ten"}),
    );
    assert_eq!(
        arguments_refusal(&schema, &ill_typed).as_deref(),
        Some(
            "`read`'s arguments do not fit its schema: /limit: expected an integer, got a \
             string; /path: missing. Call it again with arguments that fit."
        )
    );
}

#[test]
fn a_refusal_names_five_problems_and_counts_the_rest() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {},
        "required": ["a", "b", "c", "d", "e", "f", "g"],
    });
    assert_eq!(
        arguments_refusal(&schema, &tool_call("c1")).as_deref(),
        Some(
            "`add`'s arguments do not fit its schema: /a: missing; /b: missing; /c: missing; \
             /d: missing; /e: missing; and 2 more. Call it again with arguments that fit."
        )
    );
}

#[test]
fn arguments_that_are_not_an_object_are_refused_before_the_schema() {
    let mut call = tool_call("c1");
    call.function = ToolFunction::parse(
        crate::message::ToolName::new("add").expect("tool name"),
        "[1, 2]",
    );
    let schema = serde_json::json!({"properties": {}, "required": ["a"]});
    assert_eq!(
        arguments_refusal(&schema, &call),
        Some(invalid_arguments_feedback("add", "[1, 2]"))
    );
}

#[test]
fn a_schema_that_names_no_arguments_or_allows_more_accepts_any_argument() {
    let mut call = tool_call("c1");
    call.function = ToolFunction::new(
        crate::message::ToolName::new("open").expect("tool name"),
        serde_json::json!({"path": "x"}),
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
