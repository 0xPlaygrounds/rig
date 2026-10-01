use super::*;

#[test]
fn the_output_tool_is_callable_unless_the_choice_excludes_it() {
    let specific = |names: &[&str]| ToolChoice::Specific {
        function_names: names
            .iter()
            .map(|name| crate::message::ToolName::new(*name).expect("tool name"))
            .collect(),
    };
    assert!(output_tool_callable(None, "final_result"));
    assert!(output_tool_callable(
        Some(&ToolChoice::Auto),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&ToolChoice::Required),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&specific(&["add", "final_result"])),
        "final_result"
    ));
    assert!(!output_tool_callable(
        Some(&ToolChoice::None),
        "final_result"
    ));
    assert!(!output_tool_callable(
        Some(&specific(&["add"])),
        "final_result"
    ));
}

#[test]
fn the_output_tool_name_numbers_from_one_past_taken_names() {
    let taken = |names: &'static [&'static str]| move |name: &str| names.contains(&name);
    assert_eq!(output_tool_name(taken(&["add"])), "final_result");
    assert_eq!(output_tool_name(taken(&["final_result"])), "final_result_1");
    assert_eq!(
        output_tool_name(taken(&["final_result", "final_result_1"])),
        "final_result_2"
    );
}

#[test]
fn required_fields_are_checked_on_arguments_and_json_text() {
    let schema = serde_json::json!({"type": "object", "required": ["a", "b"]});
    assert_eq!(
        missing_required_fields(&schema, &serde_json::json!({"a": 1})),
        ["b"]
    );
    assert_eq!(
        missing_required_fields(&schema, &serde_json::Value::Null),
        ["a", "b"]
    );
    assert!(missing_required_fields(&serde_json::json!({}), &serde_json::Value::Null).is_empty());
    assert!(text_satisfies_schema(
        Some(&schema),
        r#" {"a": 1, "b": 2} "#
    ));
    assert!(!text_satisfies_schema(Some(&schema), r#"{"a": 1}"#));
    assert!(!text_satisfies_schema(None, "not json"));
    assert!(text_satisfies_schema(None, "[1]"));
}
