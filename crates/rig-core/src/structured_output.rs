//! Model-facing structured-output policy shared by the agent runtimes: the
//! synthetic output tool's name and description, the instructions and reprompts
//! the model sees, and the required-field check on an answer.
//!
//! ```
//! use rig_core::structured_output::{output_tool_augmentation, output_tool_name};
//! let name = output_tool_name(|taken| taken == "final_result");
//! assert_eq!(name, "final_result_1");
//! assert!(output_tool_augmentation(&name).contains("`final_result_1`"));
//! ```

use crate::message::ToolChoice;

/// The output tool's default name.
pub const OUTPUT_TOOL_NAME: &str = "final_result";

/// The output tool's default description.
pub const OUTPUT_TOOL_DESCRIPTION: &str = "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.";

/// The separator between the preamble and an augmentation.
pub const AUGMENTATION_SEPARATOR: &str = "\n\n";

/// Appended to the preamble when the answer is asked for through the output
/// tool named `name`.
pub fn output_tool_augmentation(name: &str) -> String {
    format!(
        "When you have gathered enough information to answer, call the `{name}` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text."
    )
}

/// Appended to the preamble when the answer is asked for as prompted JSON;
/// `schema` is the schema's canonical rendering.
pub fn prompted_augmentation(schema: &str) -> String {
    format!(
        "Respond with ONLY a single JSON object that conforms to this JSON Schema. Do not include any prose, explanation, or markdown code fences.\n{schema}"
    )
}

/// The reprompt when the model answered as text instead of calling the output
/// tool named `name`.
pub fn reprompt_text_answer(name: &str) -> String {
    format!(
        "Provide your final answer by calling the `{name}` tool with the structured result as its arguments, not as plain text."
    )
}

/// The reprompt when the output tool named `name` was called without the
/// `missing` required fields.
pub fn reprompt_missing_fields(name: &str, missing: &[String]) -> String {
    format!(
        "The `{name}` arguments were missing required field(s): {}. Call `{name}` again with every required field.",
        missing.join(", ")
    )
}

/// The output tool's name for a run: the default, numbered from 1 while
/// `is_taken` reports a collision (`final_result`, `final_result_1`, ...).
pub fn output_tool_name(is_taken: impl Fn(&str) -> bool) -> String {
    let mut name = OUTPUT_TOOL_NAME.to_owned();
    let mut suffix = 1u32;
    while is_taken(&name) {
        name = format!("{OUTPUT_TOOL_NAME}_{suffix}");
        suffix += 1;
    }
    name
}

/// Whether the tool choice permits calling the output tool named `name`.
/// No choice, `Auto`, `Required`, and a `Specific` set naming it permit it.
pub fn output_tool_callable(choice: Option<&ToolChoice>, name: &str) -> bool {
    match choice {
        None | Some(ToolChoice::Auto | ToolChoice::Required) => true,
        Some(ToolChoice::None) => false,
        Some(ToolChoice::Specific { function_names }) => {
            function_names.iter().any(|named| named == name)
        }
    }
}

/// The top-level required fields of an object schema that `arguments` lacks,
/// in the schema's order. A non-object argument lacks every one.
pub fn missing_required_fields(
    schema: &serde_json::Value,
    arguments: &serde_json::Value,
) -> Vec<String> {
    let object = arguments.as_object();
    schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .filter(|name| object.is_none_or(|object| !object.contains_key(*name)))
        .map(str::to_owned)
        .collect()
}

/// Whether `text` parses as JSON with every top-level required field of
/// `schema`. Without a schema any JSON passes. Field types and other schema
/// constraints are not checked.
pub fn text_satisfies_schema(schema: Option<&serde_json::Value>, text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text.trim())
        .ok()
        .is_some_and(|value| {
            schema.is_none_or(|schema| missing_required_fields(schema, &value).is_empty())
        })
}
