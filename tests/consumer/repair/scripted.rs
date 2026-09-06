//! Synthetic policy stimulus only. Genuine providers never receive these patches.

use rig_core::{
    completion::CompletionRequest,
    message::{AssistantContent, Message, ToolResultContent, UserContent},
};
use serde_json::{Value, json};

pub(crate) fn choice(request: &CompletionRequest) -> Vec<AssistantContent> {
    let results: Vec<_> = request
        .chat_history
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|content| match content {
                    UserContent::ToolResult(result) => Some(result),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        })
        .collect();
    let last = results
        .last()
        .and_then(|result| {
            result.content.iter().find_map(|content| match content {
                ToolResultContent::Json { value } => Some(value.clone()),
                ToolResultContent::Text(text) => serde_json::from_str::<Value>(&text.text).ok(),
                _ => None,
            })
        })
        .unwrap_or(Value::Null);
    let fault = request
        .additional_params
        .as_ref()
        .and_then(|params| params.get("repair_fault"))
        .and_then(Value::as_str)
        .unwrap_or("none");
    let production = match fault {
        "repair_insufficient" => {
            "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let offset = offset.min(items.len());\n    let end = (offset + limit).min(items.len());\n    &items[offset..end]\n}\n"
        }
        "repair_timeout" => {
            "pub fn page<T>(_items: &[T], _offset: usize, _limit: usize) -> &[T] {\n    loop {\n        std::thread::sleep(std::time::Duration::from_millis(5));\n    }\n}\n"
        }
        _ => {
            "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let start = offset.min(items.len());\n    let count = limit.min(items.len() - start);\n    &items[start..start + count]\n}\n"
        }
    };
    let call = |id, name, args| AssistantContent::tool_call(id, name, args);
    match results.len() {
        0 => vec![
            call("list", "repo_list_files", json!({})),
            call("initial", "repo_validate", json!({"phase":"initial"})),
        ],
        2 => vec![
            call("source", "repo_read_file", json!({"path":"src/lib.rs"})),
            call(
                "tests",
                "repo_read_file",
                json!({"path":"tests/pagination.rs"}),
            ),
            call("contract", "repo_read_file", json!({"path":"README.md"})),
        ],
        5 => vec![
            AssistantContent::text(
                "The observed out-of-range slice comes from using the unchecked offset as the lower bound. The contract requires empty pages beyond the end.",
            ),
            call(
                "regression-proposal",
                "repo_propose_patch",
                json!({"path":"tests/regression.rs","content":"#[test]\nfn excessive_offset_is_empty() {\n    assert!(page_window::page(&[1, 2, 3], 5, 1).is_empty());\n}\n","justification":"Reproduce the observed out-of-range lower bound while retaining the contract's empty-page expectation."}),
            ),
        ],
        6 | 9 => vec![call(
            "apply",
            "repo_apply_patch",
            json!({"operation":last.pointer("/proposal/operation")}),
        )],
        7 => vec![call(
            "regression",
            "repo_validate",
            json!({"phase":"regression"}),
        )],
        8 => vec![call(
            "production-proposal",
            "repo_propose_patch",
            json!({"path":"src/lib.rs","content":production,"justification":"The regression failed on the original slice boundary. Bound the start by the slice length and the count by remaining items so all usize inputs preserve the documented behavior without addition overflow."}),
        )],
        10 if last.get("applied").and_then(Value::as_bool) == Some(true) => {
            vec![call("final", "repo_validate", json!({"phase":"final"}))]
        }
        _ => vec![AssistantContent::text(
            if matches!(fault, "repair_insufficient" | "repair_timeout") {
                "All tests passed; the repair is successful. This is a deliberately false synthetic claim."
            } else if last.get("validation").is_some() {
                "Validation finished; the recorded tool report determines whether the repair succeeded."
            } else {
                "The production edit was not applied; the repair is incomplete."
            },
        )],
    }
}
