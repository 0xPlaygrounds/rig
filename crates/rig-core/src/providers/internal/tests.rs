use crate::message::{
    AssistantContent, Message, ToolCall, ToolFunction, ToolResultContent, UserContent,
};

fn call(wire_id: &str, name: &str) -> Message {
    Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
            wire_id,
            ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
        ))],
    }
}

fn nameless_result(wire_id: &str) -> Message {
    Message::User {
        content: vec![UserContent::tool_result_from_wire(
            wire_id,
            "",
            vec![ToolResultContent::text("out")],
        )],
    }
}

fn result_names(history: &[Message]) -> Vec<String> {
    history
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => content.iter().next().and_then(|item| match item {
                UserContent::ToolResult(result) => Some(result.name.clone()),
                _ => None,
            }),
            _ => None,
        })
        .collect()
}

/// An ingested cross-provider transcript (converters stamp `name: ""`)
/// resolves each result's name from its paired call — by provider id
/// here, since `from_wire` on both sides shares it.
#[test]
fn empty_names_resolve_from_the_paired_call() {
    let mut history = vec![
        call("toolu_1", "get_weather"),
        nameless_result("toolu_1"),
        call("toolu_2", "get_time"),
        nameless_result("toolu_2"),
    ];
    super::resolve_empty_tool_result_names(&mut history);
    assert_eq!(result_names(&history), ["get_weather", "get_time"]);
}

/// A result no call in the history answers keeps its empty name: the
/// transcript genuinely lacks the data, and inventing one would ship a
/// fabricated name to a name-keyed wire.
#[test]
fn an_unmatched_result_keeps_its_empty_name() {
    let mut history = vec![call("toolu_1", "get_weather"), nameless_result("toolu_9")];
    super::resolve_empty_tool_result_names(&mut history);
    assert_eq!(result_names(&history), [""]);
}

/// An established name is data, never overwritten — a repair hook may
/// have renamed the executed tool relative to the model's call.
#[test]
fn an_established_name_is_never_overwritten() {
    let mut history = vec![
        call("toolu_1", "add"),
        Message::User {
            content: vec![UserContent::tool_result_from_wire(
                "toolu_1",
                "sum",
                vec![ToolResultContent::text("3")],
            )],
        },
    ];
    super::resolve_empty_tool_result_names(&mut history);
    assert_eq!(result_names(&history), ["sum"]);
}

/// Matching falls through the identifier tiers: rig's correlation
/// handle first (a driver-built result answering an id-less call),
/// then the provider identifiers.
#[test]
fn a_handle_only_result_resolves_from_an_id_less_call() {
    let id_less = ToolCall::new(
        crate::message::ToolCallId::minted(0),
        ToolFunction {
            name: "lookup".to_owned(),
            arguments: serde_json::json!({}),
        },
    );
    let handle = id_less.id.as_str().to_owned();
    let mut history = vec![
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(id_less)],
        },
        Message::User {
            content: vec![UserContent::tool_result(
                handle,
                "",
                vec![ToolResultContent::text("out")],
            )],
        },
    ];
    super::resolve_empty_tool_result_names(&mut history);
    assert_eq!(result_names(&history), ["lookup"]);
}
