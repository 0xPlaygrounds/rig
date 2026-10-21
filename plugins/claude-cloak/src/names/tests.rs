use super::*;
use rig_harness::rig_core::{
    completion::{CompletionResponse, Usage},
    message::{Fingerprint, Native, Origin, ToolCall, ToolFunction},
    streaming::{Part, PartKind},
};
use serde_json::json;

#[test]
fn streamed_start_removes_exactly_one_prefix_and_preserves_builtins() -> anyhow::Result<()> {
    for (wire, expected) in [
        ("_read", "read"),
        ("__private", "_private"),
        ("web_search", "web_search"),
        ("_", "_"),
    ] {
        let Ok(name) = ToolName::new(wire) else {
            anyhow::bail!("nonempty fixture")
        };
        let result = decode(Relayed::Item(Item::Event(StreamEvent::Start {
            part: Part::new(2),
            kind: PartKind::ToolCall,
            name: Some(name),
        })));
        match result {
            Relayed::Item(Item::Event(StreamEvent::Start {
                part,
                name: Some(name),
                ..
            })) => {
                anyhow::ensure!((part.index()) == (2));
                anyhow::ensure!((name.as_str()) == (expected));
            }
            _ => anyhow::bail!("start remains start"),
        }
    }
    Ok(())
}

#[test]
fn completed_parts_and_final_response_decode_without_changing_arguments_or_ids()
-> anyhow::Result<()> {
    let Ok(name) = ToolName::new("__private") else {
        anyhow::bail!("nonempty fixture")
    };
    let mut original = ToolCall::from_wire(
        "call_1",
        ToolFunction::new(name, json!({"text":"_not_a_name"})),
    );
    original.native = Some(Native {
        item: json!({"type":"tool_use","id":"call_1","name":"__private","input":{"text":"_not_a_name"}}),
        fingerprint: Fingerprint::of(&original.function),
    });
    let content = AssistantContent::ToolCall(original.clone());
    let result = decode(Relayed::Item(Item::Event(StreamEvent::End {
        part: Part::new(0),
        content: content.clone(),
    })));
    let final_reply = decode(Relayed::Done(Arc::new(CompletionResponse::new(
        vec![content],
        Usage::default(),
        Origin::new("anthropic", "anthropic-oauth", "test"),
        json!({}),
    ))));
    let check = |content: &AssistantContent| -> anyhow::Result<()> {
        let AssistantContent::ToolCall(call) = content else {
            anyhow::bail!("tool remains tool")
        };
        anyhow::ensure!((call.function.name.as_str()) == ("_private"));
        anyhow::ensure!((call.id) == (original.id));
        anyhow::ensure!((call.function.arguments) == (original.function.arguments));
        anyhow::ensure!(call.native.is_none());
        Ok(())
    };
    match result {
        Relayed::Item(Item::Event(StreamEvent::End { content, .. })) => check(&content)?,
        _ => anyhow::bail!("end remains end"),
    }
    match final_reply {
        Relayed::Done(response) => {
            anyhow::ensure!((response.choice.len()) == (1));
            for content in &response.choice {
                check(content)?;
            }
        }
        _ => anyhow::bail!("done remains done"),
    }
    Ok(())
}
