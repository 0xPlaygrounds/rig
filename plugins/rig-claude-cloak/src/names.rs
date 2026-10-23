use std::sync::Arc;

use rig_harness::rig_core::{
    message::{AssistantContent, ToolName},
    serve::ErasedHandler,
    streaming::{Item, Relayed, StreamEvent},
};

pub(crate) fn decode_handler(handler: ErasedHandler) -> ErasedHandler {
    handler.map_stream("anthropic-oauth-tool-names", |_| |item| vec![decode(item)])
}

fn decode_name(name: &mut ToolName) {
    if let Some(unprefixed) = name.as_str().strip_prefix('_')
        && let Ok(decoded) = ToolName::new(unprefixed)
    {
        *name = decoded;
    }
}

fn decode_content(content: &mut AssistantContent) {
    if !matches!(content, AssistantContent::ToolCall(_)) {
        return;
    }
    let original = content.fingerprint();
    if let AssistantContent::ToolCall(call) = content {
        decode_name(&mut call.function.name);
    }
    let decoded = content.fingerprint();
    if original != decoded
        && let AssistantContent::ToolCall(call) = content
        && let Some(native) = &mut call.native
    {
        if native.fingerprint == original {
            // Native replay must use the canonical name before transport
            // adds its prefix, without losing provider-owned metadata.
            if let Some(name) = native.item.get_mut("name") {
                *name = serde_json::Value::String(call.function.name.to_string());
            }
            native.fingerprint = decoded;
        } else {
            // A stale item must stay stale even if its old fingerprint
            // happens to match the decoded name.
            native.fingerprint = original;
        }
    }
}

fn decode(mut item: Relayed) -> Relayed {
    match &mut item {
        Relayed::Item(Item::Event(StreamEvent::Start {
            name: Some(name), ..
        })) => decode_name(name),
        Relayed::Item(Item::Event(StreamEvent::End { content, .. })) => decode_content(content),
        Relayed::Done(response) => {
            for content in &mut Arc::make_mut(response).choice {
                decode_content(content);
            }
        }
        _ => {}
    }
    item
}

#[cfg(test)]
mod tests;
