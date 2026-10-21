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
    if let AssistantContent::ToolCall(call) = content {
        decode_name(&mut call.function.name);
        // Rebuild history from canonical names so an already-prefixed native
        // wire item cannot receive the outgoing prefix a second time.
        call.native = None;
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
