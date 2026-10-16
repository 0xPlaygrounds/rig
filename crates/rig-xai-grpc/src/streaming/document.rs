//! The xAI gRPC reply document: the `GetChatCompletionResponse` a reply
//! adds up to, as its proto3 JSON, the response's `raw`.

use serde_json::Value;

use crate::completion::XaiFrame;
use crate::json::to_json;
use crate::proto::{CompletionOutput, GetChatCompletionChunk, GetChatCompletionResponse};
use rig_core::wire::document::{Reassemble, Serves};

/// The response being rebuilt from one reply's frames. A unary reply is
/// its response; a stream's chunks fold into one as xAI's own SDK folds
/// them: text appends, tool calls and citations append, and the envelope
/// and usage are the last chunk's.
#[derive(Debug, Default)]
pub struct XaiDocument(Option<GetChatCompletionResponse>);

impl XaiDocument {
    fn chunk(&mut self, chunk: &GetChatCompletionChunk) {
        let response = self
            .0
            .get_or_insert_with(GetChatCompletionResponse::default);
        response.id.clone_from(&chunk.id);
        response.model.clone_from(&chunk.model);
        response
            .system_fingerprint
            .clone_from(&chunk.system_fingerprint);
        if chunk.created.is_some() {
            response.created = chunk.created;
        }
        if chunk.usage.is_some() {
            response.usage.clone_from(&chunk.usage);
        }
        if chunk.service_tier != 0 {
            response.service_tier = chunk.service_tier;
        }
        response.citations.extend(chunk.citations.iter().cloned());
        response
            .output_files
            .extend(chunk.output_files.iter().cloned());
        for output in &chunk.outputs {
            let position = match response
                .outputs
                .iter()
                .position(|known| known.index == output.index)
            {
                Some(position) => position,
                None => {
                    response.outputs.push(CompletionOutput {
                        index: output.index,
                        ..CompletionOutput::default()
                    });
                    response.outputs.len().saturating_sub(1)
                }
            };
            let Some(folded) = response.outputs.get_mut(position) else {
                continue;
            };
            if output.finish_reason != 0 {
                folded.finish_reason = output.finish_reason;
            }
            let Some(delta) = &output.delta else {
                continue;
            };
            let message = folded.message.get_or_insert_with(Default::default);
            if delta.role != 0 {
                message.role = delta.role;
            }
            message.content.push_str(&delta.content);
            message.reasoning_content.push_str(&delta.reasoning_content);
            message.encrypted_content.push_str(&delta.encrypted_content);
            message.tool_calls.extend(delta.tool_calls.iter().cloned());
            message.citations.extend(delta.citations.iter().cloned());
        }
    }
}

impl Reassemble<XaiFrame> for XaiDocument {
    fn absorb(&mut self, frame: &XaiFrame) {
        match frame {
            XaiFrame::Whole(response) => self.0 = Some(response.clone()),
            XaiFrame::Chunk(chunk) => self.chunk(chunk),
        }
    }

    fn finish(self) -> Value {
        self.0
            .and_then(|response| to_json(&response).ok())
            .unwrap_or(Value::Null)
    }
}

impl Serves<rig_core::operation::Completion> for XaiDocument {}
