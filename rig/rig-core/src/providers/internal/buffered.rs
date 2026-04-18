//! Helpers for adapting buffered provider responses into streaming responses.

use async_stream::stream;

use crate::completion::{self, CompletionError, GetTokenUsage};
use crate::message::AssistantContent;
use crate::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use crate::wasm_compat::WasmCompatSend;

pub(crate) fn stream_from_completion_response<R, F>(
    response: completion::CompletionResponse<R>,
    mut map_content: F,
) -> Result<StreamingCompletionResponse<R>, CompletionError>
where
    R: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    F: FnMut(AssistantContent) -> Result<Vec<RawStreamingChoice<R>>, CompletionError>
        + WasmCompatSend
        + 'static,
{
    let stream = stream! {
        for content in response.choice.clone() {
            let mapped = match map_content(content) {
                Ok(mapped) => mapped,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            for choice in mapped {
                yield Ok(choice);
            }
        }

        yield Ok(RawStreamingChoice::FinalResponse(response.raw_response.clone()));
    };

    Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
}
