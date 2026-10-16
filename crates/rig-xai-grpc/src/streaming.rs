//! Decodes xAI gRPC replies into Rig completion events.
//!
//! ```
//! use rig_core::wire::Wire;
//! use rig_xai_grpc::completion::GetCompletion;
//!
//! let decoder = GetCompletion::new("grok-4.3").decoder();
//! # let _ = decoder;
//! ```

use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::openai::wire::{ChatDecoder, ChatEvent};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};

use crate::chat::{self, Calls};
use crate::completion::XaiFrame;

/// The xAI gRPC wire's decoder: each reply message is read as the Chat
/// Completions JSON it carries, a unary response as a `chat.completion` and
/// a chunk as a `chat.completion.chunk`, by the xAI Chat decoder.
pub struct GrpcAdapter {
    chat: ChatDecoder,
    calls: Calls,
}

impl GrpcAdapter {
    /// The adapter over `chat`, the xAI Chat wire's decoder.
    pub(crate) fn new(chat: ChatDecoder) -> Self {
        Self {
            chat,
            calls: Calls::default(),
        }
    }
}

impl<'id> Decoder<'id, Completion, XaiFrame> for GrpcAdapter {
    type Event = XaiFrame;

    fn classify(&self, frame: XaiFrame) -> WireEvent<Self::Event> {
        // Tonic decodes the frames; every one is a reply message.
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: XaiFrame,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match &frame {
            XaiFrame::Whole(response) => ChatEvent::Whole(chat::whole(response)),
            XaiFrame::Chunk(chunk) => ChatEvent::Chunk(chat::chunk(chunk, &mut self.calls)),
        };
        self.chat.decode(event, out)
    }

    /// A stream tonic ended cleanly ends after its last chunk's finish
    /// reason; one without a finish reason was cut short.
    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.chat.eof(out)
    }
}

pub mod document;
