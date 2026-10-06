//! The reassembler of a reply on either Cohere API.

use serde_json::Value;

use super::{NativeChat, Shape, shape};
use crate::providers::openai::wire::Chat;
use crate::wire::document::Reassemble;
use crate::wire::{Wire, WireFrame};

/// Hands each frame to the reassembler of the API whose shape it has, as
/// the decoder routes it. The reply's document is the native API's once a
/// native frame arrived, else the Compatibility API's.
#[derive(Default)]
pub struct Routed {
    native_api: <NativeChat as Wire>::Reassembler,
    compatibility_api: <Chat as Wire>::Reassembler,
    native_seen: bool,
}

impl Routed {
    /// Route between the two APIs' reassemblers.
    pub(super) fn new(
        native_api: <NativeChat as Wire>::Reassembler,
        compatibility_api: <Chat as Wire>::Reassembler,
    ) -> Self {
        Self {
            native_api,
            compatibility_api,
            native_seen: false,
        }
    }
}

impl Reassemble<WireFrame> for Routed {
    fn absorb(&mut self, frame: &WireFrame) {
        match shape(&frame.as_str()) {
            Shape::Native => {
                self.native_seen = true;
                self.native_api.absorb(frame);
            }
            Shape::Compatibility => self.compatibility_api.absorb(frame),
            Shape::Error => {}
        }
    }

    fn finish(self) -> Value {
        if self.native_seen {
            self.native_api.finish()
        } else {
            self.compatibility_api.finish()
        }
    }
}
