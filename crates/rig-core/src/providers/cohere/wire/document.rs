//! The reassembler of a reply on either Cohere API.

use serde_json::Value;

use super::{CohereEvent, NativeChat};
use crate::providers::openai::wire::Chat;
use crate::wire::Wire;
use crate::wire::document::Reassemble;

/// Hands each event to the reassembler of the API the decoder routed it
/// to. The reply's document is the native API's once a native event
/// arrived, else the Compatibility API's.
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

impl crate::wire::document::Serves<crate::operation::Completion> for Routed {}

impl Reassemble<CohereEvent> for Routed {
    fn absorb(&mut self, event: &CohereEvent) {
        match event {
            CohereEvent::Native(event) => {
                self.native_seen = true;
                self.native_api.absorb(event);
            }
            CohereEvent::Compatibility(event) => self.compatibility_api.absorb(event),
            CohereEvent::Failure(_) => {}
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
