//! The pass-through wire: the payload is the request and each frame is one
//! item of the reply. It is what a local runtime, a test script or an
//! in-process model speaks, so its transport is the runtime itself.
//!
//! ```
//! use rig_core::driver::Local;
//! use rig_core::operation::Embedding;
//! use rig_core::wire::{Capabilities, Wire};
//!
//! let wire = Local::<Embedding>::new("local").with_capabilities(Capabilities::embedding(8, 3));
//! assert_eq!(wire.describe().name, "local");
//! assert_eq!(wire.describe().capabilities.ndims, 3);
//! ```

use std::fmt;
use std::marker::PhantomData;

use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::message::{AssistantContent, Issuer, Message};
use crate::wire::{Capabilities, Decoder, Descriptor, Mode, Operation, Out, Wire, WireEvent};

/// A wire whose payload is the request and whose frames are the events.
///
/// The transport is the runtime: it implements `Transport<Local<Op>>`,
/// takes the request and delivers the operation's events, each frame one
/// item of the reply: an event, or an in-band error the reply continues
/// past. A failure that ends the reply is the transport's own `Err`. The
/// items still pass through the operation's fold, so a completion
/// runtime's events are made canonical like any provider's.
pub struct Local<Op> {
    name: String,
    id: Option<String>,
    capabilities: Capabilities,
    op: PhantomData<fn() -> Op>,
}

impl<Op> Local<Op> {
    /// A wire named `name` (as records and telemetry name it), addressing
    /// no model id, with default capabilities.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            capabilities: Capabilities::default(),
            op: PhantomData,
        }
    }

    /// What a runtime accounts for, such as an embedding width.
    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// The model id this wire addresses.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

impl<Op> Clone for Local<Op> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            id: self.id.clone(),
            capabilities: self.capabilities,
            op: PhantomData,
        }
    }
}

impl<Op> fmt::Debug for Local<Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("name", &self.name)
            .field("id", &self.id)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl<Op: Operation> Wire for Local<Op> {
    type Op = Op;
    type Payload = Op::Request;
    type Frame = Result<Op::Event, ProviderError>;
    type Decoder = Passthrough;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(&self.name)
            .model(self.id.as_deref())
            .capabilities(self.capabilities)
    }

    fn encode(&self, mut request: Op::Request, _mode: Mode) -> Result<Op::Request, EncodeError> {
        // A completion replays only the reasoning this runtime issued, as
        // every completion wire's encode scopes it.
        let any: &mut dyn std::any::Any = &mut request;
        if let Some(request) = any.downcast_mut::<CompletionRequest>() {
            let issuers = [Issuer::from(self.name.clone())];
            for message in request.chat_history.iter_mut() {
                if let Message::Assistant { content, .. } = message
                    && let Some(kept) = content.clone().filter(|part| match part {
                        AssistantContent::Reasoning(reasoning) => {
                            reasoning.open_for(&issuers).is_some()
                        }
                        _ => true,
                    })
                {
                    *content = kept;
                }
            }
            request.chat_history = request
                .chat_history
                .clone()
                .filter(|message| message.replays_to(&issuers))
                .ok_or_else(|| {
                    EncodeError::request(
                        "no message is left once reasoning another service issued is left out",
                    )
                })?;
        }
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> Passthrough {
        Passthrough
    }
}

/// The decoder of a [`Local`] wire: every frame is an item, pushed as is.
#[derive(Debug, Default)]
pub struct Passthrough;

impl<Op: Operation> Decoder<Op, Result<Op::Event, ProviderError>> for Passthrough {
    type Event = Result<Op::Event, ProviderError>;

    fn classify(&self, item: Self::Event) -> WireEvent<Self::Event> {
        WireEvent::Known(item)
    }

    fn interpret(&mut self, item: Self::Event, out: &mut Out<'_, Op>) {
        out.push(item);
    }
}
