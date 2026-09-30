//! The pass-through wire: the payload is the request and each frame is one
//! step of the reply. It is what a local runtime or an in-process model
//! speaks, so its transport is the runtime itself.
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
//!
//! A completion's events come only from its writer's part handles, so there
//! is no local completion wire:
//!
//! ```compile_fail
//! use rig_core::driver::Local;
//! use rig_core::operation::Completion;
//!
//! let wire = Local::<Completion>::new("scripted");
//! ```

use std::fmt;
use std::marker::PhantomData;

use crate::error::{EncodeError, ProviderError};
use crate::streaming::UnknownPayload;
use crate::wire::{
    Capabilities, Decoder, Descriptor, Flow, Free, Mode, Operation, Out, Wire, WireEvent,
};

/// A wire whose payload is the request and whose frames are the reply's
/// steps, for an operation whose events the runtime builds itself.
///
/// The transport is the runtime: it implements `Transport<Local<Op>>`,
/// takes the request and delivers the reply's [`Step`]s, ending with the
/// operation's end. A runtime that stops without it leaves the reply
/// truncated. A failure that ends the reply is the transport's own `Err`.
pub struct Local<Op> {
    name: String,
    id: Option<String>,
    capabilities: Capabilities,
    op: PhantomData<fn() -> Op>,
}

/// One step of a local runtime's reply.
pub enum Step<Op: Operation> {
    /// An event of the reply.
    Event(Op::Event),
    /// A payload the runtime does not model.
    Unknown(UnknownPayload),
    /// The runtime's end of the reply.
    End(Op::End),
}

impl<Op: Operation<Emit = Free>> Local<Op> {
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

/// Declaring a width sets the width the capabilities report and checks every
/// reply against it; the runtime is asked for nothing.
impl crate::embeddings::EmbeddingWidth for Local<crate::operation::Embedding> {
    fn with_ndims(mut self, ndims: usize) -> Self {
        self.capabilities.ndims = ndims;
        self.capabilities.declared = Some(ndims);
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

impl<Op: Operation<Emit = Free>> Wire for Local<Op> {
    type Op = Op;
    type Payload = Op::Request;
    type Frame = Step<Op>;
    type Decoder<'id> = Steps;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(&self.name)
            .model(self.id.as_deref())
            .capabilities(self.capabilities)
    }

    fn encode(&self, request: Op::Request, _mode: Mode) -> Result<Op::Request, EncodeError> {
        Ok(request)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        Steps
    }
}

/// The decoder of a [`Local`] wire: every step is written as it is.
#[derive(Debug, Default)]
pub struct Steps;

impl<'id, Op: Operation<Emit = Free>> Decoder<'id, Op, Step<Op>> for Steps {
    type Event = Step<Op>;

    fn classify(&self, step: Step<Op>) -> WireEvent<Step<Op>> {
        WireEvent::Known(step)
    }

    fn decode(&mut self, step: Step<Op>, mut out: Out<'id, Op>) -> Result<Flow, ProviderError> {
        Ok(match step {
            Step::Event(event) => {
                out.event(event);
                Flow::More
            }
            Step::Unknown(payload) => {
                out.unknown(payload);
                Flow::More
            }
            Step::End(end) => out.end(end),
        })
    }
}
