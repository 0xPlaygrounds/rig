//! The pass-through wire: the payload is the request and each frame is one
//! item of the operation's sink. It is what a local runtime, a test script
//! or an in-process model speaks, so its transport is the runtime itself.
//!
//! ```
//! use rig_core::driver::Local;
//! use rig_core::operation::{Embedding, EmbeddingCapabilities};
//! use rig_core::wire::Wire;
//!
//! let wire = Local::<Embedding>::new("local").with_capabilities(EmbeddingCapabilities::new(8, 3));
//! assert_eq!(wire.name(), "local");
//! assert_eq!(wire.capabilities().ndims, 3);
//! ```

use std::fmt;
use std::marker::PhantomData;

use crate::error::{EncodeError, ProviderError};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{Decoder, Mode, Operation, Output, Sink, Wire, WireEvent};

/// A wire whose payload is the request and whose frames are the events.
///
/// The transport is the runtime: it implements `Transport<Local<Op>>`,
/// takes the request and delivers the operation's events, each frame one
/// item of the sink: an event, or an in-band error the reply continues
/// past. A failure that ends the reply is the transport's own `Err`. The
/// items still pass through the operation's sink, so a completion
/// runtime's events are made canonical like any provider's.
pub struct Local<Op: Operation> {
    name: String,
    id: Option<String>,
    capabilities: Op::Capabilities,
}

impl<Op: Operation> Local<Op> {
    /// A wire named `name` (as records and telemetry name it), addressing
    /// no model id, with the operation's default capabilities.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            capabilities: Op::Capabilities::default(),
        }
    }

    /// What a runtime accounts for, such as an embedding width.
    pub fn with_capabilities(mut self, capabilities: Op::Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// The model id this wire addresses.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

impl<Op> Clone for Local<Op>
where
    Op: Operation,
    Op::Capabilities: Clone,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            id: self.id.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
}

impl<Op> fmt::Debug for Local<Op>
where
    Op: Operation,
    Op::Capabilities: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("name", &self.name)
            .field("id", &self.id)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl<Op> Wire for Local<Op>
where
    Op: Operation,
    Op::Capabilities: Clone + WasmCompatSend + WasmCompatSync,
{
    type Op = Op;
    type Payload = Op::Request;
    type Frame = Result<Op::Event, ProviderError>;
    type Decoder = Passthrough<Op>;

    fn name(&self) -> &str {
        &self.name
    }

    fn encode(&self, request: Op::Request, _mode: Mode) -> Result<Op::Request, EncodeError> {
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> Passthrough<Op> {
        Passthrough(PhantomData)
    }

    fn capabilities(&self) -> Op::Capabilities {
        self.capabilities.clone()
    }

    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }
}

/// The decoder of a [`Local`] wire: every frame is a sink item, pushed as
/// is.
pub struct Passthrough<Op>(PhantomData<fn() -> Op>);

impl<Op: Operation> Decoder<Op, Result<Op::Event, ProviderError>> for Passthrough<Op> {
    type Event = Result<Op::Event, ProviderError>;

    fn classify(&self, item: Self::Event) -> WireEvent<Self::Event> {
        WireEvent::Known(item)
    }

    fn interpret(&mut self, item: Self::Event, out: &mut Output<Op>) {
        out.push(item);
    }
}
