//! A wire whose payload is the request and whose frames are the events:
//! what a local runtime, a scripted test or an in-process model speaks.
//! The transport is the runtime: it implements `Transport<Local<Op>>` and
//! answers a request with the operation's own events, which the driver's
//! sink canonicalizes as it does a provider's.
//!
//! ```
//! use rig_core::driver::Local;
//! use rig_core::operation::{Embedding, EmbeddingCapabilities};
//!
//! let wire = Local::<Embedding>::new("local")
//!     .with_id("all-minilm")
//!     .with_capabilities(EmbeddingCapabilities::new(8, 384));
//! # let _ = wire;
//! ```

use std::fmt;
use std::marker::PhantomData;

use crate::error::EncodeError;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{Decoder, Mode, Operation, Output, Sink, Wire, WireEvent};

/// A wire that carries its operation's own request and events. The name
/// is the provider descriptor name, the id the model the wire addresses
/// and the capabilities what a runtime accounts for.
pub struct Local<Op: Operation> {
    name: String,
    id: Option<String>,
    capabilities: Op::Capabilities,
}

impl<Op: Operation> Local<Op> {
    /// A wire named `name` that addresses no model and declares the
    /// operation's default capabilities.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            capabilities: Op::Capabilities::default(),
        }
    }

    /// The model id this wire addresses.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// What a runtime accounts for about this wire's model.
    pub fn with_capabilities(mut self, capabilities: Op::Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

impl<Op: Operation> Clone for Local<Op>
where
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

impl<Op: Operation> fmt::Debug for Local<Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local")
            .field("name", &self.name)
            .field("id", &self.id)
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
    type Frame = Op::Event;
    type Decoder = Passthrough<Op>;

    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn capabilities(&self) -> Op::Capabilities {
        self.capabilities.clone()
    }

    fn encode(&self, request: Op::Request, _mode: Mode) -> Result<Op::Request, EncodeError> {
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> Passthrough<Op> {
        Passthrough(PhantomData)
    }
}

/// The decoder of a [`Local`] wire: every frame is an event, pushed as it
/// is.
pub struct Passthrough<Op>(PhantomData<fn() -> Op>);

impl<Op: Operation> Decoder<Op, Op::Event> for Passthrough<Op> {
    type Event = Op::Event;

    fn classify(&self, event: Op::Event) -> WireEvent<Op::Event> {
        WireEvent::Known(event)
    }

    fn interpret(&mut self, event: Op::Event, out: &mut Output<Op>) {
        out.push(Ok(event));
    }
}
