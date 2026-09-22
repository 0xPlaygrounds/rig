//! Pairs provider configuration with a transport.
//!
//! ```no_run
//! use rig_core::driver::Bind;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # fn example(http: impl rig_core::driver::Socket) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.responses(openai::GPT_5_2).bind(http);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use crate::http_client::{BoxedHttpClient, HttpClientExt};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{Capabilities, Wire};

/// A [`Wire`] paired with a transport, implementing the corresponding consumer
/// model traits. `H` defaults to [`BoxedHttpClient`] in type annotations;
/// [`Self::new`] infers it from the supplied transport.
#[derive(Clone, Debug, PartialEq)]
pub struct Bound<W, H = BoxedHttpClient> {
    /// What to send and how to read the reply.
    pub wire: W,
    /// The socket it speaks over.
    pub http: H,
}

impl<W, H> Bound<W, H> {
    /// Bind `wire` to `http`.
    pub fn new(wire: W, http: H) -> Self {
        Self { wire, http }
    }

    /// Transforms the wire while retaining the transport.
    pub fn map_wire<V>(self, map: impl FnOnce(W) -> V) -> Bound<V, H> {
        Bound {
            wire: map(self.wire),
            http: self.http,
        }
    }
}

impl<W: Wire, H> Bound<W, H> {
    /// The provider descriptor name.
    pub fn provider(&self) -> &str {
        self.wire.name()
    }

    /// What a runtime should account for about this wire.
    pub fn capabilities(&self) -> Capabilities<W> {
        self.wire.capabilities()
    }
}

impl<W, H> Bound<W, H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Erase the transport, so many providers can share one socket type in
    /// a table.
    pub fn boxed(self) -> Bound<W, BoxedHttpClient> {
        Bound {
            wire: self.wire,
            http: BoxedHttpClient::new(self.http),
        }
    }
}

/// Pairs any sized value with a transport. Model traits require a supported wire.
pub trait Bind: Sized {
    /// Bundle `self` with the socket it will speak over.
    fn bind<H>(self, http: H) -> Bound<Self, H> {
        Bound::new(self, http)
    }
}

impl<W: Sized> Bind for W {}
