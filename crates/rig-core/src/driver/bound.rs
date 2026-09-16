//! A wire and its socket.

use crate::http_client::{BoxedHttpClient, HttpClientExt};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{Capabilities, Wire};

/// A [`Wire`] bundled with the transport it speaks over.
///
/// This is the one type in rig-core that implements the consumer-facing
/// model traits ([`CompletionModel`](crate::completion::CompletionModel),
/// [`EmbeddingModel`](crate::embeddings::EmbeddingModel), and the rest): the
/// wire says what to send and how to read the reply, `H` carries the bytes,
/// and [`call`](super::call) / [`stream`](super::stream) join them.
///
/// `H` defaults to the erased [`BoxedHttpClient`], so `Bound<W>` means "any
/// transport" — the shape a host that owns one transport for many providers
/// holds. The default does not apply in expression position, so
/// [`Bound::new`] still infers `H` from its argument.
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

    /// Replace the wire, keeping the socket.
    ///
    /// The one forwarder for a wire's options: `bound.map_wire(|wire|
    /// wire.with_prompt_caching())` rather than a forwarding method per
    /// option on every bound type.
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

/// Bind any wire or provider config to a transport.
///
/// One blanket impl rather than a `bind` method on every config: whether a
/// provider can be bound is not a fact about the provider.
pub trait Bind: Sized {
    /// Bundle `self` with the socket it will speak over.
    fn bind<H>(self, http: H) -> Bound<Self, H> {
        Bound::new(self, http)
    }
}

impl<W: Sized> Bind for W {}
