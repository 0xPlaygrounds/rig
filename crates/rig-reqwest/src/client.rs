//! Construction of a provider client with the bundled [`crate::ReqwestClient`]
//! behind the erased default transport.
//!
//! rig-core's provider types default to
//! [`BoxedHttpClient`], the erased
//! transport, so `openai::Client` names a concrete type in every
//! configuration; but rig-core deliberately depends on no transport, so it
//! cannot build one. These two traits are that value: implemented exactly once,
//! for the erased client, they construct it over a fresh `ReqwestClient`. That
//! single applicable impl is what lets `openai::Client::new(key)` infer the
//! transport in expression position, where a type alias default does not
//! apply. An inherent method could do the same, but only from inside rig-core,
//! which would have to know reqwest; a trait in this crate is the
//! orphan-rule-legal seam.
//!
//! Bring them into scope with `use rig::prelude::*` or
//! `use rig_reqwest::prelude::*`. To keep the concrete transport in the type
//! instead, use rig-core's `Client::new_with(key, ReqwestClient::default())`
//! or `.http_client(ReqwestClient::default()).build()`.

use rig_core::client::{Client, ClientBuilder, Provider, ProviderClientError};
use rig_core::http_client::BoxedHttpClient;
use rig_core::markers::Missing;

fn bundled() -> BoxedHttpClient {
    BoxedHttpClient::from(crate::ReqwestClient::default())
}

/// One-argument construction of a provider client over the erased default
/// transport, backed by the bundled `crate::ReqwestClient`: `new(api_key)`,
/// `from_env()`, `from_val(input)`.
///
/// `builder()` needs no trait — rig-core's `Client::builder()` already infers
/// its `Missing` transport slot; pair it with [`DefaultTransportBuilder`] to
/// `build()` without calling `.http_client(..)`.
pub trait DefaultTransportClient: Sized {
    /// The provider's API-key input (what [`ClientBuilder::api_key`] accepts).
    type ApiKey;
    /// The provider's explicit-input type for [`Self::from_val`].
    type Input;

    /// Construct a provider client over the bundled transport.
    fn new(api_key: impl Into<Self::ApiKey>) -> Result<Self, ProviderClientError>;

    /// Construct a provider client from the process's environment over the
    /// bundled transport.
    fn from_env() -> Result<Self, ProviderClientError>;

    /// Construct a provider client from an explicit provider-specific input
    /// over the bundled transport.
    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError>;
}

impl<P> DefaultTransportClient for Client<P, BoxedHttpClient>
where
    P: Provider,
{
    type ApiKey = P::ApiKey;
    type Input = P::EnvInput;

    fn new(api_key: impl Into<Self::ApiKey>) -> Result<Self, ProviderClientError> {
        Client::new_with(api_key, bundled())
    }

    fn from_env() -> Result<Self, ProviderClientError> {
        P::from_env(bundled())
    }

    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError> {
        P::from_val(input, bundled())
    }
}

/// `build()` for a [`ClientBuilder`] whose transport slot is still
/// [`Missing`]: substitutes the erased default backed by the bundled
/// `crate::ReqwestClient`.
///
/// rig-core's own `build()` exists only once `.http_client(..)` has been
/// called, so this trait is what makes
/// `provider::Client::builder().api_key(..).build()` resolve.
pub trait DefaultTransportBuilder {
    /// The built client type.
    type Client;

    /// Build the client over the bundled transport.
    fn build(self) -> Result<Self::Client, ProviderClientError>;
}

impl<P> DefaultTransportBuilder for ClientBuilder<P, Missing>
where
    P: Provider,
{
    type Client = Client<P, BoxedHttpClient>;

    fn build(self) -> Result<Self::Client, ProviderClientError> {
        self.http_client(bundled()).build()
    }
}

/// The construction spellings resolve with only the prelude and the
/// provider module in scope, and the transport-less ones all name the same
/// type; `new_with` keeps the concrete transport in the type.
///
/// ```no_run
/// use rig_core::providers::openai;
/// use rig_reqwest::prelude::*;
/// use rig_reqwest::ReqwestClient;
///
/// fn takes(_: openai::Client) {}
/// fn takes_reqwest(_: openai::Client<ReqwestClient>) {}
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a: openai::Client = openai::Client::new("k")?;
/// let b = openai::Client::from_env()?;
/// let c = openai::Client::builder().api_key("k").build()?;
/// let d = openai::Client::new_with("k", ReqwestClient::default())?;
/// takes(a);
/// takes(b);
/// takes(c);
/// takes_reqwest(d);
/// # Ok(())
/// # }
/// ```
#[cfg(doc)]
const _CONSTRUCTION_SPELLINGS: () = ();
