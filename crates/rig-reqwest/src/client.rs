//! Construction of a provider client with the bundled [`crate::ReqwestClient`]
//! behind the erased default transport.
//!
//! rig-core's provider types default to
//! [`BoxedHttpClient`](rig_core::http_client::BoxedHttpClient), the erased
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

use rig_core::client::{
    Client, ClientBuilder, Provider, ProviderBuilder, ProviderClientError, ProviderFromEnv,
};
use rig_core::http_client::{self, BoxedHttpClient};
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
    fn new(api_key: impl Into<Self::ApiKey>) -> http_client::Result<Self>;

    /// Construct a provider client from the process's environment over the
    /// bundled transport.
    fn from_env() -> Result<Self, ProviderClientError>;

    /// Construct a provider client from an explicit provider-specific input
    /// over the bundled transport.
    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError>;
}

impl<Ext> DefaultTransportClient for Client<Ext, BoxedHttpClient>
where
    Ext: ProviderFromEnv,
    Ext::Builder: ProviderBuilder<Extension<BoxedHttpClient> = Ext> + Default,
{
    type ApiKey = <Ext::Builder as ProviderBuilder>::ApiKey;
    type Input = Ext::Input;

    fn new(api_key: impl Into<Self::ApiKey>) -> http_client::Result<Self> {
        Client::new_with(api_key, bundled())
    }

    fn from_env() -> Result<Self, ProviderClientError> {
        Ext::from_env_with(bundled())
    }

    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError> {
        Ext::from_val_with(input, bundled())
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
    fn build(self) -> http_client::Result<Self::Client>;
}

impl<ExtBuilder, Key> DefaultTransportBuilder for ClientBuilder<ExtBuilder, Key, Missing>
where
    ExtBuilder: ProviderBuilder<ApiKey = Key>,
    Key: rig_core::client::ApiKey,
    ExtBuilder::Extension<BoxedHttpClient>: Provider,
{
    type Client = Client<ExtBuilder::Extension<BoxedHttpClient>, BoxedHttpClient>;

    fn build(self) -> http_client::Result<Self::Client> {
        self.http_client(bundled()).build()
    }
}

/// The four construction spellings resolve with only the prelude and the
/// provider module in scope, and all name the same type.
///
/// ```no_run
/// use rig::prelude::*;
/// use rig::providers::openai;
///
/// fn takes(_: openai::Client) {}
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a: openai::Client = openai::Client::new("k")?;
/// let b = openai::Client::from_env()?;
/// let c = openai::Client::builder().api_key("k").build()?;
/// takes(a);
/// takes(b);
/// takes(c);
/// # Ok(())
/// # }
/// ```
#[cfg(doc)]
const _CONSTRUCTION_SPELLINGS: () = ();
