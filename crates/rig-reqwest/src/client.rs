//! Default-transport conveniences: construct any rig-core provider client
//! with the bundled [`crate::ReqwestClient`] without naming a transport.
//!
//! rig-core deliberately has no default transport — every constructor there
//! takes an `H: HttpClientExt`. These two traits are implemented exactly once,
//! for `Client<Ext, crate::ReqwestClient>`, which is what lets
//! `rig::providers::openai::Client::from_env()` infer `H` at the call site
//! (a single applicable impl). Bring them into scope with `use rig::prelude::*`
//! or `use rig_reqwest::prelude::*`.

use rig_core::client::{
    Client, ClientBuilder, Provider, ProviderBuilder, ProviderClientError, ProviderFromEnv,
};
use rig_core::http_client;
use rig_core::markers::Missing;

/// One-argument construction of a provider client over the bundled
/// `crate::ReqwestClient`: `new(api_key)`, `from_env()`, `from_val(input)`.
///
/// `builder()` needs no trait — rig-core's `Client::builder()` already infers
/// its `Missing` transport slot; pair it with [`DefaultTransportBuilder`] to
/// `build()` without calling `.http_client(..)`.
pub trait DefaultTransportClient: Sized {
    /// The provider's API-key input (what [`ClientBuilder::api_key`] accepts).
    type ApiKey;
    /// The provider's explicit-input type for [`Self::from_val`].
    type Input;

    /// Construct a provider client with the bundled `crate::ReqwestClient`.
    fn new(api_key: impl Into<Self::ApiKey>) -> http_client::Result<Self>;

    /// Construct a provider client from the process's environment with the
    /// bundled `crate::ReqwestClient`.
    fn from_env() -> Result<Self, ProviderClientError>;

    /// Construct a provider client from an explicit provider-specific input
    /// with the bundled `crate::ReqwestClient`.
    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError>;
}

impl<Ext> DefaultTransportClient for Client<Ext, crate::ReqwestClient>
where
    Ext: ProviderFromEnv,
    Ext::Builder: ProviderBuilder<Extension<crate::ReqwestClient> = Ext> + Default,
{
    type ApiKey = <Ext::Builder as ProviderBuilder>::ApiKey;
    type Input = Ext::Input;

    fn new(api_key: impl Into<Self::ApiKey>) -> http_client::Result<Self> {
        Client::new_with(api_key, crate::ReqwestClient::default())
    }

    fn from_env() -> Result<Self, ProviderClientError> {
        Ext::from_env_with(crate::ReqwestClient::default())
    }

    fn from_val(input: Self::Input) -> Result<Self, ProviderClientError> {
        Ext::from_val_with(input, crate::ReqwestClient::default())
    }
}

/// `build()` for a [`ClientBuilder`] whose transport slot is still
/// [`Missing`]: substitutes the bundled `crate::ReqwestClient`.
///
/// rig-core's own `build()` exists only once `.http_client(..)` has been
/// called, so this trait is what makes
/// `provider::Client::builder().api_key(..).build()` resolve.
pub trait DefaultTransportBuilder {
    /// The built client type.
    type Client;

    /// Build the client with the bundled `crate::ReqwestClient`.
    fn build(self) -> http_client::Result<Self::Client>;
}

impl<ExtBuilder, Key> DefaultTransportBuilder for ClientBuilder<ExtBuilder, Key, Missing>
where
    ExtBuilder: ProviderBuilder<ApiKey = Key>,
    Key: rig_core::client::ApiKey,
    ExtBuilder::Extension<crate::ReqwestClient>: Provider,
{
    type Client = Client<ExtBuilder::Extension<crate::ReqwestClient>, crate::ReqwestClient>;

    fn build(self) -> http_client::Result<Self::Client> {
        self.http_client(crate::ReqwestClient::default()).build()
    }
}
