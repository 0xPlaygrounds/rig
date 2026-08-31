//! Construction over the bundled transport, without naming it.
//!
//! rig-core is transport-agnostic — every constructor takes an
//! `H: HttpClientExt`. With the `reqwest` feature these inherent impls pin `H`
//! to [`reqwest::Client`], which is what lets
//! `rig::providers::openai::Client::from_env()` infer its transport.
//!
//! These were the `DefaultTransportClient` / `DefaultTransportBuilder` traits
//! of the old `rig-reqwest` crate. They only had to be traits because the impls
//! lived downstream of the types; here they are inherent, so callers no longer
//! import anything to construct a client.

use super::{
    Client, ClientBuilder, Provider, ProviderBuilder, ProviderClientError, ProviderFromEnv,
};
use crate::http_client::{self, DefaultHttp};
use crate::markers::Missing;

impl<Ext> Client<Ext, DefaultHttp>
where
    Ext: ProviderFromEnv,
    Ext::Builder: ProviderBuilder<Extension<DefaultHttp> = Ext> + Default,
{
    /// Construct a provider client over the bundled transport.
    pub fn new(
        api_key: impl Into<<Ext::Builder as ProviderBuilder>::ApiKey>,
    ) -> http_client::Result<Self> {
        Client::new_with(api_key, DefaultHttp::default())
    }

    /// Construct a provider client from the process's environment, over the
    /// bundled transport.
    pub fn from_env() -> Result<Self, ProviderClientError> {
        Ext::from_env_with(DefaultHttp::default())
    }

    /// Construct a provider client from an explicit provider-specific input,
    /// over the bundled transport.
    pub fn from_val(input: Ext::Input) -> Result<Self, ProviderClientError> {
        Ext::from_val_with(input, DefaultHttp::default())
    }
}

/// `build()` for a builder whose transport slot is still [`Missing`]:
/// substitutes the bundled transport.
///
/// rig-core's own `build()` exists only once `.http_client(..)` has been
/// called, so this is what makes
/// `provider::Client::builder().api_key(..).build()` resolve.
impl<ExtBuilder, Key> ClientBuilder<ExtBuilder, Key, Missing>
where
    ExtBuilder: ProviderBuilder<ApiKey = Key>,
    Key: super::ApiKey,
    ExtBuilder::Extension<DefaultHttp>: Provider,
{
    /// Build the client over the bundled transport.
    pub fn build(
        self,
    ) -> http_client::Result<Client<ExtBuilder::Extension<DefaultHttp>, DefaultHttp>> {
        self.http_client(DefaultHttp::default()).build()
    }
}
