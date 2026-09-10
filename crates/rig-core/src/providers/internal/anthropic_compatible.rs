//! Shared base-URL resolution for providers exposing both OpenAI- and
//! Anthropic-compatible endpoints.

/// Describes how one provider maps its OpenAI-compatible endpoint onto its
/// Anthropic-compatible endpoint.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AnthropicBaseUrl {
    known_bases: &'static [(&'static str, &'static str)],
    openai_paths: &'static [&'static str],
    anthropic_path: &'static str,
}

impl AnthropicBaseUrl {
    pub(crate) const fn new(
        known_bases: &'static [(&'static str, &'static str)],
        openai_paths: &'static [&'static str],
        anthropic_path: &'static str,
    ) -> Self {
        Self {
            known_bases,
            openai_paths,
            anthropic_path,
        }
    }

    /// Read the dedicated Anthropic override first, falling back to the
    /// provider's general base URL only when it can be mapped safely.
    pub(crate) fn resolve_from_env(
        self,
        primary_env: &'static str,
        fallback_env: &'static str,
    ) -> crate::client::ProviderClientResult<Option<String>> {
        let primary = crate::client::optional_env_var(primary_env)?;
        let fallback = crate::client::optional_env_var(fallback_env)?;

        Ok(self.resolve(primary.as_deref(), fallback.as_deref()))
    }

    pub(crate) fn resolve(self, primary: Option<&str>, fallback: Option<&str>) -> Option<String> {
        primary
            .map(str::to_owned)
            .or_else(|| fallback.and_then(|base_url| self.normalize(base_url)))
    }

    /// Preserve an explicitly Anthropic-shaped URL, map canonical provider
    /// endpoints exactly, or rewrite a recognized OpenAI-compatible path on a
    /// custom host. Unknown paths are not guessed.
    pub(crate) fn normalize(self, base_url: &str) -> Option<String> {
        if base_url.contains("/anthropic") {
            return Some(base_url.to_owned());
        }

        let trimmed = base_url.trim_end_matches('/');
        if let Some((_, anthropic_base)) = self
            .known_bases
            .iter()
            .find(|(openai_base, _)| *openai_base == trimmed)
        {
            return Some((*anthropic_base).to_owned());
        }

        let mut url = url::Url::parse(base_url).ok()?;
        if !self.openai_path(url.path()) {
            return None;
        }
        url.set_path(self.anthropic_path);
        Some(url.to_string())
    }

    fn openai_path(self, path: &str) -> bool {
        self.openai_paths.contains(&path)
    }
}

/// Generates the client scaffolding shared by providers that expose both an
/// OpenAI-compatible and an Anthropic-compatible endpoint: the two provider
/// types, `Client`/`ClientBuilder` aliases for both dialects, their
/// `Provider` impls (env construction included; the Anthropic one resolved
/// through the module's `ANTHROPIC_BASE_URLS` rule), and the Anthropic side's
/// `HasCompletion` and `AnthropicCompatibleProvider` impls.
///
/// The OpenAI-side `Has*` impls and `OpenAICompatibleProvider` impl stay in
/// the provider module: they are where providers genuinely differ (extra
/// capabilities, request preparation, response-format support).
macro_rules! impl_dual_dialect_provider {
    (
        provider = $provider:ident,
        anthropic_provider = $anthropic_provider:ident,
        client_input = $client_input:ty,
        name = $name:literal,
        api_key_env = $api_key_env:literal,
        base_url = $base_url:expr,
        base_url_env = $base_url_env:literal,
        anthropic_provider_name = $anthropic_name:literal,
        anthropic_base_url = $anthropic_base_url:expr,
        anthropic_base_url_env = $anthropic_base_url_env:literal $(,)?
    ) => {
        /// The OpenAI-compatible dialect of this provider.
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $provider;

        /// The Anthropic-compatible dialect of this provider.
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $anthropic_provider;

        pub type Client<H = $crate::http_client::BoxedHttpClient> =
            $crate::client::Client<$provider, H>;
        pub type ClientBuilder<H = $crate::markers::Missing> =
            $crate::client::ClientBuilder<$provider, H>;

        pub type AnthropicClient<H = $crate::http_client::BoxedHttpClient> =
            $crate::client::Client<$anthropic_provider, H>;
        pub type AnthropicClientBuilder<H = $crate::markers::Missing> =
            $crate::client::ClientBuilder<$anthropic_provider, H>;

        impl $crate::client::Provider for $provider {
            const NAME: &'static str = $name;
            const BASE_URL: &'static str = $base_url;
            const VERIFY_PATH: &'static str = "/models";
            type ApiKey = $crate::client::BearerAuth;
            type Config = ();
            type EnvInput = $client_input;

            fn build(_: (), _: &$crate::client::BearerAuth) -> $crate::http_client::Result<Self> {
                Ok($provider)
            }

            fn from_env<H: $crate::http_client::HttpClientExt>(
                http: H,
            ) -> $crate::client::ProviderClientResult<Client<H>> {
                Client::from_env_api_key($api_key_env, Some($base_url_env), http)
            }

            fn from_val<H: $crate::http_client::HttpClientExt>(
                input: $client_input,
                http: H,
            ) -> $crate::client::ProviderClientResult<Client<H>> {
                Client::new_with(input, http)
            }
        }

        impl $crate::client::Provider for $anthropic_provider {
            const NAME: &'static str = $name;
            const BASE_URL: &'static str = $anthropic_base_url;
            const VERIFY_PATH: &'static str = "/v1/models";
            type ApiKey = $crate::providers::anthropic::client::AnthropicKey;
            type Config = $crate::providers::anthropic::client::AnthropicConfig;
            type EnvInput = String;

            fn build(_: Self::Config, _: &Self::ApiKey) -> $crate::http_client::Result<Self> {
                Ok($anthropic_provider)
            }

            fn finish<H>(
                &self,
                builder: $crate::client::ClientBuilder<Self, H>,
            ) -> $crate::http_client::Result<$crate::client::ClientBuilder<Self, H>> {
                $crate::providers::anthropic::client::finish_anthropic_builder(builder)
            }

            fn from_env<H: $crate::http_client::HttpClientExt>(
                http: H,
            ) -> $crate::client::ProviderClientResult<AnthropicClient<H>> {
                let api_key = $crate::client::required_env_var($api_key_env)?;
                let mut builder =
                    AnthropicClient::<$crate::markers::Missing>::builder().api_key(api_key);
                if let Some(base_url) =
                    ANTHROPIC_BASE_URLS.resolve_from_env($anthropic_base_url_env, $base_url_env)?
                {
                    builder = builder.base_url(base_url);
                }
                builder.http_client(http).build()
            }

            fn from_val<H: $crate::http_client::HttpClientExt>(
                input: String,
                http: H,
            ) -> $crate::client::ProviderClientResult<AnthropicClient<H>> {
                AnthropicClient::new_with(input, http)
            }
        }

        impl $crate::client::HasCompletion for $anthropic_provider {
            type Model<H>
                = $crate::providers::anthropic::completion::GenericCompletionModel<
                $anthropic_provider,
                H,
            >
            where
                H: $crate::client::ModelTransport;

            fn completion_model<H: $crate::client::ModelTransport>(
                client: &AnthropicClient<H>,
                model: String,
            ) -> Self::Model<H> {
                $crate::providers::anthropic::completion::GenericCompletionModel::new(
                    client.clone(),
                    model,
                )
            }
        }

        impl $crate::providers::anthropic::completion::AnthropicCompatibleProvider
            for $anthropic_provider
        {
            const PROVIDER_NAME: &'static str = $anthropic_name;

            fn default_max_tokens(_model: &str) -> Option<u64> {
                Some(4096)
            }
        }
    };
}

pub(crate) use impl_dual_dialect_provider;

#[cfg(test)]
mod tests;
