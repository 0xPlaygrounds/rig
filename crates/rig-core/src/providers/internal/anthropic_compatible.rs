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
/// OpenAI-compatible and an Anthropic-compatible endpoint: the marker/builder
/// structs, `Client`/`ClientBuilder` aliases for both dialects, `Provider`,
/// `DebugExt`, and `AnthropicCompatibleProvider` impls, builder wiring, and
/// env-driven `ProviderClient` impls (the Anthropic one resolved through the
/// module's `ANTHROPIC_BASE_URLS` rule).
///
/// The OpenAI-side capabilities and `OpenAICompatibleProvider` impl stay in
/// the provider module: they are where providers genuinely differ (extra
/// capabilities, request preparation, response-format support).
macro_rules! impl_dual_dialect_provider {
    (
        ext = $ext:ident,
        builder = $builder:ident,
        anthropic_ext = $anthropic_ext:ident,
        anthropic_builder = $anthropic_builder:ident,
        client_input = $client_input:ty,
        api_key_env = $api_key_env:literal,
        base_url = $base_url:expr,
        base_url_env = $base_url_env:literal,
        anthropic_provider_name = $anthropic_name:literal,
        anthropic_base_url = $anthropic_base_url:expr,
        anthropic_base_url_env = $anthropic_base_url_env:literal $(,)?
    ) => {
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $ext;

        #[derive(Debug, Default, Clone, Copy)]
        pub struct $builder;

        #[derive(Debug, Default, Clone)]
        pub struct $anthropic_builder {
            anthropic: $crate::providers::anthropic::client::AnthropicBuilder,
        }

        #[derive(Debug, Default, Clone, Copy)]
        pub struct $anthropic_ext;

        pub type Client<H = reqwest::Client> = $crate::client::Client<$ext, H>;
        pub type ClientBuilder<H = $crate::markers::Missing> =
            $crate::client::ClientBuilder<$builder, $crate::client::BearerAuth, H>;

        pub type AnthropicClient<H = reqwest::Client> = $crate::client::Client<$anthropic_ext, H>;
        pub type AnthropicClientBuilder<H = $crate::markers::Missing> = $crate::client::ClientBuilder<
            $anthropic_builder,
            $crate::providers::anthropic::client::AnthropicKey,
            H,
        >;

        impl $crate::client::Provider for $ext {
            type Builder = $builder;

            const VERIFY_PATH: &'static str = "/models";
        }

        impl $crate::client::Provider for $anthropic_ext {
            type Builder = $anthropic_builder;

            const VERIFY_PATH: &'static str = "/v1/models";
        }

        impl $crate::client::DebugExt for $ext {}
        impl $crate::client::DebugExt for $anthropic_ext {}

        $crate::client::impl_capabilities!(
            $anthropic_ext,
            completion =
                $crate::providers::anthropic::completion::GenericCompletionModel<$anthropic_ext, H>,
        );

        $crate::client::impl_default_provider_builder!(
            $builder => $ext,
            api_key = $crate::client::BearerAuth,
            base_url = $base_url,
        );
        $crate::providers::anthropic::client::impl_anthropic_compatible_builder!(
            $anthropic_builder => $anthropic_ext,
            base_url = $anthropic_base_url,
        );

        impl $crate::providers::anthropic::completion::AnthropicCompatibleProvider
            for $anthropic_ext
        {
            const PROVIDER_NAME: &'static str = $anthropic_name;

            fn default_max_tokens(_model: &str) -> Option<u64> {
                Some(4096)
            }
        }

        $crate::client::impl_provider_client!(
            Client,
            input = $client_input,
            api_key_env = $api_key_env,
            base_url_env = $base_url_env,
        );

        $crate::client::impl_provider_client!(
            AnthropicClient,
            input = String,
            api_key_env = $api_key_env,
            base_url = ANTHROPIC_BASE_URLS.resolve_from_env($anthropic_base_url_env, $base_url_env)?,
        );
    };
}

pub(crate) use impl_dual_dialect_provider;

#[cfg(test)]
mod tests {
    use super::AnthropicBaseUrl;

    const RULE: AnthropicBaseUrl = AnthropicBaseUrl::new(
        &[(
            "https://api.example.com/v1",
            "https://api.example.com/anthropic",
        )],
        &["/v1", "/v1/"],
        "/anthropic",
    );

    #[test]
    fn maps_known_and_custom_openai_bases() {
        assert_eq!(
            RULE.normalize("https://api.example.com/v1/").as_deref(),
            Some("https://api.example.com/anthropic")
        );
        assert_eq!(
            RULE.normalize("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn primary_wins_and_unknown_fallback_paths_are_ignored() {
        assert_eq!(
            RULE.resolve(
                Some("https://primary.example.com/anthropic"),
                Some("https://proxy.example.com/v1")
            )
            .as_deref(),
            Some("https://primary.example.com/anthropic")
        );
        assert_eq!(
            RULE.resolve(None, Some("https://proxy.example.com/api")),
            None
        );
    }
}
