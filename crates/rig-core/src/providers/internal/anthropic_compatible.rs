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
