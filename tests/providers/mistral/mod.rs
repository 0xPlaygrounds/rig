mod agent;
mod agent_tool_sessions;
#[cfg(feature = "derive")]
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod permission_control;
mod request_hook;
mod streaming;
mod streaming_tools;
mod support;
mod transcription;
mod typed_prompt_tools;

pub(super) const DEFAULT_MODEL: &str = "mistral-small-latest";
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;

/// Live provider config resolved from the environment, for `#[ignore]`d tests
/// that talk to the real API.
#[allow(dead_code)]
pub(super) fn live(model: &str) -> rig::provider::ProviderConfig {
    rig::provider::ProviderConfig::Mistral(
        rig::providers::mistral::functions::Config::from_env(model)
            .expect("provider config should build from env"),
    )
}
