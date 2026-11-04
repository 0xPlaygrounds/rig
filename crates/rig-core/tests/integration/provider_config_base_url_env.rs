//! A configuration read from the environment reports the base URL the
//! dialect's override variable names, normalized as the family's builder
//! normalizes it. Gemini names no override, so it keeps the public host. A
//! local server that needs no key is usable once its base URL is set.
//!
//! The variables are process-wide, which is why this test is its own binary.

#![allow(clippy::expect_used)]

use rig_core::catalog::{Connector, ModelStatus};
use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::gemini::{self, GeminiConfig};
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::registry::ProviderConfig;

#[test]
fn the_environment_sets_base_urls_and_configures_local_servers() {
    // SAFETY: this test binary has one test and no other threads read the
    // environment before it runs.
    unsafe {
        for name in [
            "OLLAMA_API_BASE_URL",
            "OLLAMA_API_KEY",
            "LLAMACPP_API_BASE_URL",
            "LLAMACPP_API_KEY",
        ] {
            std::env::remove_var(name);
        }
    }
    let usable = |vendor: &str| {
        let connector = Connector::default();
        let statuses = connector.statuses().into_iter();
        let mut theirs = statuses.filter(|(spec, _)| spec.provider.vendor() == vendor);
        theirs.any(|(_, status)| status == ModelStatus::Usable)
    };
    assert!(
        !usable("ollama") && !usable("llamacpp"),
        "nothing configures them"
    );
    // SAFETY: as above.
    unsafe { std::env::set_var("OLLAMA_API_BASE_URL", "http://127.0.0.1:9/") };
    assert!(usable("ollama") && !usable("llamacpp"));
    // A key alone configures one too.
    // SAFETY: as above.
    unsafe { std::env::set_var("LLAMACPP_API_KEY", "test") };
    assert!(usable("llamacpp"));
    // The daemon's OpenAI-compatible API, where its models connect; this
    // reader of the dialect asks for a key.
    // SAFETY: as above.
    unsafe { std::env::set_var("OLLAMA_API_KEY", "test") };
    let ollama = OpenAIConfig::from_env_with(&rig_core::providers::openai::wire::OLLAMA)
        .expect("the Ollama variable is set");
    assert_eq!(
        ProviderConfig::OpenAi(ollama).base_url(),
        "http://127.0.0.1:9/v1"
    );

    // SAFETY: as above.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        std::env::set_var("OPENAI_BASE_URL", "http://127.0.0.1:9/openai/v1");
        std::env::set_var("ANTHROPIC_API_KEY", "sk-test");
        std::env::set_var("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/anthropic/v1/");
        std::env::set_var("GEMINI_API_KEY", "test");
    }

    let openai =
        ProviderConfig::OpenAi(OpenAIConfig::from_env().expect("the OpenAI variables are set"));
    assert_eq!(openai.base_url(), "http://127.0.0.1:9/openai/v1");

    let anthropic = ProviderConfig::Anthropic(
        AnthropicConfig::from_env().expect("the Anthropic variables are set"),
    );
    assert_eq!(anthropic.base_url(), "http://127.0.0.1:9/anthropic");

    let gemini = ProviderConfig::Gemini(GeminiConfig::from_env().expect("the Gemini key is set"));
    assert_eq!(gemini.base_url(), gemini::BASE_URL);
}
