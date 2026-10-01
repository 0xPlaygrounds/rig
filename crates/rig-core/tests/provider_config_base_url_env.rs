//! A configuration read from the environment reports the base URL the
//! dialect's override variable names, normalized as the family's builder
//! normalizes it. Gemini names no override, so it keeps the public host.
//!
//! The variables are process-wide, which is why this test is its own binary.

#![allow(clippy::expect_used)]

use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::gemini::{self, GeminiConfig};
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::registry::ProviderConfig;

#[test]
fn a_configuration_from_the_environment_reports_the_overridden_base_url() {
    // SAFETY: this test binary has one test and no other threads read the
    // environment before it runs.
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
