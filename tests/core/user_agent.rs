//! The default `user-agent` a gateway that asks for one is sent.

use rig::completion::CompletionRequest;
use rig::providers::{chatgpt, openai};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};

#[test]
fn the_default_user_agent_names_the_crate_version_and_this_host() -> anyhow::Result<()> {
    let wire = openai::responses_api::wire::Responses::new(
        openai::wire::OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-key"),
        "gpt-5.4",
    );
    let prepared = Completion::prepare(CompletionRequest::new("hi"), &wire.describe())?;
    let encoded = wire.encode(prepared, Mode::Unary)?;
    let expected = format!(
        "rig/{} ({} {}; rig)",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    anyhow::ensure!(
        encoded
            .request
            .headers()
            .get("user-agent")
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str())
    );
    Ok(())
}
