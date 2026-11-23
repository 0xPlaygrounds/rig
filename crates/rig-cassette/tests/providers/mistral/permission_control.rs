//! Mistral permission-control regression coverage.

use anyhow::Result;
use rig::agent::stream_to_stdout;
use rig::providers::openai::wire::MISTRAL;
use rig_test_support::cassette_models::OpenAiModels;

use crate::permission_control::{PermissionHook, ScratchFile};
use crate::support::assert_nonempty_response;

use super::TOOL_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn permission_control_prompt_example() -> Result<()> {
    let scratch = ScratchFile::new()?;

    let agent = rig::AgentBuilder::new(
        OpenAiModels::from_env_for(&MISTRAL)
            .expect("MISTRAL_API_KEY should be set")
            .completion(TOOL_MODEL),
    )
    .preamble("You are a helpful assistant that can read files using different methods.")
    .tool(scratch.head())
    .tool(scratch.tail())
    .build();

    let hook = PermissionHook::default();

    let _response = agent
        .prompt(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook.clone())
        .await?;

    let last = hook.last_result();
    anyhow::ensure!(last.as_deref() == Some("hello world"));
    anyhow::ensure!(hook.calls() == 2);

    Ok(())
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn permission_control_streaming_example() -> Result<()> {
    let scratch = ScratchFile::new()?;

    let agent = rig::AgentBuilder::new(
        OpenAiModels::from_env_for(&MISTRAL)
            .expect("MISTRAL_API_KEY should be set")
            .completion(TOOL_MODEL),
    )
    .preamble("You are a helpful assistant that can read files using different methods.")
    .tool(scratch.head())
    .tool(scratch.tail())
    .build();

    let hook = PermissionHook::default();

    let mut stream = agent
        .prompt(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook.clone())
        .stream();

    let final_response = stream_to_stdout(&mut stream).await?;
    let last = hook.last_result();
    assert_nonempty_response(&final_response.output());
    anyhow::ensure!(
        final_response
            .output()
            .to_ascii_lowercase()
            .contains("hello world"),
        "expected the streamed final response to mention the file content, got {:?}",
        final_response.output()
    );
    anyhow::ensure!(last.as_deref() == Some("hello world"));
    anyhow::ensure!(hook.calls() == 2);

    Ok(())
}
