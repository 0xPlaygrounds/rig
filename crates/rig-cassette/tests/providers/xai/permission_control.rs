//! xAI permission-control regression coverage.

use anyhow::Result;
use rig::agent::stream_to_stdout;
use rig::providers::xai;

use super::support::with_xai_cassette_result;
use crate::permission_control::{PermissionHook, ScratchFile};
use crate::support::assert_nonempty_response;

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let scratch = ScratchFile::new()?;

            let agent = rig::AgentBuilder::new(client.completion(xai::GROK_4))
                .preamble("You are a helpful assistant that can read files using different methods.")
                .tool(scratch.head())
                .tool(scratch.tail())
                .additional_params(serde_json::json!({ "store": false }))
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
        },
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| async move {
            let scratch = ScratchFile::new()?;

            let agent = rig::AgentBuilder::new(client.completion(xai::GROK_4))
                .preamble("You are a helpful assistant that can read files using different methods.")
                .tool(scratch.head())
                .tool(scratch.tail())
                .additional_params(serde_json::json!({ "store": false }))
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
        },
    )
    .await
}
