//! Copilot permission-control regression coverage.

use anyhow::Result;
use rig::agent::stream_to_stdout;

use crate::copilot::{LIVE_LIGHT_MODEL, live_client, with_copilot_cassette_result};
use crate::permission_control::{PermissionHook, ScratchFile};
use crate::support::assert_nonempty_response;

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_copilot_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let scratch = ScratchFile::new()?;

            let agent = rig::AgentBuilder::new(client.completion(LIVE_LIGHT_MODEL))
                .preamble(
                    "You are a helpful assistant that can read files using different methods.",
                )
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
            anyhow::ensure!(
                hook.calls() >= 2,
                "expected at least one skipped call followed by an allowed call"
            );
            Ok(())
        },
    )
    .await
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn permission_control_streaming_example() -> Result<()> {
    let scratch = ScratchFile::new()?;

    let agent = rig::AgentBuilder::new(live_client().await.completion(LIVE_LIGHT_MODEL))
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
    anyhow::ensure!(
        hook.calls() >= 2,
        "expected at least one skipped call followed by an allowed call"
    );

    Ok(())
}
