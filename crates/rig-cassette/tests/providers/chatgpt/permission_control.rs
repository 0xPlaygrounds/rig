//! ChatGPT permission-control regression coverage.

use anyhow::Result;
use rig::agent::stream_to_stdout;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::permission_control::{PermissionHook, ScratchFile};
use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn permission_control_prompt_example() -> Result<()> {
    let scratch = ScratchFile::new()?;

    let agent = rig::AgentBuilder::new(live_client().await.completion(LIVE_MODEL))
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
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn permission_control_streaming_example() -> Result<()> {
    let scratch = ScratchFile::new()?;

    let agent = rig::AgentBuilder::new(live_client().await.completion(LIVE_MODEL))
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
