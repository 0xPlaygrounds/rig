//! Cassette-backed OpenRouter permission-control regression coverage.

use anyhow::Result;

use super::super::{TOOL_MODEL, support::with_openrouter_cassette_result};
use crate::permission_control::{PermissionHook, ScratchFile};

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_openrouter_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let scratch = ScratchFile::new()?;

            let agent = rig::AgentBuilder::new(client.completion(TOOL_MODEL))
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
        },
    )
    .await
}
