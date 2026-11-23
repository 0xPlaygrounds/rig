use anyhow::Result;

use crate::permission_control::{PermissionHook, ScratchFile};
use crate::support::assert_nonempty_response;

use super::super::cassette_support::*;

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_llamacpp_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let scratch = ScratchFile::new()?;

            let agent = rig::AgentBuilder::new(client.clone().completion(CASSETTE_MODEL))
                .preamble(
                    "You are a helpful assistant that can read files using different methods.",
                )
                .tool(scratch.head())
                .tool(scratch.tail())
                .build();

            let hook = PermissionHook::default();

            let response = agent
                .prompt(
                    "Use the available tools to read test.txt now. \
                 Do not ask any follow-up questions; just read the file and report its content.",
                )
                .max_turns(5)
                .add_hook(hook.clone())
                .await?;

            assert_nonempty_response(&response.output());
            let last = hook.last_result();
            if let Some(last) = last {
                anyhow::ensure!(last == "hello world");
            }
            anyhow::ensure!(hook.calls() >= 1);
            Ok(())
        },
    )
    .await
}
