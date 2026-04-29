use anyhow::Result;
use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;

use crate::support::assert_nonempty_response;

#[derive(Deserialize)]
struct AddArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = AddArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers together. Returns x + y.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "First number" },
                    "y": { "type": "number", "description": "Second number" }
                },
                "required": ["x", "y"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Clone)]
struct ArgRewriteHook {
    rewritten_args: Arc<Mutex<Option<String>>>,
    result_args: Arc<Mutex<Option<String>>>,
}

impl<M: CompletionModel> PromptHook<M> for ArgRewriteHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        let mut parsed: serde_json::Value =
            serde_json::from_str(args).expect("args should be valid JSON");
        parsed["x"] = json!(100);
        parsed["y"] = json!(200);
        let new_args = serde_json::to_string(&parsed).expect("should serialize");
        *self.rewritten_args.lock().expect("lock") = Some(new_args.clone());
        ToolCallHookAction::continue_with(new_args)
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        _result: &str,
    ) -> HookAction {
        *self.result_args.lock().expect("lock") = Some(args.to_string());
        HookAction::cont()
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn arg_rewrite_hook_modifies_tool_args() -> Result<()> {
    let agent = providers::openai::Client::from_env()
        .expect("client should build")
        .agent(providers::openai::GPT_4O_MINI)
        .preamble("You have an add tool. Use it when asked to add numbers.")
        .tool(Adder)
        .build();

    let rewritten_args = Arc::new(Mutex::new(None));
    let result_args = Arc::new(Mutex::new(None));
    let hook = ArgRewriteHook {
        rewritten_args: rewritten_args.clone(),
        result_args: result_args.clone(),
    };

    let response = agent
        .prompt("What is 1 + 2? Use the add tool.")
        .max_turns(3)
        .with_hook(hook)
        .await?;

    assert_nonempty_response(&response);

    let rewritten = rewritten_args.lock().expect("lock").clone();
    anyhow::ensure!(
        rewritten.is_some(),
        "on_tool_call should have been invoked and rewritten args"
    );
    let rewritten = rewritten.unwrap();
    anyhow::ensure!(
        rewritten.contains("100") && rewritten.contains("200"),
        "rewritten args should contain 100 and 200, got: {rewritten}"
    );

    let result_args_val = result_args.lock().expect("lock").clone();
    anyhow::ensure!(
        result_args_val.is_some(),
        "on_tool_result should have been invoked"
    );
    let result_args_val = result_args_val.unwrap();
    anyhow::ensure!(
        result_args_val.contains("100") && result_args_val.contains("200"),
        "on_tool_result should receive the rewritten args, got: {result_args_val}"
    );

    anyhow::ensure!(
        response.contains("300"),
        "response should contain 300 (100+200), got: {response}"
    );

    Ok(())
}
