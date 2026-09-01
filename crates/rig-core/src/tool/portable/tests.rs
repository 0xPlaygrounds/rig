use std::convert::Infallible;

use serde::{Deserialize, Serialize};

use super::*;

#[derive(Deserialize)]
struct AddArgs {
    left: i64,
    right: i64,
}

#[derive(Serialize)]
struct Sum {
    value: i64,
}

struct Add;

impl PortableTool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = Sum;
    type Error = Infallible;

    fn description(&self) -> String {
        "Add two integers".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(Sum {
            value: arguments.left + arguments.right,
        })
    }
}

#[tokio::test]
async fn portable_tools_execute_without_runtime_context() {
    let output = Add.call(AddArgs { left: 2, right: 3 }).await;
    let Ok(output) = output;
    assert_eq!(output.value, 5);
    assert_eq!(portable_tool_definition(&Add).name, "add");
}

#[tokio::test]
async fn portable_dynamic_tools_receive_owned_arguments() {
    let tool = PortableDynamicTool::new(
        "echo",
        "Echo a JSON value",
        serde_json::json!({"type": "object"}),
        |arguments| Box::pin(async move { Ok(ToolOutput::json(arguments)) }),
    );

    let arguments = serde_json::json!({"value": "hello"});
    let output = tool.execute(arguments.clone()).await;
    assert!(output.is_ok());
    let Ok(output) = output else {
        return;
    };
    assert_eq!(output.as_json(), Some(&arguments));
}
