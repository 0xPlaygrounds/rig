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

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct CallData(u32);

impl crate::tool::ContextValue for CallData {
    const KEY: &'static str = "test.portable.call_data";
}

fn mutate_context(
    context: &mut ToolContext,
    arguments: serde_json::Value,
) -> WasmBoxedFuture<'_, Result<ToolOutput, ToolExecutionError>> {
    Box::pin(async move {
        assert_eq!(context.require::<CallData>()?, CallData(1));
        context.insert(CallData(99))?;
        context.insert_result(CallData(7))?;
        if arguments["pending"] == true {
            return std::future::pending().await;
        }
        if arguments["fail"] == true {
            return Err(ToolExecutionError::other("tool failed"));
        }
        Ok(ToolOutput::text("done"))
    })
}

#[test]
fn cancelling_inline_portable_execution_preserves_the_callers_context() {
    use futures::{FutureExt, task::noop_waker_ref};

    let tool = PortableDynamicTool::new_with_context(
        "mutate",
        "test context isolation",
        serde_json::json!({}),
        mutate_context,
    );
    let scope = Arc::new(42_u32);
    let mut context = ToolContext::new().with_scope(scope.clone());
    context.insert(CallData(1)).expect("inbound");
    context.insert_result(CallData(3)).expect("prior result");
    let before = context.clone();
    let mut run = Box::pin(tool.execute_with(&mut context, serde_json::json!({"pending": true})));
    let mut cx = std::task::Context::from_waker(noop_waker_ref());
    assert!(run.poll_unpin(&mut cx).is_pending());
    drop(run);
    assert_eq!(
        context, before,
        "cancellation must not consume caller data or publish a partial result"
    );
    assert!(Arc::ptr_eq(
        &context.scope::<u32>().expect("scope retained"),
        &scope
    ));
}

#[tokio::test]
async fn inline_portable_execution_publishes_only_result_changes() {
    let tool = PortableDynamicTool::new_with_context(
        "mutate",
        "test context isolation",
        serde_json::json!({}),
        mutate_context,
    );
    for fail in [false, true] {
        let scope = Arc::new(42_u32);
        let mut context = ToolContext::new().with_scope(scope.clone());
        context.insert(CallData(1)).expect("inbound");
        context.insert_result(CallData(3)).expect("prior result");
        let result = tool
            .execute_with(&mut context, serde_json::json!({"fail": fail}))
            .await;
        assert_eq!(result.is_err(), fail);
        assert_eq!(
            context.get::<CallData>().expect("inbound"),
            Some(CallData(1))
        );
        assert_eq!(
            context.result::<CallData>().expect("published result"),
            Some(CallData(7))
        );
        assert!(Arc::ptr_eq(
            &context.scope::<u32>().expect("scope retained"),
            &scope
        ));
    }
}
