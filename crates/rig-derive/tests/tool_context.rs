//! Tests for host-only `ToolContext` parameters in `#[rig_tool]` functions.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_agent::tool::{ContextualTool, ToolContext};
use rig_core::tool::Tool;
use rig_derive::rig_tool;

#[derive(Clone)]
struct Offset(i32);

#[derive(Clone)]
struct Prefix(String);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Invocation(&'static str);

#[rig_tool(
    description = "Add two numbers using a host-provided offset",
    params(left = "Left operand", right = "Right operand")
)]
fn sync_context_in_the_middle(
    left: i32,
    #[rig(context)] context: &mut ToolContext,
    right: i32,
) -> Result<i32, rig_core::tool::ToolExecutionError> {
    let offset = context.require::<Offset>()?.0;
    context.insert_result(Invocation("sync"));
    Ok(left + right + offset)
}

#[tokio::test]
async fn required_context_uses_the_standard_tool_error_path() {
    let error = SyncContextInTheMiddle
        .call(
            &mut ToolContext::new(),
            SyncContextInTheMiddleParameters { left: 10, right: 5 },
        )
        .await
        .unwrap_err();

    assert!(error.is::<rig_agent::tool::MissingToolContext>());
}

#[rig_tool(description = "Prefix text using host context")]
async fn async_context_first(
    context: &mut rig_agent::tool::ToolContext,
    value: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    let prefix = context
        .get::<Prefix>()
        .map(|prefix| prefix.0.clone())
        .unwrap_or_default();
    std::future::ready(()).await;
    context.insert_result(Invocation("async"));
    Ok(format!("{prefix}{value}"))
}

mod domain {
    #[derive(serde::Deserialize, rig_core::schemars::JsonSchema)]
    pub struct ToolContext {
        pub label: String,
    }
}

#[rig_tool(description = "An application argument named ToolContext is not runtime context")]
fn domain_context_is_an_ordinary_argument(
    context: domain::ToolContext,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(context.label)
}

type RuntimeContextAlias = rig_agent::tool::ToolContext;

#[rig_tool(description = "An explicitly marked alias receives runtime context")]
fn marked_context_alias(
    #[rig(context)] context: &mut RuntimeContextAlias,
    value: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    context.insert_result(Invocation("alias"));
    Ok(value)
}

#[tokio::test]
async fn sync_context_is_excluded_from_schema_and_passed_in_argument_order() {
    let definition = rig_agent::tool::contextual_tool_definition(&SyncContextInTheMiddle);
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert_eq!(properties.len(), 2);
    assert!(properties.contains_key("left"));
    assert!(properties.contains_key("right"));
    assert!(!properties.contains_key("context"));

    assert_eq!(
        definition.parameters["required"],
        serde_json::json!(["left", "right"])
    );

    let mut context = ToolContext::new();
    context.insert(Offset(4));
    let output = SyncContextInTheMiddle
        .call(
            &mut context,
            SyncContextInTheMiddleParameters { left: 10, right: 5 },
        )
        .await
        .unwrap();

    assert_eq!(output, 19);
    assert_eq!(context.result::<Invocation>(), Some(&Invocation("sync")));
}

#[tokio::test]
async fn async_context_is_excluded_from_schema_and_passed_to_the_function() {
    let definition = rig_agent::tool::contextual_tool_definition(&AsyncContextFirst);
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert_eq!(properties.len(), 1);
    assert!(properties.contains_key("value"));
    assert!(!properties.contains_key("context"));
    assert_eq!(
        definition.parameters["required"],
        serde_json::json!(["value"])
    );

    let mut context = ToolContext::new();
    context.insert(Prefix("hello ".to_string()));
    let output = AsyncContextFirst
        .call(
            &mut context,
            AsyncContextFirstParameters {
                value: "world".to_string(),
            },
        )
        .await
        .unwrap();

    assert_eq!(output, "hello world");
    assert_eq!(context.result::<Invocation>(), Some(&Invocation("async")));
}

#[tokio::test]
async fn same_named_domain_type_remains_a_model_argument() {
    let definition = rig_core::tool::tool_definition(&DomainContextIsAnOrdinaryArgument);
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("context"));

    let output = DomainContextIsAnOrdinaryArgument
        .call(DomainContextIsAnOrdinaryArgumentParameters {
            context: domain::ToolContext {
                label: "domain".to_string(),
            },
        })
        .await
        .unwrap();
    assert_eq!(output, "domain");
}

#[tokio::test]
async fn explicit_marker_supports_imported_aliases() {
    let definition = rig_agent::tool::contextual_tool_definition(&MarkedContextAlias);
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert_eq!(properties.len(), 1);
    assert!(properties.contains_key("value"));

    let mut context = ToolContext::new();
    let output = MarkedContextAlias
        .call(
            &mut context,
            MarkedContextAliasParameters {
                value: "ok".to_string(),
            },
        )
        .await
        .unwrap();
    assert_eq!(output, "ok");
    assert_eq!(context.result::<Invocation>(), Some(&Invocation("alias")));
}

#[test]
fn invalid_context_parameters_are_rejected() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/tool_context/fail_immutable_context.rs");
    tests.compile_fail("tests/ui/tool_context/fail_owned_context.rs");
    tests.compile_fail("tests/ui/tool_context/fail_multiple_contexts.rs");
    tests.compile_fail("tests/ui/tool_context/fail_context_in_params.rs");
    tests.compile_fail("tests/ui/tool_context/fail_context_in_required.rs");
}
