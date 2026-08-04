//! `ToolContext` was removed with the data-oriented migration: `#[rig_tool]`
//! functions are portable and context-free. A parameter that names the old
//! runtime context (or is explicitly marked `#[rig(context)]`) must fail with
//! the targeted removal diagnostic, while application arguments that merely
//! *look* like a context (a domain type named `ToolContext`) stay ordinary
//! model-facing arguments.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_agent::tool::PortableTool;
use rig_derive::rig_tool;

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

#[tokio::test]
async fn same_named_domain_type_remains_a_model_argument() {
    let definition = rig_agent::tool::portable_tool_definition(&DomainContextIsAnOrdinaryArgument);
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

/// Contextual `#[rig_tool]` parameters are a deliberate compile error now:
/// the macro points authors at closing over state (or
/// `PortableDynamicTool::new`) instead of a `&mut ToolContext` parameter.
#[test]
fn removed_context_parameters_are_rejected_with_the_targeted_error() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/tool_context/fail_removed_context_marker.rs");
    tests.compile_fail("tests/ui/tool_context/fail_removed_context_path.rs");
}
