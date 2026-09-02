use super::*;
use crate::agent::AgentBuilder;
use crate::test_utils::{MockCompletionModel, MockContextProbeTool, MockTurn, SessionId};
use crate::tool::ToolContext;

/// A `ToolContext` set on the outer run propagates into a sub-agent
/// invoked as a tool, so the inner agent's own tools observe it.
#[tokio::test]
async fn context_propagates_into_sub_agent() {
    // Inner agent: calls a context-probing tool, then answers.
    let probe = MockContextProbeTool::default();
    let inner_model = MockCompletionModel::new([
        MockTurn::tool_call("c1", "context_probe", json!({})),
        MockTurn::text("inner done"),
    ]);
    let inner = AgentBuilder::new(inner_model)
        .name("researcher")
        .tool(probe.clone())
        .build();

    // Outer agent: delegates to the inner agent (registered as the
    // "researcher" tool), then answers.
    let outer_model = MockCompletionModel::new([
        MockTurn::tool_call("c2", "researcher", json!({"prompt": "do research"})),
        MockTurn::text("outer done"),
    ]);
    let outer = AgentBuilder::new(outer_model)
        .dynamic_tool(inner.into_tool())
        .build();

    let mut context = ToolContext::new();
    context.insert(SessionId("abc-123".to_string())).unwrap();

    let out = outer
        .prompt("start")
        .tool_context(context)
        .max_turns(5)
        .await
        .expect("run succeeds");

    assert_eq!(out.output, "outer done");
    assert_eq!(probe.observed().as_deref(), Some("session:abc-123"));
}
