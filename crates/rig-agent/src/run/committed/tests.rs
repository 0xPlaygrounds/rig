use super::*;
use crate::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, TurnPolicy};
use rig_core::completion::Usage;
use rig_core::message::{
    AssistantMessage, CallId, Reasoning, ToolFunction, ToolName, ToolResultContent,
};
use serde_json::json;

fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall::from_wire(
        id,
        ToolFunction::new(ToolName::new(name).expect("tool name"), args),
    )
}

fn result(id: &str) -> UserContent {
    UserContent::tool_result(
        CallId::from_wire(id),
        ToolName::new("add").expect("tool name"),
        vec![ToolResultContent::text("3")],
    )
}

fn turn(choice: Vec<AssistantContent>, output_tool: Option<&str>) -> ModelTurn {
    let advertised = ["add".to_string()].into_iter().collect();
    let policy = TurnPolicy::new(advertised, None, output_tool.map(str::to_owned)).expect("policy");
    ModelTurn::new(
        AssistantMessage::default(),
        choice,
        Usage::default(),
        policy,
        json!({"origin": "hand-built test turn"}),
    )
}

fn accept(run: &mut AgentRun, choice: Vec<AssistantContent>, output_tool: Option<&str>) {
    let outcome = run
        .model_response(turn(choice, output_tool))
        .expect("the turn is accepted");
    assert!(
        matches!(outcome, ModelTurnOutcome::Continue { .. }),
        "{outcome:?}"
    );
}

/// Ids in projection order, `call:` or `result:` prefixed.
fn projected(messages: &[Message]) -> Vec<String> {
    project(messages)
        .map(|item| match item {
            CommittedItem::ToolCall(call) => format!("call:{}", call.id),
            CommittedItem::ToolResult(result) => format!("result:{}", result.call),
        })
        .collect()
}

#[test]
fn project_yields_calls_and_results_in_message_order_and_skips_the_rest() {
    let log = vec![
        Message::user("prompt"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::Reasoning(Reasoning::new("thinking")),
            AssistantContent::text("checking"),
            AssistantContent::ToolCall(call("a", "add", json!({}))),
            AssistantContent::ToolCall(call("b", "add", json!({}))),
        ])),
        Message::User {
            content: vec![result("b"), result("a")],
        },
        Message::System {
            content: "note".to_string(),
        },
        Message::assistant("done"),
    ];

    assert_eq!(
        projected(&log),
        ["call:a", "call:b", "result:b", "result:a"]
    );
}

#[test]
fn a_resume_cursor_projects_only_what_is_committed_after_it() {
    let mut run = AgentRun::new("add twice").max_turns(3);
    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallModel { .. })
    ));
    accept(
        &mut run,
        vec![AssistantContent::ToolCall(call("a", "add", json!({})))],
        None,
    );
    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallTools { .. })
    ));
    run.tool_results(vec![result("a")]).expect("results commit");

    // The process restarts: the restored run starts its cursor at its end.
    let saved = serde_json::to_string(&run).expect("run serializes");
    let mut run: AgentRun = serde_json::from_str(&saved).expect("run restores");
    let cursor = run.messages().len();
    assert!(projected(&run.messages()[cursor..]).is_empty());

    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallModel { .. })
    ));
    accept(
        &mut run,
        vec![AssistantContent::ToolCall(call("b", "add", json!({})))],
        None,
    );
    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallTools { .. })
    ));
    run.tool_results(vec![result("b")]).expect("results commit");

    assert_eq!(projected(&run.messages()[cursor..]), ["call:b", "result:b"]);
}

#[test]
fn an_output_reprompt_projects_its_call_with_the_feedback_result() {
    let schema = json!({"type": "object", "required": ["answer"]});
    let mut run = AgentRun::new("answer")
        .max_turns(2)
        .with_output_tool_name("final_result")
        .with_output_validation(Some(schema), 1);
    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallModel { .. })
    ));
    let missing_answer = call("out", "final_result", json!({"other": 1}));
    accept(
        &mut run,
        vec![AssistantContent::ToolCall(missing_answer)],
        Some("final_result"),
    );

    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallModel { turn: 2, .. })
    ));
    assert_eq!(projected(run.messages()), ["call:out", "result:out"]);
}

#[test]
fn a_final_output_turn_projects_no_call() {
    let mut run = AgentRun::new("answer").with_output_tool_name("final_result");
    assert!(matches!(
        run.next_step(),
        Ok(AgentRunStep::CallModel { .. })
    ));
    let answer = call("out", "final_result", json!({"answer": 4}));
    accept(
        &mut run,
        vec![AssistantContent::ToolCall(answer)],
        Some("final_result"),
    );

    assert!(matches!(run.next_step(), Ok(AgentRunStep::Done(_))));
    assert!(projected(run.messages()).is_empty(), "{:?}", run.messages());
}
