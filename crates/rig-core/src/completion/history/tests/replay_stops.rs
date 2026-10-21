use super::*;

/// Synthetic stop states let the replay boundary exercise interruptions without
/// interrupting a credentialed provider call. Rig excludes the entire failed turn.
#[test]
fn failed_turns_and_their_results_do_not_replay_completed_neighbors_do() {
    for stop in [
        StopReason::Aborted("cancelled".into()),
        StopReason::Error("connection lost".into()),
    ] {
        let completed = turn(Some(same()), vec![signed_reasoning(), call("completed")]);
        let failed = Message::Assistant(AssistantMessage {
            content: vec![signed_reasoning(), call("failed")],
            origin: Some(same()),
            stop: Some(stop),
        });
        let history = vec![
            Message::user("start"),
            completed.clone(),
            Message::User {
                content: vec![result("completed", "done")],
            },
            failed,
            Message::User {
                content: vec![result("failed", "partial"), UserContent::text("continue")],
            },
        ];
        let adapted = adapt(&history, &TARGET);
        assert!(adapted.contains(&completed));
        assert!(!adapted.iter().any(|message| match message {
            Message::Assistant(message) => message.content.iter().any(|content| matches!(content, AssistantContent::ToolCall(call) if call.id == CallId::from_wire("failed"))),
            Message::User { content } => content.iter().any(|content| matches!(content, UserContent::ToolResult(result) if result.call == CallId::from_wire("failed"))),
            _ => false,
        }));
        assert!(adapted.iter().any(|message| matches!(message, Message::User { content } if content.contains(&result("completed", "done")))));
        assert!(adapted.iter().any(|message| matches!(message, Message::User { content } if content.contains(&UserContent::text("continue")))));
    }
}

/// The same signed tool turn may end normally or request tools; both are
/// replayable without another live provider response.
#[test]
fn completed_tool_turns_preserve_signed_reasoning_for_both_stop_reasons() {
    for stop in [StopReason::Stop, StopReason::ToolUse] {
        let assistant = Message::Assistant(AssistantMessage {
            content: vec![signed_reasoning(), call("lookup")],
            origin: Some(same()),
            stop: Some(stop),
        });
        let history = vec![
            Message::user("start"),
            assistant.clone(),
            Message::User {
                content: vec![result("lookup", "found")],
            },
        ];
        assert_eq!(adapt(&history, &TARGET), history);
    }
}

/// Origin changes are deterministic metadata inputs, not a cassette-dependent
/// behavior. Native signatures must not cross either API or provider boundary.
#[test]
fn changed_api_or_provider_drops_native_signatures_but_keeps_visible_content() {
    for origin in [
        Origin::new("other.api", "test", "model-a"),
        Origin::new("test.api", "other", "model-a"),
    ] {
        let history = vec![turn(
            Some(origin),
            vec![signed_reasoning(), AssistantContent::text("answer")],
        )];
        let adapted = adapt(&history, &TARGET);
        let Some(Message::Assistant(message)) = adapted.first() else {
            panic!("assistant must survive")
        };
        assert_eq!(
            message.content,
            vec![
                AssistantContent::text("thinking"),
                AssistantContent::text("answer")
            ]
        );
    }
}
