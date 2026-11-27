use serde_json::json;

use super::*;

#[test]
fn a_prompted_event_gives_its_session_and_prompt_whichever_field_holds_the_body() {
    let event = |activity: Value| {
        json!({
            "type": "AgentSessionEvent",
            "action": "prompted",
            "agentSession": { "id": "session-1" },
            "agentActivity": activity,
        })
    };
    let schema = event(json!({
        "id": "p1",
        "createdAt": "2026-01-01T00:00:00.000Z",
        "content": { "type": "prompt", "body": "Use sqlite." },
        "user": { "name": "Ana" },
    }));
    assert_eq!(
        prompted(&schema),
        Some(Prompted {
            session: "session-1".to_owned(),
            seen: Seen {
                id: "p1".to_owned(),
                created_at: "2026-01-01T00:00:00.000Z".to_owned(),
                prompt: Some("Use sqlite.".to_owned()),
                stop: false,
                by: Some("Ana".to_owned()),
                from_app: false,
            },
        })
    );
    let guide = event(json!({ "id": "p2", "body": "Stop.", "signal": "stop" }));
    let seen = prompted(&guide).map(|prompted| prompted.seen);
    assert_eq!(
        seen.as_ref().and_then(|seen| seen.prompt.as_deref()),
        Some("Stop.")
    );
    assert!(seen.is_some_and(|seen| seen.stop));
    assert_eq!(prompted(&event(json!({ "body": "No id." }))), None);
}
