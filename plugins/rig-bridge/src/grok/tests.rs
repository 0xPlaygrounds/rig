use serde_json::json;

use super::{Grok, models};
use crate::acp::Agent;

#[test]
fn the_listed_models_are_read_with_the_default_first() {
    let listing = "You are not authenticated.\n\nDefault model: grok-4.6\n\nAvailable models:\n  - grok-4.5\n  * grok-4.6 (default)\n";

    assert_eq!(
        models(listing),
        vec!["grok-4.6".to_owned(), "grok-4.5".to_owned()]
    );
    assert_eq!(models("error: something else"), Vec::<String>::new());
}

#[test]
fn only_a_call_that_reaches_rig_s_server_is_a_rig_tool() {
    let meta = |name: &str| json!({ "x.ai/tool": { "name": name } });
    let search = json!({ "title": "search_tool", "rawInput": { "query": "read" }, "_meta": meta("search_tool") });
    let ours = json!({ "rawInput": { "tool_name": "rig__read", "tool_input": {} }, "_meta": meta("use_tool") });
    let other_server =
        json!({ "rawInput": { "tool_name": "rigged__read" }, "_meta": meta("use_tool") });
    // A built-in tool whose arguments name one of Rig's tools is still built in.
    let dressed =
        json!({ "rawInput": { "tool_name": "rig__read" }, "_meta": meta("run_terminal_cmd") });
    let unnamed = json!({ "rawInput": { "tool_name": "rig__read" } });

    let judged = [search, ours, other_server, dressed, unnamed].map(|call| Grok.is_rig_tool(&call));

    assert_eq!(judged, [true, true, false, false, false]);
}

#[test]
fn the_sign_in_is_one_the_program_offers() {
    let signed_in = json!({ "authMethods": [{ "id": "grok.com" }, { "id": "cached_token" }] });
    let signed_out = json!({ "authMethods": [{ "id": "grok.com" }] });

    assert_eq!(Grok.sign_in(&signed_in), Some("cached_token".to_owned()));
    assert_eq!(Grok.sign_in(&signed_out), None);
}

#[test]
fn an_inference_is_counted_from_grok_s_own_report_with_the_cache_in_its_input() {
    let report = json!({ "sessionId": "s", "update": { "sessionUpdate": "response_completed", "usage": {
        "input_tokens": 771, "output_tokens": 711, "cache_read_input_tokens": 4736,
        "cache_creation_input_tokens": 0, "reasoning_tokens": 682,
    } } });
    let other = json!({ "sessionId": "s", "update": { "sessionUpdate": "turn_completed", "usage": { "inputTokens": 1 } } });

    let counted = Grok
        .counted("_x.ai/session_notification", &report)
        .map(|usage| {
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.reasoning_tokens,
            )
        });

    assert_eq!(
        counted,
        Some((Some(5507), Some(711), Some(4736), Some(682)))
    );
    assert_eq!(Grok.counted("_x.ai/session_notification", &other), None);
    assert_eq!(Grok.counted("session/update", &report), None);
}

#[test]
fn a_prompt_ends_when_grok_says_so_for_that_prompt() {
    let complete = json!({ "sessionId": "s", "promptId": "p1", "stopReason": "end_turn" });
    let turn = json!({ "sessionId": "s", "update": {
        "sessionUpdate": "turn_completed", "prompt_id": "p1", "stop_reason": "max_tokens",
    } });
    let wake = json!({ "sessionId": "s", "update": {
        "sessionUpdate": "turn_completed", "prompt_id": "task-completed-7", "stop_reason": "end_turn",
    } });

    assert_eq!(
        Grok.ended("_x.ai/session/prompt_complete", &complete, "p1"),
        Some(json!({ "stopReason": "end_turn" })),
    );
    assert_eq!(
        Grok.ended("_x.ai/session_notification", &turn, "p1"),
        Some(json!({ "stopReason": "max_tokens" })),
    );
    assert_eq!(
        Grok.ended("_x.ai/session/prompt_complete", &complete, "p2"),
        None
    );
    assert_eq!(Grok.ended("_x.ai/session_notification", &wake, "p1"), None);
    assert_eq!(Grok.ended("session/update", &turn, "p1"), None);
}
