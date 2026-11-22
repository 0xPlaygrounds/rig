use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::sync::mpsc;

use super::{Codex, Listed, Opening, Wire, converse};
use crate::harness::{Event, Part};

#[test]
fn the_catalog_lists_the_shown_models_and_gives_none_of_them_tools() {
    let listing = json!({ "models": [
        { "slug": "gpt-a", "display_name": "GPT A", "visibility": "list", "context_window": 500000,
          "tool_mode": "code_mode_only", "multi_agent_version": "v2", "supports_search_tool": true,
          "experimental_supported_tools": ["clock"], "apply_patch_tool_type": "freeform", "kept": 1 },
        { "slug": "gpt-hidden", "display_name": "Hidden", "visibility": "hide", "tool_mode": "code_mode_only" },
    ] });

    let read = Codex::new(&listing.to_string());
    let listed = read.as_ref().map(|(_, listed)| listed.clone());
    let catalog = read.and_then(|(codex, _)| serde_json::from_str::<Value>(&codex.catalog).ok());

    assert_eq!(
        listed,
        Some(vec![Listed {
            id: "gpt-a".to_owned(),
            name: "GPT A".to_owned(),
            context_window: Some(500_000)
        }]),
    );
    assert_eq!(
        catalog
            .as_ref()
            .and_then(|catalog| catalog.pointer("/models/0")),
        Some(&json!({
            "slug": "gpt-a", "display_name": "GPT A", "visibility": "list", "context_window": 500000,
            "tool_mode": "direct", "multi_agent_version": null, "supports_search_tool": false,
            "experimental_supported_tools": [], "apply_patch_tool_type": null, "kept": 1,
        })),
    );
    assert_eq!(
        catalog
            .as_ref()
            .and_then(|catalog| catalog.pointer("/models/1/tool_mode")),
        Some(&json!("direct")),
    );
    assert!(Codex::new("You are not signed in.").is_none());
}

/// What the scripted program was sent, in order.
type Received = Arc<Mutex<Vec<Value>>>;

/// A scripted app server on the other end of `stream`. It answers a turn
/// with a reasoning summary, an approval request for one of Rig's tools
/// and for something else, a counted model call and a sentence; a turn
/// whose text is `fail` fails.
async fn server(stream: DuplexStream, received: Received, signed_in: bool) {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        received
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(message.clone());
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let notify = |method: &str, params: Value| json!({ "method": method, "params": params });
        let mut out = Vec::new();
        match message.get("method").and_then(Value::as_str) {
            Some("initialize") => out.push(json!({ "id": id, "result": { "userAgent": "x" } })),
            Some("account/read") => {
                let account = if signed_in {
                    json!({ "type": "chatgpt" })
                } else {
                    Value::Null
                };
                out.push(json!({ "id": id, "result": { "account": account } }));
            }
            Some("thread/start") => {
                out.push(notify(
                    "mcpServer/startupStatus/updated",
                    json!({ "name": "rig", "status": "starting" }),
                ));
                out.push(json!({ "id": id, "result": { "thread": { "id": "t1" } } }));
                out.push(notify(
                    "mcpServer/startupStatus/updated",
                    json!({ "name": "other", "status": "failed" }),
                ));
                out.push(notify(
                    "mcpServer/startupStatus/updated",
                    json!({ "name": "rig", "status": "ready" }),
                ));
            }
            Some("turn/start") => {
                let failing = message.pointer("/params/input/0/text") == Some(&json!("fail"));
                out.push(json!({ "id": id, "result": { "turn": { "id": "u1", "status": "inProgress" } } }));
                if failing {
                    out.push(notify(
                        "turn/completed",
                        json!({ "turn": { "status": "failed",
                        "error": { "message": "The model is not supported." } } }),
                    ));
                } else {
                    out.push(notify(
                        "item/reasoning/summaryPartAdded",
                        json!({ "summaryIndex": 0 }),
                    ));
                    out.push(notify(
                        "item/reasoning/summaryTextDelta",
                        json!({ "delta": "Plan" }),
                    ));
                    out.push(notify(
                        "item/reasoning/summaryPartAdded",
                        json!({ "summaryIndex": 1 }),
                    ));
                    // The program's request ids are its own, from 0.
                    out.push(json!({ "id": 0, "method": "mcpServer/elicitation/request",
                        "params": { "serverName": "rig", "message": "Allow?" } }));
                    out.push(json!({ "id": 1, "method": "item/commandExecution/requestApproval", "params": {} }));
                    out.push(notify("thread/tokenUsage/updated", json!({ "tokenUsage": {
                        "total": { "inputTokens": 999 },
                        "last": { "inputTokens": 627, "cachedInputTokens": 100, "outputTokens": 208, "reasoningOutputTokens": 138 },
                    } })));
                    out.push(notify(
                        "item/agentMessage/delta",
                        json!({ "delta": "Hello" }),
                    ));
                    out.push(notify("thread/status/changed", json!({})));
                    out.push(notify(
                        "turn/completed",
                        json!({ "turn": { "status": "completed", "error": null } }),
                    ));
                }
            }
            _ => {}
        }
        for message in out {
            if writer
                .write_all(format!("{message}\n").as_bytes())
                .await
                .is_err()
            {
                return;
            }
        }
    }
}

/// Runs `prompts` against the scripted server: the events reported and
/// what the server received.
async fn conversation(prompts: &[&str], signed_in: bool) -> (Vec<Event>, Vec<Value>) {
    let (ours, theirs) = tokio::io::duplex(1 << 16);
    let received = Received::default();
    let scripted = tokio::spawn(server(theirs, received.clone(), signed_in));
    let (reader, writer) = tokio::io::split(ours);
    let (events, mut reported) = mpsc::unbounded_channel();
    let mut wire = Wire::new(reader, writer, events);
    let (sender, prompted) = mpsc::unbounded_channel();
    for prompt in prompts {
        let _ = sender.send(vec![Part::Text((*prompt).to_owned())]);
    }
    drop(sender);
    let opening = Opening {
        dir: "/session".into(),
        system: "Be brief.".to_owned(),
        model: "gpt-a".to_owned(),
        effort: Some(rig_harness::rig_core::completion::Effort::High),
    };
    let ended = converse(&mut wire, opening, prompted).await;
    drop(wire);
    let _ = scripted.await;
    let mut events = Vec::new();
    while let Ok(event) = reported.try_recv() {
        events.push(event);
    }
    if let Err(why) = ended {
        events.push(Event::Done(Err(why)));
    }
    let received = received
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    (events, received)
}

#[tokio::test]
async fn a_turn_is_answered_with_what_the_model_says_and_each_call_s_count() {
    let (events, _) = conversation(&["one", "fail"], true).await;

    let counted = events.iter().find_map(|event| match event {
        Event::Usage(usage) => Some((
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
        )),
        _ => None,
    });
    let said: Vec<&Event> = events
        .iter()
        .filter(|event| !matches!(event, Event::Usage(_)))
        .collect();
    assert_eq!(counted, Some((Some(627), Some(100), Some(208), Some(138))));
    assert_eq!(
        said,
        [
            &Event::Reasoning("Plan".to_owned()),
            &Event::Reasoning("\n\n".to_owned()),
            &Event::Text("Hello".to_owned()),
            &Event::Done(Ok(())),
            &Event::Done(Err("The model is not supported.".to_owned())),
        ],
    );
}

#[tokio::test]
async fn the_thread_opens_with_rig_s_prompt_and_nothing_to_approve() {
    let (_, received) = conversation(&["one"], true).await;

    let sent = |method: &str| {
        let named =
            |message: &&Value| message.get("method").and_then(Value::as_str) == Some(method);
        received
            .iter()
            .find(named)
            .and_then(|message| message.get("params"))
            .cloned()
    };
    let methods: Vec<&str> = received
        .iter()
        .filter_map(|message| message.get("method")?.as_str())
        .collect();
    assert_eq!(
        methods,
        [
            "initialize",
            "initialized",
            "account/read",
            "thread/start",
            "turn/start"
        ]
    );
    assert_eq!(
        sent("thread/start"),
        Some(json!({
            "model": "gpt-a", "cwd": "/session", "approvalPolicy": "never", "sandbox": "read-only",
            "ephemeral": true, "baseInstructions": "Be brief.",
        })),
    );
    assert_eq!(
        sent("turn/start"),
        Some(json!({
            "threadId": "t1", "input": [{ "type": "text", "text": "one" }], "summary": "auto", "effort": "high",
        })),
    );
}

#[tokio::test]
async fn only_a_call_of_rig_s_tools_is_approved() {
    let (_, received) = conversation(&["one"], true).await;

    let answer = |id: u64| {
        let answered = |message: &&Value| {
            message.get("method").is_none() && message.get("id").and_then(Value::as_u64) == Some(id)
        };
        received.iter().find(answered).cloned().unwrap_or_default()
    };
    assert_eq!(
        answer(0).get("result"),
        Some(&json!({ "action": "accept", "content": null }))
    );
    assert_eq!(answer(1).pointer("/error/code"), Some(&json!(-32601)));
}

#[tokio::test]
async fn a_program_that_is_not_signed_in_says_how_to_sign_in() {
    let (events, received) = conversation(&["one"], false).await;

    assert_eq!(
        events,
        [Event::Done(Err(
            "it is not signed in: run `codex login`".to_owned()
        ))]
    );
    assert_eq!(received.len(), 3);
}
