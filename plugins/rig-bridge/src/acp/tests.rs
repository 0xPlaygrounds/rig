use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::rig_core::completion::{Effort, FinishReason};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::sync::mpsc;

use super::{Agent, Opening, Peer, converse};
use crate::harness::{ContextSnapshot, Event, Part, Prompt};

/// An agent with no system prompt channel, one sign-in and a model request.
struct Plain;

impl Agent for Plain {
    fn name(&self) -> &'static str {
        "plain"
    }

    fn prepare(&self, _: &mut Command, _: &Opening) -> Result<(), String> {
        Ok(())
    }

    fn session_meta(&self, _: &Value, _: &str) -> Option<Value> {
        None
    }

    fn choose(&self, session: &str, _: &Value, opening: &Opening) -> Vec<(&'static str, Value)> {
        vec![(
            "session/set_model",
            json!({ "sessionId": session, "modelId": opening.model }),
        )]
    }

    fn sign_in(&self, _: &Value) -> Option<String> {
        Some("cached".to_owned())
    }

    fn login_hint(&self) -> &'static str {
        "plain login"
    }

    fn note(&self, _: Option<&std::path::Path>) -> String {
        String::new()
    }

    fn is_rig_tool(&self, tool_call: &Value, _: Option<&str>) -> bool {
        tool_call.get("title").and_then(Value::as_str) == Some("rig")
    }
}

/// What the scripted agent was sent, in order.
type Received = Arc<Mutex<Vec<Value>>>;

/// A scripted ACP agent on the other end of `stream`: it refuses the first
/// `session/new` until signed in, and answers a prompt with a thought, a
/// sentence and two permission requests, one for a tool of Rig's.
async fn agent(stream: DuplexStream, received: Received) {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();
    let mut signed_in = false;
    let mut asked = 100;
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        received
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(message.clone());
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let mut out = Vec::new();
        match message.get("method").and_then(Value::as_str) {
            Some("initialize") => out.push(json!({ "id": id, "result": { "protocolVersion": 1 } })),
            Some("session/new") if !signed_in => out.push(json!({ "id": id,
                "error": { "code": -32000, "message": "Authentication required" } })),
            Some("authenticate") => {
                signed_in = true;
                out.push(json!({ "id": id, "result": {} }));
            }
            Some("session/new") => out.push(json!({ "id": id, "result": { "sessionId": "s1" } })),
            Some("session/set_model") => out.push(json!({ "id": id, "result": {} })),
            Some("session/prompt") => {
                let update = |kind: &str, text: &str| {
                    json!({ "method": "session/update", "params": {
                    "sessionId": "s1",
                    "update": { "sessionUpdate": kind, "content": { "type": "text", "text": text } },
                } })
                };
                let options = json!([
                    { "optionId": "no", "kind": "reject_once" },
                    { "optionId": "yes", "kind": "allow_once" },
                ]);
                out.push(update("agent_thought_chunk", "thinking"));
                out.push(update("agent_message_chunk", "Hello"));
                out.push(update("tool_call", "ignored"));
                for title in ["rig", "terminal"] {
                    asked += 1;
                    out.push(
                        json!({ "id": asked, "method": "session/request_permission", "params": {
                        "sessionId": "s1", "toolCall": { "title": title }, "options": options,
                    } }),
                    );
                }
                out.push(json!({ "id": 999, "method": "fs/read_text_file", "params": {} }));
                out.push(json!({ "id": id, "result": {
                    "stopReason": "end_turn",
                    "usage": { "inputTokens": 10, "outputTokens": 5, "cachedReadTokens": 4 },
                } }));
            }
            _ => {}
        }
        for mut message in out {
            if let Some(message) = message.as_object_mut() {
                message.insert("jsonrpc".to_owned(), "2.0".into());
            }
            let line = format!("not json\n{message}\n");
            if writer.write_all(line.as_bytes()).await.is_err() {
                return;
            }
        }
    }
}

/// Runs two prompts against the scripted agent: the events reported and
/// what the agent received.
async fn conversation() -> (Vec<Event>, Vec<Value>) {
    let (ours, theirs) = tokio::io::duplex(1 << 16);
    let received = Received::default();
    let scripted = tokio::spawn(agent(theirs, received.clone()));
    let (reader, writer) = tokio::io::split(ours);
    let (events, mut reported) = mpsc::unbounded_channel();
    let mut peer = Peer::new(reader, writer, events, Arc::new(Plain));
    let (prompts, prompted) = mpsc::unbounded_channel();
    let opening = Opening {
        dir: "/work".into(),
        system: "Be brief.".to_owned(),
        model: "model-1".to_owned(),
        effort: None,
        mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
        mcp_token: "secret".to_owned(),
        tools: vec!["read".to_owned()],
    };
    let _ = prompts.send(Prompt {
        parts: vec![Part::Text("one".to_owned())],
        effort: None,
    });
    let _ = prompts.send(Prompt {
        parts: vec![Part::Text("two".to_owned())],
        effort: None,
    });
    drop(prompts);
    let ended = converse(&mut peer, opening, prompted).await;
    drop(peer);
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

fn sent<'a>(received: &'a [Value], method: &str) -> Vec<&'a Value> {
    let named = |message: &&Value| message.get("method").and_then(Value::as_str) == Some(method);
    received.iter().filter(named).collect()
}

#[tokio::test]
async fn a_prompt_is_answered_with_what_the_agent_says_and_counts() {
    let (events, _) = conversation().await;

    let turn = |events: &[Event]| events.iter().take(4).cloned().collect::<Vec<_>>();
    let counted = events.iter().find_map(|event| match event {
        Event::Usage(usage) => Some((
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens,
        )),
        _ => None,
    });
    assert_eq!(events.len(), 8);
    assert_eq!(
        turn(&events).get(..2),
        Some(
            [
                Event::Reasoning("thinking".to_owned()),
                Event::Text("Hello".to_owned())
            ]
            .as_slice()
        ),
    );
    assert_eq!(counted, Some((Some(10), Some(5), Some(4))));
    assert_eq!(events.get(3), Some(&Event::Done(Ok(FinishReason::Stop))));
    assert_eq!(events.last(), Some(&Event::Done(Ok(FinishReason::Stop))));
}

#[tokio::test]
async fn the_session_opens_signed_in_with_rig_s_server_and_the_model() {
    let (_, received) = conversation().await;

    let methods: Vec<&str> = received
        .iter()
        .filter_map(|message| message.get("method")?.as_str())
        .take(6)
        .collect();
    let new = sent(&received, "session/new")
        .first()
        .copied()
        .cloned()
        .unwrap_or_default();
    let initialize = sent(&received, "initialize")
        .first()
        .copied()
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        methods,
        [
            "initialize",
            "session/new",
            "authenticate",
            "session/new",
            "session/set_model",
            "session/prompt"
        ],
    );
    assert_eq!(
        new.get("params"),
        Some(&json!({ "cwd": "/work", "mcpServers": [{
            "type": "http", "name": "rig", "url": "http://127.0.0.1:1/mcp",
            "headers": [{ "name": "Authorization", "value": "Bearer secret" }],
        }] })),
    );
    assert_eq!(
        initialize.pointer("/params/clientCapabilities"),
        Some(
            &json!({ "fs": { "readTextFile": false, "writeTextFile": false }, "terminal": false })
        ),
    );
    assert_eq!(
        sent(&received, "session/set_model")
            .first()
            .and_then(|chosen| chosen.pointer("/params/modelId")),
        Some(&json!("model-1")),
    );
}

#[tokio::test]
async fn the_system_prompt_opens_the_first_prompt_only() {
    let (_, received) = conversation().await;

    let prompts: Vec<Value> = sent(&received, "session/prompt")
        .into_iter()
        .filter_map(|prompt| prompt.pointer("/params/prompt").cloned())
        .collect();
    assert_eq!(
        prompts,
        vec![
            json!([
                { "type": "text", "text": "<instructions>\nBe brief.\n</instructions>\n" },
                { "type": "text", "text": "one" },
            ]),
            json!([{ "type": "text", "text": "two" }]),
        ],
    );
}

#[tokio::test]
async fn only_rig_s_tools_are_permitted_and_other_requests_refused() {
    let (_, received) = conversation().await;

    let answer = |id: u64| {
        let answered = |message: &&Value| {
            message.get("method").is_none() && message.get("id").and_then(Value::as_u64) == Some(id)
        };
        received.iter().find(answered).cloned().unwrap_or_default()
    };
    assert_eq!(
        answer(101).pointer("/result/outcome"),
        Some(&json!({ "outcome": "selected", "optionId": "yes" })),
    );
    assert_eq!(
        answer(102).pointer("/result/outcome"),
        Some(&json!({ "outcome": "selected", "optionId": "no" })),
    );
    assert_eq!(answer(999).pointer("/error/code"), Some(&json!(-32601)));
}

#[tokio::test]
async fn usage_update_session_window_and_max_tokens_are_emitted_cleanly() {
    let (client, server_stream) = tokio::io::duplex(16 * 1024);
    let task = tokio::spawn(async move {
        let (reader, mut writer) = tokio::io::split(server_stream);
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            let method = message.get("method").and_then(Value::as_str);
            let out: Vec<Value> = match method {
                Some("initialize") => vec![json!({ "jsonrpc": "2.0", "id": id, "result": {} })],
                Some("session/new") => vec![json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "sessionId": "s1",
                        "models": {
                            "availableModels": [{
                                "modelId": "model-1",
                                "_meta": { "totalContextTokens": 256000 }
                            }]
                        }
                    }
                })],
                Some("session/set_model") => {
                    vec![json!({ "jsonrpc": "2.0", "id": id, "result": {} })]
                }
                Some("session/prompt") => vec![
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 0, "size": 1048576 } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 100000, "size": 1048576 } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 105000, "size": 1048576 } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "stopReason": "max_tokens",
                            "usage": {
                                "inputTokens": 3500000,
                                "outputTokens": 40000,
                                "cachedReadTokens": 3300000,
                                "totalTokens": 3540000
                            }
                        }
                    }),
                ],
                _ => vec![],
            };
            for msg in out {
                let mut line = msg.to_string();
                line.push('\n');
                let _ = writer.write_all(line.as_bytes()).await;
            }
        }
    });

    let (reader, writer) = tokio::io::split(client);
    let (events_tx, mut events_rx) = mpsc::unbounded_channel();
    let (prompts_tx, prompts_rx) = mpsc::unbounded_channel();
    let mut peer = Peer::new(reader, writer, events_tx, Arc::new(Plain));
    let _ = prompts_tx.send(Prompt {
        parts: vec![Part::Text("go".to_owned())],
        effort: None,
    });
    drop(prompts_tx);
    let ended = converse(
        &mut peer,
        Opening {
            dir: PathBuf::from("/work"),
            system: String::new(),
            model: "model-1".to_owned(),
            effort: None,
            mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
            mcp_token: "secret".to_owned(),
            tools: Vec::new(),
        },
        prompts_rx,
    )
    .await;
    drop(peer);
    let _ = task.await;
    assert_eq!(ended, Ok(()));

    let mut events = Vec::new();
    while let Ok(event) = events_rx.try_recv() {
        events.push(event);
    }
    let contexts: Vec<ContextSnapshot> = events
        .iter()
        .filter_map(|event| match event {
            Event::Context(snapshot) => Some(*snapshot),
            _ => None,
        })
        .collect();
    assert_eq!(
        contexts.first().copied(),
        Some(ContextSnapshot {
            used: None,
            window: Some(256_000),
        })
    );
    assert_eq!(
        contexts.last().copied(),
        Some(ContextSnapshot {
            used: Some(105_000),
            window: Some(1_048_576),
        })
    );
    let usage = events.iter().find_map(|event| match event {
        Event::Usage(usage) => Some(*usage),
        _ => None,
    });
    assert_eq!(usage.and_then(|u| u.input_tokens), Some(3_500_000));
    assert_eq!(usage.and_then(|u| u.output_tokens), Some(40_000));
    assert_eq!(events.last(), Some(&Event::Done(Ok(FinishReason::Length))));
}

#[tokio::test]
async fn acp_updates_reasoning_effort_mid_session_and_restores_default() {
    struct EffortAgent;
    impl Agent for EffortAgent {
        fn name(&self) -> &'static str {
            "effort"
        }
        fn prepare(&self, _: &mut Command, _: &Opening) -> Result<(), String> {
            Ok(())
        }
        fn session_meta(&self, _: &Value, _: &str) -> Option<Value> {
            Some(json!({}))
        }
        fn choose(&self, _: &str, _: &Value, _: &Opening) -> Vec<(&'static str, Value)> {
            Vec::new()
        }
        fn sign_in(&self, _: &Value) -> Option<String> {
            None
        }
        fn note(&self, _: Option<&std::path::Path>) -> String {
            String::new()
        }
        fn login_hint(&self) -> &'static str {
            "effort login"
        }
        fn is_rig_tool(&self, _: &Value, _: Option<&str>) -> bool {
            true
        }
        fn changes_effort(&self) -> bool {
            true
        }
        fn set_effort(&self, session: &str, effort: Effort) -> Option<(&'static str, Value)> {
            Some((
                "session/set_config_option",
                json!({
                    "sessionId": session,
                    "configId": "reasoning_effort",
                    "value": effort.as_str(),
                }),
            ))
        }
    }

    let (client, server) = tokio::io::duplex(1 << 16);
    let received = Received::default();
    let seen = received.clone();
    let task = tokio::spawn(async move {
        let (reader, mut writer) = tokio::io::split(server);
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            seen.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(msg.clone());
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            let out = match msg.get("method").and_then(Value::as_str) {
                Some("initialize") => json!({ "id": id, "result": {} }),
                Some("session/new") => json!({
                    "id": id,
                    "result": {
                        "sessionId": "s1",
                        "configOptions": [{
                            "id": "reasoning_effort",
                            "currentValue": "medium",
                        }]
                    }
                }),
                Some("session/set_config_option") => json!({ "id": id, "result": {} }),
                Some("session/prompt") => {
                    json!({ "id": id, "result": { "stopReason": "end_turn" } })
                }
                _ => continue,
            };
            let mut text = out.to_string();
            text.push('\n');
            let _ = writer.write_all(text.as_bytes()).await;
        }
    });

    let (reader, writer) = tokio::io::split(client);
    let (events_tx, _events_rx) = mpsc::unbounded_channel();
    let (prompts_tx, prompts_rx) = mpsc::unbounded_channel();
    let mut peer = Peer::new(reader, writer, events_tx, Arc::new(EffortAgent));
    let _ = prompts_tx.send(Prompt {
        parts: vec![Part::Text("first".to_owned())],
        effort: None,
    });
    let _ = prompts_tx.send(Prompt {
        parts: vec![Part::Text("second".to_owned())],
        effort: Some(Effort::Low),
    });
    let _ = prompts_tx.send(Prompt {
        parts: vec![Part::Text("third".to_owned())],
        effort: Some(Effort::Low),
    });
    let _ = prompts_tx.send(Prompt {
        parts: vec![Part::Text("fourth".to_owned())],
        effort: None,
    });
    drop(prompts_tx);

    let ended = converse(
        &mut peer,
        Opening {
            dir: PathBuf::from("/work"),
            system: String::new(),
            model: "model-1".to_owned(),
            effort: None,
            mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
            mcp_token: "secret".to_owned(),
            tools: Vec::new(),
        },
        prompts_rx,
    )
    .await;
    drop(peer);
    let _ = task.await;
    assert_eq!(ended, Ok(()));

    let received = received
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let updates: Vec<&str> = sent(&received, "session/set_config_option")
        .into_iter()
        .filter_map(|msg| msg.pointer("/params/value").and_then(Value::as_str))
        .collect();
    // Turn 1: default (no-op), Turn 2: low, Turn 3: low (deduped), Turn 4: default -> restored to medium.
    assert_eq!(updates, vec!["low", "medium"]);
}
