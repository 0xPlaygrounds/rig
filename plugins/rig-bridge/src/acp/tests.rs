use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::rig_core::completion::{Effort, FinishReason};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use super::{Agent, Opening, Peer, converse};
use crate::harness::{ContextSnapshot, Event, Part, Prompt};

/// An agent with no system prompt channel, one sign-in and a model request.
struct Plain {
    dynamic_effort: bool,
    cumulative_usage: bool,
}

impl Plain {
    const fn basic() -> Self {
        Self {
            dynamic_effort: false,
            cumulative_usage: false,
        }
    }
}

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

    fn effort_value(&self, effort: Effort) -> Option<&'static str> {
        self.dynamic_effort.then(|| effort.as_str())
    }

    fn cumulative_usage(&self) -> bool {
        self.cumulative_usage
    }
}

/// What the scripted agent was sent, in order.
type Received = Arc<Mutex<Vec<Value>>>;

/// Runs `prompts` against a scripted ACP duplex responder, returning `(events, received)`.
async fn run_acp(
    agent: Arc<dyn Agent>,
    system: &str,
    prompts: Vec<Prompt>,
    mut responder: impl FnMut(&Value) -> Vec<Value> + Send + 'static,
) -> (Vec<Event>, Vec<Value>) {
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
            for mut out in responder(&msg) {
                if let Some(obj) = out.as_object_mut() {
                    obj.entry("jsonrpc".to_owned()).or_insert("2.0".into());
                }
                let line = format!("not json\n{out}\n");
                if writer.write_all(line.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
    });

    let (reader, writer) = tokio::io::split(client);
    let (events_tx, mut events_rx) = mpsc::unbounded_channel();
    let (prompts_tx, prompts_rx) = mpsc::unbounded_channel();
    for prompt in prompts {
        let _ = prompts_tx.send(prompt);
    }
    drop(prompts_tx);
    let mut peer = Peer::new(reader, writer, events_tx, agent);
    let ended = converse(
        &mut peer,
        Opening {
            dir: PathBuf::from("/work"),
            system: system.to_owned(),
            model: "model-1".to_owned(),
            effort: None,
            mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
            mcp_token: "secret".to_owned(),
            tools: vec!["read".to_owned()],
        },
        prompts_rx,
    )
    .await;
    drop(peer);
    let _ = task.await;
    let mut events = Vec::new();
    while let Ok(event) = events_rx.try_recv() {
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

/// Runs two prompts against a scripted agent that requires sign-in and sends
/// thoughts, text, and permission requests.
async fn conversation() -> (Vec<Event>, Vec<Value>) {
    let mut signed_in = false;
    let mut asked = 100u64;
    let prompt = |text: &str| Prompt {
        parts: vec![Part::Text(text.to_owned())],
        effort: None,
    };
    run_acp(
        Arc::new(Plain::basic()),
        "Be brief.",
        vec![prompt("one"), prompt("two")],
        move |message| {
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            match message.get("method").and_then(Value::as_str) {
                Some("initialize") => vec![json!({ "id": id, "result": { "protocolVersion": 1 } })],
                Some("session/new") if !signed_in => vec![json!({ "id": id,
                    "error": { "code": -32000, "message": "Authentication required" } })],
                Some("authenticate") => {
                    signed_in = true;
                    vec![json!({ "id": id, "result": {} })]
                }
                Some("session/new") => vec![json!({ "id": id, "result": { "sessionId": "s1" } })],
                Some("session/set_model") => vec![json!({ "id": id, "result": {} })],
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
                    let mut out = vec![
                        update("agent_thought_chunk", "thinking"),
                        update("agent_message_chunk", "Hello"),
                        update("tool_call", "ignored"),
                    ];
                    for title in ["rig", "terminal"] {
                        asked += 1;
                        out.push(json!({ "id": asked, "method": "session/request_permission", "params": {
                            "sessionId": "s1", "toolCall": { "title": title }, "options": options,
                        } }));
                    }
                    out.push(json!({ "id": 999, "method": "fs/read_text_file", "params": {} }));
                    out.push(json!({ "id": id, "result": {
                        "stopReason": "end_turn",
                        "usage": { "inputTokens": 10, "outputTokens": 5, "cachedReadTokens": 4 },
                    } }));
                    out
                }
                _ => vec![],
            }
        },
    )
    .await
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
    let (events, _) = run_acp(
        Arc::new(Plain::basic()),
        "",
        vec![Prompt {
            parts: vec![Part::Text("go".to_owned())],
            effort: None,
        }],
        |message| {
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            match message.get("method").and_then(Value::as_str) {
                Some("initialize" | "session/set_model") => vec![json!({ "id": id, "result": {} })],
                Some("session/new") => vec![json!({
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
                Some("session/prompt") => vec![
                    json!({
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 0, "size": 1048576 } }
                    }),
                    json!({
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 100000, "size": 1048576 } }
                    }),
                    json!({
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 105000, "size": 1048576 } }
                    }),
                    json!({
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
            }
        },
    )
    .await;

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
    let prompt = |text: &str, effort: Option<Effort>| Prompt {
        parts: vec![Part::Text(text.to_owned())],
        effort,
    };
    let (_, received) = run_acp(
        Arc::new(Plain {
            dynamic_effort: true,
            cumulative_usage: false,
        }),
        "",
        vec![
            prompt("first", None),
            prompt("second", Some(Effort::Low)),
            prompt("third", Some(Effort::Low)),
            prompt("fourth", None),
        ],
        |msg| {
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            match msg.get("method").and_then(Value::as_str) {
                Some("initialize" | "session/set_model" | "session/set_config_option") => {
                    vec![json!({ "id": id, "result": {} })]
                }
                Some("session/new") => vec![json!({
                    "id": id,
                    "result": {
                        "sessionId": "s1",
                        "configOptions": [{ "id": "reasoning_effort", "currentValue": "medium" }]
                    }
                })],
                Some("session/prompt") => {
                    vec![json!({ "id": id, "result": { "stopReason": "end_turn" } })]
                }
                _ => vec![],
            }
        },
    )
    .await;

    let updates: Vec<&str> = sent(&received, "session/set_config_option")
        .into_iter()
        .filter_map(|msg| msg.pointer("/params/value").and_then(Value::as_str))
        .collect();
    // Turn 1: default (no-op), Turn 2: low, Turn 3: low (deduped), Turn 4: default -> restored to medium.
    assert_eq!(updates, vec!["low", "medium"]);
}

#[tokio::test]
async fn cumulative_usage_agent_emits_per_turn_deltas_and_handles_resets() {
    let prompt = |text: &str| Prompt {
        parts: vec![Part::Text(text.to_owned())],
        effort: None,
    };
    let mut turn = 0usize;
    let (events, _) = run_acp(
        Arc::new(Plain {
            dynamic_effort: false,
            cumulative_usage: true,
        }),
        "",
        vec![prompt("one"), prompt("two"), prompt("three")],
        move |msg| {
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            match msg.get("method").and_then(Value::as_str) {
                Some("initialize" | "session/set_model") => vec![json!({ "id": id, "result": {} })],
                Some("session/new") => vec![json!({ "id": id, "result": { "sessionId": "s1" } })],
                Some("session/prompt") => {
                    turn += 1;
                    let usage = match turn {
                        // Turn 1: cumulative = 22139 in / 76 out / 17982 cached / 4145 cache_write
                        1 => json!({
                            "inputTokens": 22139,
                            "outputTokens": 76,
                            "cachedReadTokens": 17982,
                            "cachedWriteTokens": 4145,
                        }),
                        // Turn 2: cumulative grew by 5660 in / 17 out / 5588 cached / 69 cache_write
                        2 => json!({
                            "inputTokens": 27799,
                            "outputTokens": 93,
                            "cachedReadTokens": 23570,
                            "cachedWriteTokens": 4214,
                        }),
                        // Turn 3: counter reset lower than Turn 2 -> pass through unchanged
                        _ => json!({
                            "inputTokens": 6000,
                            "outputTokens": 20,
                            "cachedReadTokens": 5000,
                            "cachedWriteTokens": 100,
                        }),
                    };
                    vec![
                        json!({ "id": id, "result": { "stopReason": "end_turn", "usage": usage } }),
                    ]
                }
                _ => vec![],
            }
        },
    )
    .await;

    let usages: Vec<_> = events
        .into_iter()
        .filter_map(|ev| match ev {
            Event::Usage(usage) => Some((
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                usage.total_tokens,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        usages,
        vec![
            (Some(22139), Some(76), Some(17982), Some(4145), Some(22215)),
            (Some(5660), Some(17), Some(5588), Some(69), Some(5677)),
            (Some(6000), Some(20), Some(5000), Some(100), Some(6020)),
        ]
    );
}
