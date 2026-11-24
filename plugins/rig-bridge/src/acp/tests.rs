use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::rig_core::completion::{Effort, FinishReason};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use super::{Agent, Opening, Peer, Steering, converse};
use crate::harness::{ContextSnapshot, Event, Part, Prompt, Steer};

/// An agent with no system prompt channel, one sign-in and a model request.
struct Plain {
    dynamic_effort: bool,
    cumulative_usage: bool,
    /// How it takes a message for the prompt it is answering: by a request
    /// of its own, or (`true`) as the next prompt.
    steers: Option<bool>,
}

impl Plain {
    const fn basic() -> Self {
        Self {
            dynamic_effort: false,
            cumulative_usage: false,
            steers: None,
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

    fn steer(&self, _: &Value, session: &str, _: &[Part]) -> Option<Steering> {
        Some(match self.steers? {
            true => Steering::Prompt,
            false => Steering::Request("x/steer", json!({ "sessionId": session })),
        })
    }

    fn steered(&self, result: &Value) -> bool {
        result.get("status") == Some(&json!("queued"))
    }
}

/// What the scripted agent was sent, in order.
type Received = Arc<Mutex<Vec<Value>>>;

/// Runs `prompts` against a scripted ACP duplex responder, returning `(events, received)`.
async fn run_acp(
    agent: Arc<dyn Agent>,
    system: &str,
    prompts: Vec<Prompt>,
    responder: impl FnMut(&Value) -> Vec<Value> + Send + 'static,
) -> (Vec<Event>, Vec<Value>) {
    run_acp_steered(
        agent,
        system,
        prompts,
        mpsc::unbounded_channel().1,
        responder,
    )
    .await
}

/// As [`run_acp`], with `steers` for the prompts being answered.
async fn run_acp_steered(
    agent: Arc<dyn Agent>,
    system: &str,
    prompts: Vec<Prompt>,
    steers: mpsc::UnboundedReceiver<Steer>,
    responder: impl FnMut(&Value) -> Vec<Value> + Send + 'static,
) -> (Vec<Event>, Vec<Value>) {
    let stops = mpsc::unbounded_channel().1;
    run_acp_on(agent, system, None, prompts, steers, stops, responder).await
}

/// As [`run_acp_steered`], on the session saved as `resume` if any, with
/// `stops` for the prompts being answered.
async fn run_acp_on(
    agent: Arc<dyn Agent>,
    system: &str,
    resume: Option<&str>,
    prompts: Vec<Prompt>,
    steers: mpsc::UnboundedReceiver<Steer>,
    stops: mpsc::UnboundedReceiver<()>,
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
            resume: resume.map(str::to_owned),
            system: system.to_owned(),
            model: "model-1".to_owned(),
            effort: None,
            mcp_url: "http://127.0.0.1:1/mcp".to_owned(),
            mcp_token: "secret".to_owned(),
            tools: vec!["read".to_owned()],
        },
        prompts_rx,
        steers,
        stops,
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
    let (mut events, _) = conversation().await;
    // First the agent's name for the session it opened.
    assert!(matches!(events.first(), Some(Event::Saved(_))));
    events.remove(0);

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
            steers: None,
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
            steers: None,
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

/// One prompt to an agent that takes a message for it (`takes_over`: as the
/// next prompt), steered once the agent has the prompt: the events, what
/// the agent was sent, and what the steer's sender was told.
async fn steered(takes_over: bool) -> (Vec<Event>, Vec<Value>, Option<Result<(), String>>) {
    let (steers, steered) = mpsc::unbounded_channel();
    let (taken, mut held) = oneshot::channel();
    let mut steer = Some(Steer {
        parts: vec![Part::Text("and b".to_owned())],
        taken,
    });
    let mut first = Value::Null;
    let ends = |id: &Value, input: u64| {
        json!({ "id": id, "result": { "stopReason": "end_turn",
            "usage": { "inputTokens": input, "outputTokens": 1 } } })
    };
    let agent = Plain {
        steers: Some(takes_over),
        ..Plain::basic()
    };
    let prompt = Prompt {
        parts: vec![Part::Text("one".to_owned())],
        effort: None,
    };
    let (events, received) =
        run_acp_steered(Arc::new(agent), "", vec![prompt], steered, move |message| {
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            match message.get("method").and_then(Value::as_str) {
                Some("session/new") => vec![json!({ "id": id, "result": { "sessionId": "s1" } })],
                // The first prompt waits, as one with a tool call open does.
                Some("session/prompt") if first.is_null() => {
                    first = id;
                    if let Some(steer) = steer.take() {
                        let _ = steers.send(steer);
                    }
                    Vec::new()
                }
                // The agent ends the first prompt and goes on under this one.
                Some("session/prompt") => vec![ends(&first, 10), ends(&id, 20)],
                Some("x/steer") => vec![
                    json!({ "id": id, "result": { "status": "queued" } }),
                    ends(&first, 10),
                ],
                _ => vec![json!({ "id": id, "result": {} })],
            }
        })
        .await;
    (events, received, held.try_recv().ok())
}

#[tokio::test]
async fn a_steer_is_the_agent_s_own_request_beside_the_prompt() {
    let (events, received, taken) = steered(false).await;

    assert_eq!(taken, Some(Ok(())));
    assert_eq!(sent(&received, "x/steer").len(), 1);
    assert_eq!(sent(&received, "session/prompt").len(), 1);
    assert_eq!(events.last(), Some(&Event::Done(Ok(FinishReason::Stop))));
}

#[tokio::test]
async fn a_steer_sent_as_the_next_prompt_carries_the_turn_on_under_it() {
    let (events, received, taken) = steered(true).await;

    assert_eq!(taken, Some(Ok(())));
    let prompts = sent(&received, "session/prompt");
    assert_eq!(
        prompts
            .last()
            .and_then(|sent| sent.pointer("/params/prompt")),
        Some(&json!([{ "type": "text", "text": "and b" }])),
    );
    // The first prompt's end is not the turn's: both are counted, one ends.
    let counted: Vec<Option<u64>> = events
        .iter()
        .filter_map(|event| match event {
            Event::Usage(usage) => Some(usage.input_tokens),
            _ => None,
        })
        .collect();
    let ends = |event: &&Event| matches!(event, Event::Done(_));
    assert_eq!(counted, [Some(10), Some(20)]);
    assert_eq!(events.iter().filter(ends).count(), 1);
    assert_eq!(events.last(), Some(&Event::Done(Ok(FinishReason::Stop))));
}

/// The agent says `text`.
fn says(text: &str) -> Value {
    json!({ "method": "session/update", "params": { "sessionId": "s9", "update": {
        "sessionUpdate": "agent_message_chunk",
        "content": { "type": "text", "text": text },
    } } })
}

#[tokio::test]
async fn a_saved_session_is_resumed_or_loaded_and_what_it_reports_again_is_no_answer() {
    let able = [
        (
            json!({ "loadSession": true, "sessionCapabilities": { "resume": {} } }),
            "session/resume",
        ),
        (json!({ "loadSession": true }), "session/load"),
    ];
    for (able, opens) in able {
        let prompt = Prompt {
            parts: vec![Part::Text("two".to_owned())],
            effort: None,
        };
        let (events, received) = run_acp_on(
            Arc::new(Plain::basic()),
            "Be brief.",
            Some("s9"),
            vec![prompt],
            mpsc::unbounded_channel().1,
            mpsc::unbounded_channel().1,
            move |message| {
                let id = message.get("id").cloned().unwrap_or(Value::Null);
                match message.get("method").and_then(Value::as_str) {
                    Some("initialize") => {
                        vec![json!({ "id": id, "result": { "agentCapabilities": able } })]
                    }
                    // An agent that loads a session says all of it again.
                    Some("session/load") => {
                        vec![says("echo: one"), json!({ "id": id, "result": null })]
                    }
                    Some("session/prompt") => vec![
                        says("echo: two"),
                        json!({ "id": id, "result": { "stopReason": "end_turn" } }),
                    ],
                    _ => vec![json!({ "id": id, "result": {} })],
                }
            },
        )
        .await;

        assert_eq!(
            events,
            [
                Event::Saved("s9".to_owned()),
                Event::Text("echo: two".to_owned()),
                Event::Done(Ok(FinishReason::Stop)),
            ],
            "{opens}"
        );
        let opened = sent(&received, opens);
        let named = opened
            .first()
            .and_then(|sent| sent.pointer("/params/sessionId"));
        assert_eq!(named, Some(&json!("s9")), "{opens}");
        assert!(sent(&received, "session/new").is_empty());
        // The saved session has the system prompt: it is not sent again.
        assert_eq!(
            sent(&received, "session/prompt")
                .first()
                .and_then(|sent| sent.pointer("/params/prompt")),
            Some(&json!([{ "type": "text", "text": "two" }])),
        );
    }
}

#[tokio::test]
async fn a_stop_cancels_the_prompt_being_answered() {
    let (stops, stopped) = mpsc::unbounded_channel();
    let mut asked = Value::Null;
    let prompt = Prompt {
        parts: vec![Part::Text("one".to_owned())],
        effort: None,
    };
    let (events, received) = run_acp_on(
        Arc::new(Plain::basic()),
        "",
        None,
        vec![prompt],
        mpsc::unbounded_channel().1,
        stopped,
        move |message| {
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            match message.get("method").and_then(Value::as_str) {
                Some("session/new") => vec![json!({ "id": id, "result": { "sessionId": "s1" } })],
                // The prompt waits, and the user stops it.
                Some("session/prompt") => {
                    asked = id;
                    let _ = stops.send(());
                    Vec::new()
                }
                Some("session/cancel") => {
                    vec![json!({ "id": asked, "result": { "stopReason": "cancelled" } })]
                }
                _ => vec![json!({ "id": id, "result": {} })],
            }
        },
    )
    .await;

    let cancelled = sent(&received, "session/cancel");
    assert_eq!(
        cancelled.first().and_then(|sent| sent.get("params")),
        Some(&json!({ "sessionId": "s1" })),
    );
    assert_eq!(
        events.last(),
        Some(&Event::Done(Err("the turn was cancelled".to_owned()))),
    );
}
