use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};

use rig_basics::coding_tools::ReadTool;
use rig_basics::telemetry::effect_log::{EffectLog, EffectLogPlugin};
use rig_core::message::Text;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, collect, first_agent, log_ended, run_until, tui_view,
};
use rig_tui::{MainPane, TranscriptLayout, TranscriptView};
use serde_json::{Value, json};

use super::*;

/// The first reply's blob, in the two pieces it streams in.
const B1: [&str; 2] = ["B1-sig-first-half.", "B1-sig-second-half"];
/// The blob of a later reply, which the provider then rejects.
const B2: &str = "B2-sig-the-provider-rejects";
const SUMMARY: &str = "checking the file";

/// What a fake provider answers each request body with.
type Answer = Box<dyn FnMut(&str) -> (u16, String) + Send>;

/// A fake provider on a local port, and the bodies of the requests it got.
struct Fake {
    address: SocketAddr,
    bodies: Arc<Mutex<Vec<String>>>,
}

impl Fake {
    fn start(mut answer: Answer) -> Option<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        let address = listener.local_addr().ok()?;
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&bodies);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                serve(stream, &kept, &mut answer).ok();
            }
        });
        Some(Self { address, bodies })
    }

    fn bodies(&self) -> Vec<String> {
        self.bodies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn last(&self) -> String {
        self.bodies().pop().unwrap_or_default()
    }
}

fn serve(
    mut stream: TcpStream,
    bodies: &Mutex<Vec<String>>,
    answer: &mut Answer,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8_lossy(&body).into_owned();
    let (status, reply) = answer(&body);
    bodies
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(body);
    let kind = if status == 200 {
        "text/event-stream"
    } else {
        "application/json"
    };
    write!(
        stream,
        "HTTP/1.1 {status} X\r\ncontent-type: {kind}\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{reply}",
        reply.len()
    )?;
    stream.flush()
}

/// An Anthropic Messages stream of `blocks`, each a content block's start
/// and its deltas, ending for `stop`.
fn messages_stream(blocks: &[(Value, Vec<Value>)], stop: &str) -> String {
    let start = json!({"type": "message_start", "message": {"id": "msg_1", "type": "message",
        "role": "assistant", "model": "think", "content": [], "stop_reason": null,
        "usage": {"input_tokens": 10, "output_tokens": 1}}});
    let mut events = vec![start];
    for (index, (block, deltas)) in blocks.iter().enumerate() {
        events.push(json!({"type": "content_block_start", "index": index, "content_block": block}));
        for delta in deltas {
            events.push(json!({"type": "content_block_delta", "index": index, "delta": delta}));
        }
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    events.push(
        json!({"type": "message_delta", "delta": {"stop_reason": stop},
        "usage": {"output_tokens": 5}}),
    );
    events.push(json!({"type": "message_stop"}));
    events
        .iter()
        .map(|event| {
            let kind = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect()
}

/// A thinking block with `summary`, signed by `pieces`.
fn thinking(summary: &str, pieces: &[&str]) -> (Value, Vec<Value>) {
    let mut deltas = vec![json!({"type": "thinking_delta", "thinking": summary})];
    deltas.extend(
        pieces
            .iter()
            .map(|piece| json!({"type": "signature_delta", "signature": piece})),
    );
    (
        json!({"type": "thinking", "thinking": "", "signature": ""}),
        deltas,
    )
}

fn text(text: &str) -> (Value, Vec<Value>) {
    (
        json!({"type": "text", "text": ""}),
        vec![json!({"type": "text_delta", "text": text})],
    )
}

/// The reasoning provider: rejects any request that carries [`B2`], and
/// answers the others in order.
fn thinker() -> Answer {
    let read = (
        json!({"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}),
        vec![json!({"type": "input_json_delta", "partial_json": "{\"path\":\"notes.txt\"}"})],
    );
    let mut replies: VecDeque<String> = [
        messages_stream(&[thinking(SUMMARY, &B1), read], "tool_use"),
        messages_stream(&[text("It says hello.")], "end_turn"),
        messages_stream(&[text("Still here.")], "end_turn"),
        messages_stream(
            &[thinking("second thought", &[B2]), text("Second.")],
            "end_turn",
        ),
        messages_stream(&[text("Recovered.")], "end_turn"),
    ]
    .into();
    Box::new(move |body| {
        if body.contains(B2) {
            let error = json!({"type": "error", "error": {"type": "invalid_request_error",
                "message": "messages.7.content.0: Invalid `signature` in `thinking` block"}});
            return (400, error.to_string());
        }
        (200, replies.pop_front().unwrap_or_default())
    })
}

/// A local Chat Completions model that takes reasoning back as text.
fn local() -> Answer {
    let chunk = |delta: Value, finish: Value| {
        json!({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "plain",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let stream = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk(
            json!({"role": "assistant", "content": "Fine."}),
            Value::Null
        ),
        chunk(json!({}), json!("stop")),
    );
    Box::new(move |_| (200, stream.clone()))
}

/// A Chat Completions stream of `deltas`, ending for `finish`.
fn chat_stream(deltas: &[Value], finish: &str) -> String {
    let chunk = |delta: &Value, finish: Value| {
        json!({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "think",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let mut frames: Vec<Value> = deltas
        .iter()
        .map(|delta| chunk(delta, Value::Null))
        .collect();
    frames.push(chunk(&json!({}), json!(finish)));
    frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .chain(["data: [DONE]\n\n".to_owned()])
        .collect()
}

/// Thinking `summary` as a Chat gateway in front of Anthropic's API
/// streams it, closed by `signature`.
fn gateway_thinking(summary: &str, signature: &str) -> [Value; 2] {
    [
        json!({"role": "assistant", "reasoning_content": summary,
            "thinking_blocks": [{"type": "thinking", "thinking": summary}]}),
        json!({"reasoning_content": "", "thinking_blocks": [{"type": "thinking",
            "thinking": "", "signature": signature}]}),
    ]
}

/// The reasoning model behind a Chat gateway: rejects any request that
/// carries [`B2`], and answers the others in order.
fn gateway() -> Answer {
    let [summary, signed] = gateway_thinking(SUMMARY, &B1.concat());
    let read = json!({"tool_calls": [{"index": 0, "id": "toolu_1", "type": "function",
        "function": {"name": "read", "arguments": "{\"path\":\"notes.txt\"}"}}]});
    let [second, resigned] = gateway_thinking("second thought", B2);
    let mut replies: VecDeque<String> = [
        chat_stream(&[summary, signed, read], "tool_calls"),
        chat_stream(&[json!({"content": "It says hello."})], "stop"),
        chat_stream(&[second, resigned, json!({"content": "Second."})], "stop"),
        chat_stream(&[json!({"content": "Recovered."})], "stop"),
    ]
    .into();
    Box::new(move |body| {
        if body.contains(B2) {
            let error = json!({"error": {"type": "invalid_request_error", "code": "400",
                "message": "messages.5.content.0: Invalid `signature` in `thinking` block"}});
            return (400, error.to_string());
        }
        (200, replies.pop_front().unwrap_or_default())
    })
}

/// The plugin's table: the reasoning provider and the local model.
fn settings(thinker: &Fake, local: &Fake) -> String {
    format!(
        r#"
        [[opaque_reasoning.provider]]
        name = "thinker"
        base_url = "http://{}"
        api = "anthropic"
        models = ["think"]

        [[opaque_reasoning.provider]]
        name = "local"
        base_url = "http://{}/v1"
        api = "openai-chat"
        models = ["plain"]
        reasoning_field = "reasoning_content"
        "#,
        thinker.address, local.address
    )
}

/// A project with `notes.txt`, beside the test binary.
fn project() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name("oc-opaque-reasoning-project");
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("notes.txt"), "hello\n").ok();
    dir
}

/// The kernel, `read`, the effect log, the terminal view and the plugin
/// on `store`, after the first frame restored the session, and its agent.
fn start(store: &MemoryStore, settings: &str) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(project()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, EffectLogPlugin))
        .add_plugins(OpaqueReasoningPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<StreamDelta>(&mut app);
    collect::<Notice>(&mut app);
    tui_view(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// Sends `text` to `agent` and runs until its turn ended; how it ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) -> String {
    let before = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
    match app.world().resource::<Ended>().0.last() {
        Some(TurnOutcome::Answered(message)) => message.final_answer().unwrap_or_default(),
        other => format!("{other:?}"),
    }
}

fn switch(app: &mut App, agent: Entity, model: &str) {
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model.to_owned(),
    });
    app.update();
}

/// The rows of `agent`'s transcript as the terminal view lays it out.
fn transcript(app: &mut App, agent: Entity) -> String {
    let world = app.world_mut();
    let mut panes = world.query_filtered::<&TranscriptView, With<MainPane>>();
    let view = panes.single(world).ok().cloned();
    let drawn = view.and_then(|view| TranscriptLayout::lay_out(world, &view, agent, 120));
    let rows = drawn.iter().flat_map(|drawn| drawn.rows().iter());
    rows.map(|row| row.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn reasoning_blobs_go_back_to_their_model_alone_and_never_stale() {
    let (thinker, local) = (Fake::start(thinker()), Fake::start(local()));
    assert!(
        thinker.is_some() && local.is_some(),
        "the fake providers need local ports"
    );
    let (Some(thinker), Some(local)) = (thinker, local) else {
        return;
    };
    let settings = settings(&thinker, &local);
    let blob = B1.concat();
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &settings);
    switch(&mut app, agent, "thinker/think");

    // The summary streams into the reasoning view; the blob goes back with
    // the read's result, before the call.
    assert_eq!(
        ask(&mut app, &wakes, agent, "What do the notes say?"),
        "It says hello."
    );
    let deltas = &app.world().resource::<Collected<StreamDelta>>().0;
    let streamed: String = deltas
        .iter()
        .filter_map(|delta| match &delta.event {
            StreamEvent::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, SUMMARY);
    let with_result = thinker.last();
    let at_blob = with_result.find(&blob);
    let at_call = with_result.find(r#""type":"tool_use""#);
    assert!(
        at_blob.is_some() && at_blob < at_call && with_result.contains("tool_result"),
        "{with_result}"
    );

    // Resumed, the next request still carries it.
    drop(app);
    let (mut app, wakes, agent) = start(&store, &settings);
    assert_eq!(
        ask(&mut app, &wakes, agent, "Anything else?"),
        "Still here."
    );
    assert!(thinker.last().contains(&blob));

    // Another model gets the summary as reasoning, under the field it
    // reads reasoning back in, never the blob.
    switch(&mut app, agent, "local/plain");
    assert_eq!(ask(&mut app, &wakes, agent, "Your view?"), "Fine.");
    let other = local.last();
    assert!(!other.contains(B1[0]), "{other}");
    assert!(
        other.contains(&format!(r#""reasoning_content":"{SUMMARY}""#)),
        "{other}"
    );

    // Back on the reasoning model, a history edit of the message holding
    // the blob drops it.
    switch(&mut app, agent, "thinker/think");
    let edited = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .and_then(|mut conversation| {
            let at = conversation
                .entries()
                .find(|entry| blobs_in(entry.message) > 0)
                .map(|entry| entry.index)?;
            let mut entry = conversation.entry(at)?;
            if let ChatMessage::Assistant(turn) = std::sync::Arc::make_mut(&mut entry.message) {
                turn.content
                    .push(AssistantContent::Text(Text::new("Reading it now.")));
            }
            conversation.splice(at..=at, [entry], Author::plugin("history-edit"));
            Some(at)
        });
    assert!(edited.is_some());
    assert_eq!(ask(&mut app, &wakes, agent, "And now?"), "Second.");
    let after_edit = thinker.last();
    assert!(!after_edit.contains(B1[0]), "{after_edit}");
    assert!(after_edit.contains("Reading it now."), "{after_edit}");

    // The provider rejects the blob of that reply: sent again once without
    // blobs, with a notice.
    assert_eq!(ask(&mut app, &wakes, agent, "Last one."), "Recovered.");
    let bodies = thinker.bodies();
    let carried_b2 = bodies.iter().filter(|body| body.contains(B2)).count();
    assert_eq!((bodies.len(), carried_b2), (6, 1));
    assert!(!thinker.last().contains(B2));
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("sent again without blobs")),
        "{notices:?}"
    );

    // The transcript and the effect log show the blobs' size, never their
    // bytes.
    let shown = transcript(&mut app, agent);
    assert!(shown.contains("[encrypted reasoning, 36 bytes]"), "{shown}");
    assert!(
        shown.contains(SUMMARY) && !shown.contains(B1[0]) && !shown.contains(B2),
        "{shown}"
    );
    let records = app
        .world()
        .get_resource::<EffectLog>()
        .and_then(|log| log.read().ok())
        .unwrap_or_default();
    let logged = serde_json::to_string(&records).unwrap_or_default();
    assert!(records.len() > 6 && logged.contains("[encrypted reasoning"));
    assert!(!logged.contains(B1[0]) && !logged.contains(B2));
}

/// On a Chat gateway, signed thinking goes back with the call's result,
/// is left out once an earlier message is edited, and a request the model
/// rejects over a signature goes again without it.
#[test]
fn chat_thinking_blocks_are_left_out_once_stale_or_rejected() {
    let gateway = Fake::start(gateway());
    assert!(gateway.is_some(), "the fake gateway needs a local port");
    let Some(gateway) = gateway else {
        return;
    };
    let settings = format!(
        r#"
        [[opaque_reasoning.provider]]
        name = "gateway"
        base_url = "http://{}/v1"
        api = "openai-chat"
        models = ["think"]
        reasoning_field = "reasoning_content"
        "#,
        gateway.address
    );
    let blob = B1.concat();
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &settings);
    switch(&mut app, agent, "gateway/think");

    assert_eq!(
        ask(&mut app, &wakes, agent, "What do the notes say?"),
        "It says hello."
    );
    let with_result = gateway.last();
    assert!(
        with_result.contains(&blob) && with_result.contains(r#""role":"tool""#),
        "{with_result}"
    );

    // An edit of the question before the signed message leaves its
    // thinking out.
    let edited = app
        .world_mut()
        .get_mut::<Conversation>(agent)
        .and_then(|mut conversation| {
            let at = conversation
                .entries()
                .find(|entry| matches!(entry.message, ChatMessage::User { .. }))
                .map(|entry| entry.index)?;
            let mut entry = conversation.entry(at)?;
            entry.message = std::sync::Arc::new(ChatMessage::user("What does the file say?"));
            conversation.splice(at..=at, [entry], Author::plugin("history-edit"));
            Some(at)
        });
    assert!(edited.is_some());
    assert_eq!(ask(&mut app, &wakes, agent, "And now?"), "Second.");
    let after_edit = gateway.last();
    assert!(
        !after_edit.contains(&blob) && after_edit.contains("What does the file say?"),
        "{after_edit}"
    );

    assert_eq!(ask(&mut app, &wakes, agent, "Last one."), "Recovered.");
    let bodies = gateway.bodies();
    let carried_b2 = bodies.iter().filter(|body| body.contains(B2)).count();
    assert_eq!((bodies.len(), carried_b2), (5, 1));
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("sent again without blobs")),
        "{notices:?}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(OpaqueReasoningPlugin);
}
