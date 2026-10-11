use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rig_cassette::journal::MemoryStore;
use rig_harness_test_support::{Ended, app_on, log_ended, run_until};
use rig_models::ModelsPlugin;

use super::*;

/// The streamed reply of the stub's chat endpoint: "hi", then the end.
const CHAT: &str = concat!(
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"qwen-coder\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",",
    "\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"qwen-coder\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
    "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n",
    "data: [DONE]\n\n",
);

/// An Ollama daemon on 127.0.0.1 serving `qwen-coder` (32768 tokens), one
/// connection at a time, recording the paths asked for.
struct Stub {
    url: String,
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    asked: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl Stub {
    fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let asked = Arc::new(Mutex::new(Vec::new()));
        let (stopped, log) = (stop.clone(), asked.clone());
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stopped.load(Ordering::Acquire) {
                    break;
                }
                if let Ok(stream) = stream {
                    answer(stream, &log).ok();
                }
            }
        });
        Ok(Self {
            url: format!("http://{address}"),
            address,
            stop,
            asked,
            thread: Some(thread),
        })
    }

    /// Whether `path` was asked for.
    fn asked(&self, path: &str) -> bool {
        self.asked
            .lock()
            .is_ok_and(|asked| asked.iter().any(|it| it == path))
    }

    /// Closes the port.
    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        TcpStream::connect(self.address).ok();
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Answers one request as Ollama would.
fn answer(mut stream: TcpStream, asked: &Mutex<Vec<String>>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut first = String::new();
    reader.read_line(&mut first)?;
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
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
    let path = first
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_owned();
    let (kind, reply) = match path.as_str() {
        "/api/tags" => (
            "application/json",
            r#"{"models":[{"name":"qwen-coder","model":"qwen-coder"}]}"#.to_owned(),
        ),
        "/api/show" => (
            "application/json",
            r#"{"capabilities":["completion","tools"],"model_info":{"general.architecture":"qwen2","qwen2.context_length":32768}}"#.to_owned(),
        ),
        "/api/generate" => ("application/json", r#"{"done":true}"#.to_owned()),
        "/v1/chat/completions" => ("text/event-stream", CHAT.to_owned()),
        _ => ("application/json", "{}".to_owned()),
    };
    if let Ok(mut asked) = asked.lock() {
        asked.push(path);
    }
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    )?;
    stream.flush()
}

fn qwen_available(world: &World) -> Option<bool> {
    world
        .resource::<LocalModels>()
        .get("qwen-coder")
        .map(|model| model.available)
}

/// What the scripted run saw.
#[derive(Debug, PartialEq)]
struct Seen {
    model: Option<String>,
    answer: Option<String>,
    window: Option<u32>,
    warmed_up: bool,
    available_after_stop: Option<bool>,
    /// Why a turn on the unavailable model failed.
    failed_after_stop: Option<String>,
}

#[test]
fn a_local_model_is_found_answers_and_is_marked_unavailable_once_its_server_stops() {
    let (seen, url) = run();
    assert_eq!(
        seen,
        Ok(Seen {
            model: Some("local/qwen-coder".to_owned()),
            answer: Some("hi".to_owned()),
            window: Some(32768),
            warmed_up: true,
            available_after_stop: Some(false),
            failed_after_stop: Some(format!("Ollama not running at {url}")),
        })
    );
}

/// Finds the stub's model, chooses it as `local/qwen-coder`, says hello,
/// stops the stub and refreshes.
fn run() -> (Result<Seen, String>, String) {
    let mut stub = match Stub::start() {
        Ok(stub) => stub,
        Err(error) => return (Err(error.to_string()), String::new()),
    };
    let url = stub.url.clone();
    (observe(&mut stub), url)
}

fn observe(stub: &mut Stub) -> Result<Seen, String> {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let settings = format!(
        "[local_models]\nendpoints = [{{ server = \"ollama\", url = \"{}\" }}]",
        stub.url
    );
    app.insert_resource(Settings::from_toml(&settings))
        .add_plugins((AgentPlugin, JournalPlugin, ModelsPlugin, LocalModelsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();

    run_until(&mut app, &wakes, |world| {
        qwen_available(world) == Some(true)
    });
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .ok_or("no agent")?;
    world.trigger(Deliver::new(
        agent,
        "/model local/qwen-coder",
        DeliveryMode::Queue,
    ));
    let model = |world: &World| {
        world
            .get::<Connection>(agent)
            .map(|connection| connection.spec.reference())
    };
    run_until(&mut app, &wakes, |world| model(world).is_some());
    let model = model(app.world());

    app.world_mut()
        .trigger(Deliver::new(agent, "hello", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let answer = match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => final_answer(message),
        other => return Err(format!("the turn ended as {other:?}")),
    };
    let window = app
        .world()
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.context_window);
    // The warm-up is sent beside the turn, in the background.
    run_until(&mut app, &wakes, |_| stub.asked("/api/generate"));
    let warmed_up = stub.asked("/api/generate");

    stub.stop();
    app.world_mut()
        .trigger(Deliver::new(agent, "/models refresh", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        qwen_available(world) == Some(false)
    });
    let available_after_stop = qwen_available(app.world());

    app.world_mut()
        .trigger(Deliver::new(agent, "hello again", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let failed_after_stop = match app.world().resource::<Ended>().0.as_slice() {
        [_, TurnOutcome::Failed(why)] => Some(why.clone()),
        _ => None,
    };
    Ok(Seen {
        model,
        answer,
        window,
        warmed_up,
        available_after_stop,
        failed_after_stop,
    })
}
