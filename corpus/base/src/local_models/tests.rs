use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rig_basics::models::ModelsPlugin;
use rig_harness::rig_core::catalog::ModelStatus;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, first_agent, log_ended, run_until};

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

/// An Ollama daemon on 127.0.0.1 serving `qwen-coder` (32768 tokens) and
/// `tiny` (no tools), one connection at a time, recording the paths asked
/// for and the bodies sent to them.
struct Stub {
    url: String,
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    asked: Arc<Mutex<Vec<(String, String)>>>,
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
            .is_ok_and(|asked| asked.iter().any(|(it, _)| it == path))
    }

    /// Whether a chat request told the model it has no tools.
    fn told_no_tools(&self) -> bool {
        self.asked.lock().is_ok_and(|asked| {
            asked
                .iter()
                .any(|(path, body)| path == "/v1/chat/completions" && body.contains(NO_TOOLS))
        })
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
fn answer(mut stream: TcpStream, asked: &Mutex<Vec<(String, String)>>) -> std::io::Result<()> {
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
    let body = String::from_utf8_lossy(&body).into_owned();
    let (kind, reply) = match path.as_str() {
        "/api/tags" => (
            "application/json",
            r#"{"models":[{"name":"qwen-coder"},{"name":"tiny"}]}"#.to_owned(),
        ),
        "/api/show" if body.contains("tiny") => (
            "application/json",
            r#"{"capabilities":["completion"],"model_info":{}}"#.to_owned(),
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
        asked.push((path, body));
    }
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    )?;
    stream.flush()
}

fn available(world: &World, name: &str) -> Option<bool> {
    world
        .resource::<LocalModels>()
        .get(name)
        .map(|model| model.available)
}

/// What the scripted run saw.
#[derive(Debug, PartialEq)]
struct Seen {
    model: Option<String>,
    answer: Option<String>,
    window: Option<u32>,
    warmed_up: bool,
    told_no_tools: bool,
    /// What the catalog says of the model once its server stopped.
    status_after_stop: Option<ModelStatus>,
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
            told_no_tools: false,
            status_after_stop: Some(ModelStatus::Unavailable(format!(
                "Ollama not running at {url}"
            ))),
            failed_after_stop: Some(format!(
                "The model call failed: Ollama not running at {url}"
            )),
        })
    );
}

#[test]
fn a_local_model_without_tools_is_told_so() {
    let told = Stub::start()
        .map_err(|error| error.to_string())
        .and_then(|mut stub| {
            let (mut app, wakes, agent) = started(&stub)?;
            choose(&mut app, &wakes, agent, "tiny");
            say(&mut app, &wakes, agent, "hello", 1);
            stub.stop();
            Ok(stub.told_no_tools())
        });
    assert_eq!(told, Ok(true));
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

/// An app that probes `stub`, once it found its models, and its agent.
fn started(stub: &Stub) -> Result<(App, Receiver<()>, Entity), String> {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let settings = format!(
        "[local_models]\nendpoints = [{{ server = \"ollama\", url = \"{}\" }}]",
        stub.url
    );
    app.insert_resource(Settings::from_toml(&settings))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ModelsPlugin,
            LocalModelsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    run_until(&mut app, &wakes, |world| {
        available(world, "tiny") == Some(true)
    });
    let agent = first_agent(app.world_mut()).ok_or("no agent")?;
    Ok((app, wakes, agent))
}

/// `/model local/<name>`, until the agent is connected to it.
fn choose(app: &mut App, wakes: &Receiver<()>, agent: Entity, name: &str) {
    let line = format!("/model local/{name}");
    app.world_mut()
        .trigger(Deliver::new(agent, line, DeliveryMode::Queue));
    let model = format!("local/{name}");
    run_until(app, wakes, |world| {
        world
            .get::<Connection>(agent)
            .is_some_and(|connection| connection.spec.reference() == model)
    });
}

/// Sends `text`, until `turns` turns have ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() == turns
    });
}

fn observe(stub: &mut Stub) -> Result<Seen, String> {
    let (mut app, wakes, agent) = started(stub)?;
    choose(&mut app, &wakes, agent, "qwen-coder");
    let model = app
        .world()
        .get::<Connection>(agent)
        .map(|connection| connection.spec.reference());

    say(&mut app, &wakes, agent, "hello", 1);
    let answer = match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
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
        available(world, "qwen-coder") == Some(false)
    });
    let models = &app.world().resource::<Models>().0;
    let status_after_stop = models
        .catalog()
        .resolve("local/qwen-coder")
        .ok()
        .map(|found| models.status(found.spec));

    say(&mut app, &wakes, agent, "hello again", 2);
    let failed_after_stop = match app.world().resource::<Ended>().0.as_slice() {
        [_, TurnOutcome::Failed(why)] => Some(why.clone()),
        _ => None,
    };
    Ok(Seen {
        model,
        answer,
        window,
        warmed_up,
        told_no_tools: stub.told_no_tools(),
        status_after_stop,
        failed_after_stop,
    })
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LocalModelsPlugin);
}
