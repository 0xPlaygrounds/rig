use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender, channel};

use rig_basics::models::ModelsPlugin;
use rig_basics::telemetry::{AgentUsage, UsagePlugin};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, log_ended, run_until};
use rig_tui::{CompletionSource, Editor, MainEditor};

use super::*;

/// The gateway's streamed answer: "hi", 1000 tokens in and 500 out.
const STREAM: &str = concat!(
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"big\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},",
    "\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"big\",",
    "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"big\",",
    "\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":500,",
    "\"total_tokens\":1500}}\n\n",
    "data: [DONE]\n\n",
);

/// Answers every request on `listener` with [`STREAM`], sending each
/// request's head through `heads`.
fn serve(listener: TcpListener, heads: Sender<String>) {
    for stream in listener.incoming().flatten() {
        answer(stream, &heads).ok();
    }
}

fn answer(mut stream: TcpStream, heads: &Sender<String>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut head = String::new();
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
        head.push_str(&line);
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    heads.send(head.to_ascii_lowercase()).ok();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{STREAM}",
        STREAM.len()
    )?;
    stream.flush()
}

/// The kernel, `/model`, the usage plugin and the plugin with `config` on
/// `store`, after the first frame restored the session, and its agent.
fn start(store: &MemoryStore, config: &str) -> (App, Receiver<()>, Option<Entity>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(CustomProviders::parse(config))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ModelsPlugin,
            UsagePlugin,
            CustomProviderPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    (app, wakes, agent)
}

/// The agent's model choice and its connected context window.
fn chosen(app: &App, agent: Entity) -> (Option<String>, Option<u32>) {
    let world = app.world();
    let model = world.get::<ModelChoice>(agent).map(|model| model.0.clone());
    let window = world
        .get::<Connection>(agent)
        .and_then(|connection| connection.spec.context_window);
    (model, window)
}

#[test]
fn a_configured_gateway_model_answers_with_its_headers_and_survives_a_restart() {
    let listener = TcpListener::bind("127.0.0.1:0");
    let port = listener.as_ref().ok().and_then(|l| l.local_addr().ok());
    assert!(port.is_some(), "a local port for the stub gateway");
    let (Ok(listener), Some(address)) = (listener, port) else {
        return;
    };
    let (sender, heads) = channel();
    std::thread::spawn(move || serve(listener, sender));
    let config = format!(
        r#"
        [[provider]]
        name = "corp"
        base_url = "http://{address}/v1"
        api = "openai-chat"
        key_command = "printf corp-token"
        headers = {{ "X-Corp-Team" = "rig" }}

        [[provider.models]]
        id = "big"
        context_window = 64000
        price = {{ input = 1.0, output = 2.0 }}

        [[provider]]
        name = "keyless"
        base_url = "http://{address}/v1"
        api = "anthropic"
        key_env = "CORPUS_CUSTOM_PROVIDER_UNSET_KEY"

        [[provider.models]]
        id = "small"
        "#
    );
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &config);
    assert!(agent.is_some(), "the session has an agent");
    let Some(agent) = agent else {
        return;
    };
    // Both models complete `/model`'s argument, once it is typed; the
    // keyless one says why.
    let mut typed = Editor::default();
    typed.set("/model ");
    app.world_mut().spawn((typed, MainEditor));
    app.update();
    let world = app.world_mut();
    let offered: Vec<(String, String)> = world
        .query::<&CompletionSource>()
        .iter(world)
        .flat_map(|completion| &completion.candidates)
        .filter(|item| item.text.starts_with("corp/") || item.text.starts_with("keyless/"))
        .map(|item| (item.text.clone(), item.detail.clone()))
        .collect();
    let texts: Vec<&str> = offered.iter().map(|(text, _)| text.as_str()).collect();
    assert_eq!(texts, ["corp/big", "keyless/small"], "{offered:?}");
    let keyless = offered.iter().find(|(text, _)| text == "keyless/small");
    assert!(
        keyless.is_some_and(
            |(_, detail)| detail.ends_with("(no key: set CORPUS_CUSTOM_PROVIDER_UNSET_KEY)")
        ),
        "{offered:?}"
    );

    app.world_mut()
        .trigger(RunCommand::new(agent, "model corp/big".to_owned()));
    app.update();
    assert_eq!(
        chosen(&app, agent),
        (Some("corp/big".to_owned()), Some(64_000))
    );

    app.world_mut()
        .trigger(Deliver::new(agent, "Say hi.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let answer = match app.world().resource::<Ended>().0.as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        other => Some(format!("{other:?}")),
    };
    assert_eq!(answer.as_deref(), Some("hi"));
    let head = heads.try_recv().unwrap_or_default();
    assert!(head.starts_with("post /v1/chat/completions "), "{head}");
    assert!(head.contains("x-corp-team: rig"), "{head}");
    assert!(head.contains("authorization: bearer corp-token"), "{head}");
    // 1000 tokens in at $1 and 500 out at $2 per million.
    let spent = app.world().get::<AgentUsage>(agent);
    let cost = spent
        .and_then(|spent| spent.models.get("corp/big"))
        .map(|spent| ((spent.cost - 0.002).abs() < 1e-9, spent.unpriced));
    assert_eq!(cost, Some((true, 0)), "{spent:?}");

    drop(app);
    let (mut app, _wakes, restored) = start(&store, &config);
    assert!(restored.is_some(), "the session's agent is restored");
    let Some(restored) = restored else {
        return;
    };
    assert_eq!(
        chosen(&app, restored),
        (Some("corp/big".to_owned()), Some(64_000))
    );

    // Choosing the keyless model is refused when it connects.
    app.world_mut()
        .trigger(RunCommand::new(restored, "model keyless/small".to_owned()));
    app.update();
    assert_eq!(
        chosen(&app, restored),
        (Some("keyless/small".to_owned()), None)
    );
}
