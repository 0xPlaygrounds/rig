use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::Receiver;

use bevy_time::TimeUpdateStrategy;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_core::providers::registry::{ConnectOptions, ProviderId};
use rig_harness_test_support::{
    Collected, Ended, advance, app_in, collect, first_agent, log_ended, run_until, scratch_home,
};
use rig_tui::{StatusItem, StatusItems, Tone, TuiPlugin};

use super::status::QUOTA;
use super::*;

/// A streamed OpenAI-compatible answer.
const STREAM: &str = concat!(
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":",
    "{\"role\":\"assistant\",\"content\":\"Done.\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},",
    "\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// A reply a stand-in gives: its status line and extra headers.
type Script = (&'static str, String);

/// A good reply with OpenAI's rate-limit headers.
fn limits(requests: u32, tokens: u32) -> Script {
    let headers = format!(
        "x-ratelimit-limit-requests: 50\r\nx-ratelimit-remaining-requests: {requests}\r\n\
         x-ratelimit-reset-requests: 34s\r\nx-ratelimit-limit-tokens: 40000\r\n\
         x-ratelimit-remaining-tokens: {tokens}\r\nx-ratelimit-reset-tokens: 34s\r\n"
    );
    ("200 OK", headers)
}

/// A local stand-in that answers each request with the next of `script`
/// (a plain good reply once it runs out), and counts the requests.
fn stand_in(script: Vec<Script>) -> Option<(SocketAddr, Arc<Mutex<usize>>)> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let address = listener.local_addr().ok()?;
    let seen = Arc::new(Mutex::new(0));
    let counted = seen.clone();
    std::thread::spawn(move || {
        let mut script = VecDeque::from(script);
        for stream in listener.incoming().flatten() {
            let next = script.pop_front().unwrap_or(("200 OK", String::new()));
            if let Ok(mut seen) = counted.lock() {
                *seen += 1;
            }
            answer(stream, next).ok();
        }
    });
    Some((address, seen))
}

fn answer(mut stream: TcpStream, (status, headers): Script) -> std::io::Result<()> {
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
    reader.read_exact(&mut vec![0; length])?;
    let (kind, body) = if status.starts_with("200") {
        ("text/event-stream", STREAM)
    } else {
        ("application/json", r#"{"error":{"message":"slow down"}}"#)
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\n{headers}content-type: {kind}\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

/// The agent's quota item, if shown.
fn item(app: &App, agent: Entity) -> Option<StatusItem> {
    let items = app.world().get::<StatusItems>(agent)?;
    items.iter().find(|item| item.name == QUOTA.name).cloned()
}

/// Sends `text` and runs until `turns` turns ended, then a frame more.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
}

fn requests(seen: &Mutex<usize>) -> usize {
    seen.lock().map(|seen| *seen).unwrap_or_default()
}

#[test]
fn the_item_follows_the_providers_limits_and_the_model() {
    let limited = ("429 Too Many Requests", "retry-after: 12\r\n".to_owned());
    let throttled = stand_in(vec![limits(42, 18_000), limits(41, 1_000), limited]);
    let silent = stand_in(Vec::new());
    assert!(
        throttled.is_some() && silent.is_some(),
        "local ports for the stand-ins"
    );
    let (Some((throttled, seen_throttled)), Some((silent, seen_silent))) = (throttled, silent)
    else {
        return;
    };

    let home = scratch_home("oc-rate-limit-status");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        TuiPlugin,
        RateLimitStatusPlugin,
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<Notice>(&mut app);
    let mut models = app.world_mut().resource_mut::<Models>();
    for (vendor, address) in [("deepseek", throttled), ("groq", silent)] {
        let provider = ProviderId::catalog(vendor);
        let options = ConnectOptions::new()
            .base_url(format!("http://{address}/v1"))
            .api_key(format!("sk-test-{vendor}-0000"));
        provider
            .into_iter()
            .for_each(|id| models.0.provide(id, options.clone()));
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "deepseek/deepseek-flash".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.update();
    assert_eq!(item(&app, agent), None);

    // 42 of 50 requests and 18k of 40k tokens left: plenty.
    turn(&mut app, &wakes, agent, "One.", 1);
    let shown = item(&app, agent).unwrap_or_default();
    for part in ["42/50", "18k/40k", "34s"] {
        assert!(shown.text.contains(part), "{shown:?}");
    }
    assert_eq!(shown.tone, Tone::Plain);

    // 1k of 40k tokens: the tighter allowance is nearly spent.
    turn(&mut app, &wakes, agent, "Two.", 2);
    let shown = item(&app, agent).unwrap_or_default();
    assert!(shown.text.contains("tok 1k/40k"), "{shown:?}");
    assert_eq!(shown.tone, Tone::Red);

    // A 429 asking for 12 s: limited, counting down on the clock.
    app.world_mut()
        .trigger(Deliver::new(agent, "Three.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        let items = world.get::<StatusItems>(agent);
        let items = items.into_iter().flat_map(StatusItems::iter);
        items.into_iter().any(|item| item.text.contains("limited"))
    });
    let shown = item(&app, agent).unwrap_or_default();
    assert!(
        shown.text.contains("limited") && shown.text.contains("12s"),
        "{shown:?}"
    );
    assert_eq!(shown.tone, Tone::Red);
    advance(&mut app, Duration::from_secs(5));
    let shown = item(&app, agent).unwrap_or_default();
    assert!(shown.text.contains("retry in 7s"), "{shown:?}");

    // The retry, once the wait passed, is answered: the limit is over.
    advance(&mut app, Duration::from_secs(8));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 3
    });
    app.update();
    let shown = item(&app, agent).unwrap_or_default();
    assert!(shown.text.contains("tok 1k/40k"), "{shown:?}");

    // Another model, whose provider sends no limits: hidden.
    let model = "groq/llama-3.3-70b-versatile".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.update();
    assert_eq!(item(&app, agent), None);
    turn(&mut app, &wakes, agent, "Four.", 4);
    assert_eq!(item(&app, agent), None);

    app.world_mut().trigger(RunCommand::new(agent, "quota"));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let listed = notices.iter().rev().find(|n| n.text.contains("received"));
    let listed = listed
        .map(|notice| notice.text.as_str())
        .unwrap_or_default();
    assert!(
        listed.contains("deepseek …0000: req 41/50 · tok 1k/40k"),
        "{notices:?}"
    );
    assert!(!listed.contains("groq"), "{listed}");

    // Only the turns' own requests: three and a retry to the first, one to
    // the second.
    assert_eq!(requests(&seen_throttled), 4);
    assert_eq!(requests(&seen_silent), 1);
}
