use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::Receiver;

use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, run_until, scratch_home,
};

use super::*;

/// The three keys' values, which must never show.
const SECRETS: [&str; 3] = ["sk-personal-1111", "sk-team-2222", "sk-spare-3333"];

/// The spare key's streamed answer, 100 tokens in and 20 out.
const STREAM: &str = concat!(
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"deepseek-flash\",\"choices\":[{\"index\":0,\"delta\":",
    "{\"role\":\"assistant\",\"content\":\"From the spare.\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"deepseek-flash\",\"choices\":[{\"index\":0,\"delta\":{},",
    "\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,",
    "\"model\":\"deepseek-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,",
    "\"completion_tokens\":20,\"total_tokens\":120}}\n\n",
    "data: [DONE]\n\n",
);

/// A local OpenAI-compatible stand-in: the personal key is rate-limited
/// for 30 s, the team key refused, the spare answers. Counts the requests
/// per key in `seen`.
fn serve(listener: TcpListener, seen: Arc<Mutex<[u32; 3]>>) {
    for stream in listener.incoming().flatten() {
        answer(stream, &seen).ok();
    }
}

fn answer(mut stream: TcpStream, seen: &Mutex<[u32; 3]>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let (mut length, mut key) = (0, None);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            length = value.trim().parse().unwrap_or(0);
        } else if name.eq_ignore_ascii_case("authorization") {
            key = SECRETS
                .iter()
                .position(|secret| value.trim() == format!("Bearer {secret}"));
        }
    }
    reader.read_exact(&mut vec![0; length])?;
    if let (Some(key), Ok(mut seen)) = (key, seen.lock())
        && let Some(count) = seen.get_mut(key)
    {
        *count += 1;
    }
    let error = r#"{"error":{"message":"no"}}"#;
    let (status, extra, body) = match key {
        Some(0) => ("429 Too Many Requests", "retry-after: 30\r\n", error),
        Some(2) => ("200 OK", "content-type: text/event-stream\r\n", STREAM),
        _ => ("401 Unauthorized", "", error),
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\n{extra}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

/// Runs frames until `turns` turns ended, then one more for the journal.
fn until_ended(app: &mut App, wakes: &Receiver<()>, turns: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
}

/// The notices so far.
fn notices(app: &App) -> Vec<(NoticeLevel, String)> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .map(|notice| (notice.level, notice.text.clone()))
        .collect()
}

#[test]
fn a_rate_limited_and_a_refused_key_fail_over_to_the_spare() {
    let listener = TcpListener::bind("127.0.0.1:0");
    let address = listener.as_ref().ok().and_then(|l| l.local_addr().ok());
    assert!(address.is_some(), "a local port for the stand-in");
    let (Ok(listener), Some(address)) = (listener, address) else {
        return;
    };
    let seen = Arc::new(Mutex::new([0; 3]));
    let counts = seen.clone();
    std::thread::spawn(move || serve(listener, counts));

    let home = scratch_home("ho-api-key-pool");
    let id = SessionId::generate();
    std::fs::create_dir_all(home.root()).ok();
    let pool = format!(
        "[deepseek]\nbase_url = \"http://{address}/v1\"\n\
         keys = [{{ env = \"HO_KEY_POOL_PERSONAL\" }}, \
         {{ env = \"HO_KEY_POOL_TEAM\" }}, {{ file = \"spare.key\" }}]\n"
    );
    std::fs::write(home.root().join("key-pool.toml"), pool).ok();
    std::fs::write(home.root().join("spare.key"), SECRETS[2]).ok();
    // SAFETY: the crate's only test, and no other thread reads these names.
    unsafe {
        std::env::set_var("HO_KEY_POOL_PERSONAL", SECRETS[0]);
        std::env::set_var("HO_KEY_POOL_TEAM", SECRETS[1]);
    }

    let (mut app, wakes) = app_in(&home, &id);
    app.add_plugins((AgentPlugin, SessionLogPlugin, KeyPoolPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<EffectRecorded>(&mut app);
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

    // The turn ends with the spare's reply, with no failure and no wait.
    app.world_mut()
        .trigger(Deliver::new(agent, "Hello.", DeliveryMode::Steer));
    until_ended(&mut app, &wakes, 1);
    let ended = &app.world().resource::<Ended>().0;
    let answer = match ended.first() {
        Some(TurnOutcome::Answered(message)) => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("From the spare."), "{ended:?}");
    let shown = notices(&app);
    let said = |level| {
        let said = shown.iter().filter(move |(at, _)| *at == level);
        said.map(|(_, text)| text.as_str()).collect::<Vec<_>>()
    };
    assert_eq!(said(NoticeLevel::Error), Vec::<&str>::new());
    let warnings = said(NoticeLevel::Warn);
    assert!(
        warnings.len() == 1
            && warnings
                .iter()
                .all(|text| text.contains("HO_KEY_POOL_TEAM")),
        "{warnings:?}"
    );

    // The next turn goes straight to the spare.
    app.world_mut()
        .trigger(Deliver::new(agent, "And again.", DeliveryMode::Steer));
    until_ended(&mut app, &wakes, 2);
    let counts = seen.lock().map(|seen| *seen).unwrap_or_default();
    assert_eq!(counts, [1, 1, 2]);

    app.world_mut().trigger(RunCommand::new(agent, "keys"));
    app.update();
    let shown = notices(&app);
    let listed = shown
        .iter()
        .rev()
        .find(|(_, text)| text.contains("requests"));
    let line = |reference: &str| {
        let lines = listed.map(|(_, text)| text.lines()).into_iter().flatten();
        lines
            .filter(|line| line.contains(reference))
            .collect::<String>()
    };
    assert!(
        line("HO_KEY_POOL_PERSONAL").contains("cooling 3"),
        "{shown:?}"
    );
    assert!(line("HO_KEY_POOL_TEAM").contains("disabled"), "{shown:?}");
    assert!(
        line("spare.key").contains("ok, 2 requests, 240 tokens"),
        "{shown:?}"
    );

    // No key's value in a notice, an effect record or the session's files.
    let records = &app.world().resource::<Collected<EffectRecorded>>().0;
    let records: Vec<String> = records
        .iter()
        .map(|record| format!("{:?}", record.0))
        .collect();
    let mut saved = String::new();
    for entry in walk(home.session(&id).path()) {
        saved.push_str(&std::fs::read_to_string(entry).unwrap_or_default());
    }
    assert!(
        saved.contains("HO_KEY_POOL_TEAM"),
        "the usage is saved by reference"
    );
    for secret in SECRETS {
        assert!(!shown.iter().any(|(_, text)| text.contains(secret)));
        assert!(!records.iter().any(|record| record.contains(secret)));
        assert!(!saved.contains(secret));
    }
}

/// Every file under `dir`.
fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
    entries
        .flat_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walk(&path)
            } else {
                vec![path]
            }
        })
        .collect()
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(KeyPoolPlugin);
}
