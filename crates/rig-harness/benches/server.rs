//! What the agent server costs (`cargo bench -p rig-harness --bench
//! server`): a journal batch written and fsynced, an idle frame of `rigd`
//! holding 200 sessions, a `rig/session/start` round trip over the socket,
//! and, with `RIG_BENCH_AGENT` naming a built agent binary, the time from a
//! start with a prompt to its `worker.ready` (the design's target is under
//! 300 ms). Each prints its median of a few runs.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rig_client::{Autostart, Client, Options as ClientOptions, Target};
use rig_harness::harness_protocol::Home;
use rig_harness::rig_memory::journal::Journal;
use rig_harness::server::{self, Options, Session, record::Track};
use rig_protocol::methods::{ReadEvents, Start};
use rig_protocol::{BuildId, EventRecord, Key, Seq, SessionId};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-bench-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn median(mut runs: Vec<Duration>) -> Duration {
    runs.sort();
    runs.get(runs.len() / 2).copied().unwrap_or_default()
}

fn journal_batch() {
    let dir = scratch("journal");
    let session = SessionId("1700000000-1".into());
    let Ok(mut journal) = Journal::open(&dir, &session) else {
        return;
    };
    let mut seq = 0;
    let runs = (0..50)
        .map(|_| {
            let batch: Vec<EventRecord> = (0..32)
                .map(|_| {
                    seq += 1;
                    EventRecord {
                        session: session.clone(),
                        seq: Seq(seq),
                        at_ms: 0,
                        kind: "message.delta".into(),
                        data: serde_json::json!({"agentId": "main", "messageId": "m", "text": "x".repeat(64)}),
                        source: None,
                    }
                })
                .collect();
            let started = Instant::now();
            journal.append(&batch).ok();
            started.elapsed()
        })
        .collect();
    println!(
        "journal: 32 records appended and fsynced: {:?}",
        median(runs)
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn idle_frame() {
    let home = scratch("idle");
    let Ok((_lock, token, listener)) = server::bind(&Home::at(&home)) else {
        return;
    };
    let mut app = server::server_app();
    server::serve(
        &mut app,
        Home::at(&home),
        token,
        listener,
        Options::default(),
    );
    for n in 0..200 {
        app.world_mut().spawn((
            Session {
                id: SessionId(format!("1700000000-{n}")),
                cwd: home.clone(),
                host: None,
                build: BuildId("dev".into()),
                name: None,
            },
            Track::default(),
        ));
    }
    app.finish();
    app.cleanup();
    app.update();
    let runs = (0..200)
        .map(|_| {
            let started = Instant::now();
            app.update();
            started.elapsed()
        })
        .collect();
    println!("rigd: an idle frame with 200 sessions: {:?}", median(runs));
    std::fs::remove_dir_all(&home).ok();
}

fn start_round_trip() {
    let home = scratch("start");
    let at = home.clone();
    let agent = std::env::var_os("RIG_BENCH_AGENT").map(PathBuf::from);
    let worker = agent.clone().map(|agent| (agent, Vec::new()));
    let (ready, started) = crossbeam_channel::bounded(1);
    std::thread::spawn(move || {
        let home = Home::at(&at);
        let Ok((_lock, token, listener)) = server::bind(&home) else {
            return;
        };
        let mut app = server::server_app();
        let options = Options {
            worker,
            ..Options::default()
        };
        server::serve(&mut app, home, token, listener, options);
        ready.send(()).ok();
        app.run();
    });
    if started.recv_timeout(Duration::from_secs(10)).is_err() {
        return;
    }
    let options = ClientOptions {
        autostart: Autostart::Off,
        ..ClientOptions::default()
    };
    let target = Target::Local {
        rig_home: home.clone(),
    };
    let Ok(client) = Client::connect_with(&target, "bench", options) else {
        return;
    };
    let start = |n: usize, prompt: Option<String>| Start {
        cwd: home.clone(),
        prompt,
        name: None,
        host: None,
        build: None,
        grant: None,
        policy: None,
        budget: None,
        context: None,
        unload: None,
        view: false,
        key: Key(format!("bench-{n}")),
    };
    let runs = (0..20)
        .map(|n| {
            let began = Instant::now();
            client.call::<Start>(&start(n, None)).ok();
            began.elapsed()
        })
        .collect();
    println!(
        "rigd: rig/session/start without a prompt: {:?}",
        median(runs)
    );
    if agent.is_some() {
        let runs = (100..105)
            .filter_map(|n| {
                let began = Instant::now();
                let started = client.call::<Start>(&start(n, Some("hi".into()))).ok()?;
                loop {
                    let read = ReadEvents {
                        session_id: started.session_id.clone(),
                        after_seq: Seq(0),
                        limit: None,
                    };
                    let events = client.call::<ReadEvents>(&read).ok()?.events;
                    if events.iter().any(|r| r.kind == "worker.ready") {
                        return Some(began.elapsed());
                    }
                    if began.elapsed() > Duration::from_secs(30) {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
            .collect();
        println!(
            "rigd: a start with a prompt to worker.ready: {:?}",
            median(runs)
        );
    }
    std::fs::remove_dir_all(&home).ok();
}

fn main() {
    journal_batch();
    idle_frame();
    start_round_trip();
}
