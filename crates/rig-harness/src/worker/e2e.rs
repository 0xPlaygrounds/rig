//! `rigd` with real workers: this test binary run as [`scripted_worker`],
//! the worker app on a scripted model, against a server on its own thread.
#![allow(clippy::panic_in_result_fn)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rig_client::{Autostart, Client, Options as ClientOptions, Target};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Agent;
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{app_in, connect};
use rig_protocol::methods::{Budget, ReadEvents, Send, Start, Wait, WaitStatus};
use rig_protocol::{Key, Seq, SessionId};

use super::{WorkerConfig, WorkerPlugin};
use crate::harness_protocol::Home;
use crate::host::exit::ExitPlugin;
use crate::server::{Options, bind, serve, server_app};

type Result<T = (), E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

/// Tokens each scripted reply reports.
const PER_CALL: u64 = 50;

#[test]
#[ignore = "run by the end-to-end tests as their worker"]
fn scripted_worker() -> Result {
    let (Some(config), Ok(session)) = (
        WorkerConfig::from_env(),
        std::env::var(crate::harness_protocol::env::SESSION),
    ) else {
        return Ok(());
    };
    let home = Home::from_env();
    let id: crate::harness_protocol::SessionId = session.parse()?;
    let turns = (0..20).map(|n| {
        vec![
            MockStreamEvent::text(format!("answer {n}")),
            MockStreamEvent::final_response_with_total_tokens(PER_CALL),
        ]
    });
    let model = MockCompletionModel::from_stream_turns(turns);
    let (mut app, wakes) = app_in(&home, &id);
    app.insert_resource(config).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        crate::clock::ClockPlugin,
        WorkerPlugin,
    ));
    app.add_systems(
        bevy_app::PostStartup,
        move |agents: bevy_ecs::system::Query<bevy_ecs::entity::Entity, bevy_ecs::query::With<Agent>>,
              mut commands: bevy_ecs::system::Commands| {
            for agent in &agents {
                if let Some(connection) = connect(&model) {
                    commands.entity(agent).insert(connection);
                }
            }
        },
    );
    app.finish();
    app.cleanup();
    let by = Instant::now() + Duration::from_secs(60);
    while app.should_exit().is_none() && Instant::now() < by {
        app.update();
        wakes.recv_timeout(Duration::from_millis(50)).ok();
    }
    Ok(())
}

fn key() -> Key {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    Key(format!("e2e-{}", NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)))
}

#[test]
fn a_session_stops_making_model_calls_once_its_token_budget_is_spent() -> Result {
    let home = std::env::temp_dir().join(format!("rigd-e2e-budget-{}", std::process::id()));
    std::fs::remove_dir_all(&home).ok();
    std::fs::create_dir_all(&home)?;
    let options = Options {
        idle_exit: Duration::from_millis(300),
        idle_unload: Duration::from_millis(300),
        watchdog: Duration::from_secs(10),
        crash_limit: 3,
        worker: Some((
            std::env::current_exe()?,
            ["worker::e2e::scripted_worker", "--exact", "--ignored", "--nocapture"]
                .map(Into::into)
                .to_vec(),
        )),
    };
    let at: PathBuf = home.clone();
    let (ready, started) = crossbeam_channel::bounded(1);
    let server = std::thread::spawn(move || {
        let home = Home::at(&at);
        let Ok((_lock, token, listener)) = bind(&home) else {
            return bevy_app::AppExit::error();
        };
        let mut app = server_app();
        serve(&mut app, home, token, listener, options);
        ready.send(()).ok();
        app.run()
    });
    started.recv_timeout(Duration::from_secs(10))?;
    let client = Client::connect_with(
        &Target::Local {
            rig_home: home.clone(),
        },
        "e2e",
        ClientOptions {
            autostart: Autostart::Off,
            give_up: Duration::from_secs(5),
            ..ClientOptions::default()
        },
    )?;
    // Two calls' worth: the first two turns run, the third finds it spent.
    let session = client
        .call::<Start>(&Start {
            cwd: home.clone(),
            prompt: Some("one".into()),
            name: None,
            host: None,
            build: None,
            grant: None,
            policy: None,
            budget: Some(Budget {
                tokens: Some(2 * PER_CALL - 1),
                cost_usd: None,
            }),
            context: None,
            unload: None,
            view: false,
            key: key(),
        })?
        .session_id;
    let wait = |session: &SessionId| {
        client.call::<Wait>(&Wait {
            session_id: session.clone(),
            after_command: None,
            timeout_ms: Some(20_000),
        })
    };
    let send = |text: &str| {
        client.call::<Send>(&Send {
            session_id: session.clone(),
            text: Some(text.into()),
            content: None,
            mode: None,
            unload: None,
            key: key(),
        })
    };
    let mut statuses = vec![wait(&session)?.status];
    // The worker's environment, which its commands can read, holds no token.
    #[cfg(target_os = "linux")]
    {
        let pid = client
            .call::<ReadEvents>(&ReadEvents {
                session_id: session.clone(),
                after_seq: Seq(0),
                limit: None,
            })?
            .events
            .iter()
            .find(|e| e.kind == "worker.ready")
            .and_then(|e| e.data.get("pid")?.as_u64());
        let environ = pid
            .map(|pid| std::fs::read(format!("/proc/{pid}/environ")).unwrap_or_default())
            .unwrap_or_default();
        let tokens: Vec<String> = environ
            .split(|b| *b == 0)
            .map(|var| String::from_utf8_lossy(var).into_owned())
            .filter(|var| var.starts_with("RIG_WORKER_TOKEN="))
            .collect();
        assert_eq!(tokens, ["RIG_WORKER_TOKEN=-"], "pid {pid:?}");
    }
    for text in ["two", "three"] {
        match send(text) {
            Ok(_) => statuses.push(wait(&session)?.status),
            Err(_) => statuses.push(WaitStatus::Failed),
        }
    }
    let events = client
        .call::<ReadEvents>(&ReadEvents {
            session_id: session.clone(),
            after_seq: Seq(0),
            limit: None,
        })?
        .events;
    let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
    let used: Vec<u64> = events
        .iter()
        .filter(|e| e.kind == "usage")
        .filter_map(|e| e.data.get("tokens")?.as_u64())
        .collect();
    drop(client);
    let by = Instant::now() + Duration::from_secs(30);
    while !server.is_finished() && Instant::now() < by {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::fs::remove_dir_all(&home).ok();
    assert_eq!(used, [PER_CALL, PER_CALL], "{kinds:?}");
    assert!(kinds.contains(&"budget.exhausted"), "{kinds:?}");
    assert_eq!(
        statuses.get(..2),
        Some(&[WaitStatus::Answered, WaitStatus::Answered][..])
    );
    assert_ne!(statuses.get(2), Some(&WaitStatus::Answered), "{statuses:?}");
    Ok(())
}
