use std::path::PathBuf;

use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

use super::*;

const TOKEN: &str = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Whether a turn ended.
#[derive(Resource, Default)]
struct Ended(bool);

/// A directory of this test's own, gone when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// What a run of the spec's script left: the requests sent, as JSON, what
/// `out.txt` holds and the session log.
struct Ran {
    sent: Vec<String>,
    written: String,
    logged: String,
}

/// The model reads `cfg.txt`, which holds a token, writes its placeholder
/// to `out.txt`, and answers.
fn run_the_script() -> io::Result<Ran> {
    let dir = Scratch(
        std::env::temp_dir().join(format!("corpus-secret-redaction-{}", std::process::id())),
    );
    std::fs::create_dir_all(&dir.0)?;
    let (cfg, out) = (dir.0.join("cfg.txt"), dir.0.join("out.txt"));
    std::fs::write(&cfg, format!("token={TOKEN}\n"))?;
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", serde_json::json!({ "path": cfg })),
        calls(
            "2",
            "write",
            serde_json::json!({ "path": out, "content": "‹SECRET_1›" }),
        ),
        reply("Copied the token."),
    ]);

    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        ReadTool,
        WriteTool,
        SecretRedactionPlugin::default(),
    ))
    .init_resource::<Ended>()
    .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 = true);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let (Some(agent), Some(connection)) = (agent, connect(&model)) else {
        return Err(io::Error::other("no agent or model"));
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "read cfg.txt", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0);

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    let written = std::fs::read_to_string(&out)?;
    let mut logged = String::new();
    for agent in store.agents()? {
        logged.push_str(&String::from_utf8_lossy(&store.read(&agent)?));
    }
    Ok(Ran {
        sent,
        written,
        logged,
    })
}

#[test]
fn a_secret_read_goes_out_as_a_placeholder_and_is_written_back_for_real() {
    let ran = run_the_script();
    assert!(ran.is_ok(), "the script ran");
    let Ok(Ran {
        sent,
        written,
        logged,
    }) = ran
    else {
        return;
    };
    assert_eq!(sent.len(), 3, "read, write, answer: {sent:?}");
    let after_read = sent.get(1).map(String::as_str).unwrap_or_default();
    assert!(after_read.contains("‹SECRET_1›"), "{after_read}");
    assert!(
        sent.iter().all(|request| !request.contains("ghp_")),
        "{sent:?}"
    );
    assert_eq!(written, TOKEN);
    assert!(logged.contains("‹SECRET_1›"), "{logged}");
    assert!(!logged.contains("ghp_"), "{logged}");
}
