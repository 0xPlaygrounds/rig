use std::io;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

use super::*;

const TOKEN: &str = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// A directory of this test's own, gone when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The effect log as JSON lines: every record written.
#[derive(Resource, Default)]
struct EffectLog(String);

fn keep_records(mut recorded: MessageReader<Recorded>, mut log: ResMut<EffectLog>) {
    for Recorded(record) in recorded.read() {
        log.0
            .push_str(&serde_json::to_string(record).unwrap_or_default());
        log.0.push('\n');
    }
}

/// An agent on `store`, restored if the store has one, answered by
/// `model`, with the read and write tools and the plugin.
fn start(store: &MemoryStore, model: &MockCompletionModel) -> io::Result<(App, Receiver<()>)> {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        ReadTool,
        WriteTool,
        SecretRedactionPlugin,
    ))
    .init_resource::<Ended>()
    .init_resource::<EffectLog>()
    .add_observer(log_ended)
    .add_systems(PostUpdate, keep_records);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let (Some(agent), Some(connection)) = (agent, connect(model)) else {
        return Err(io::Error::other("no agent or model"));
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    Ok((app, wakes))
}

/// The requests `model` was sent, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect()
}

/// What the two runs left.
struct Seen {
    /// The requests of the first run.
    sent: Vec<String>,
    /// What `out.txt` holds.
    written: String,
    /// The session log.
    logged: String,
    /// The first run's effect log.
    effects: String,
    /// The requests after the restart.
    sent_again: Vec<String>,
    /// Whether the call after the restart wrote its file.
    wrote_again: bool,
}

/// The model reads `cfg.txt`, which holds a token, writes its placeholder
/// to `out.txt`, and answers. Then, after a restart, writes the
/// placeholder to `again.txt`.
fn run_the_script() -> io::Result<Seen> {
    let dir = Scratch(
        std::env::temp_dir().join(format!("corpus-secret-redaction-{}", std::process::id())),
    );
    std::fs::create_dir_all(&dir.0)?;
    let (cfg, out, again) = (
        dir.0.join("cfg.txt"),
        dir.0.join("out.txt"),
        dir.0.join("again.txt"),
    );
    std::fs::write(&cfg, format!("token={TOKEN}\n"))?;
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", serde_json::json!({ "path": cfg })),
        calls(
            "2",
            "write",
            serde_json::json!({ "path": out, "content": "‹SECRET_1›" }),
        ),
        reply("Copied the token."),
    ]);
    let (app, _wakes) = start(&store, &model)?;
    let effects = app.world().resource::<EffectLog>().0.clone();
    drop(app);
    let mut logged = String::new();
    for agent in store.agents()? {
        logged.push_str(&String::from_utf8_lossy(&store.read(&agent)?));
    }

    let restarted = MockCompletionModel::from_stream_turns([
        calls(
            "3",
            "write",
            serde_json::json!({ "path": again, "content": "‹SECRET_1›" }),
        ),
        reply("It is gone."),
    ]);
    let _restarted_app = start(&store, &restarted)?;
    Ok(Seen {
        sent: sent(&model),
        written: std::fs::read_to_string(&out)?,
        logged,
        effects,
        sent_again: sent(&restarted),
        wrote_again: again.exists(),
    })
}

/// The token never leaves the machine nor reaches a log, the file gets
/// it, and after a restart its placeholder is unknown.
#[test]
fn a_secret_read_goes_out_as_a_placeholder_and_is_written_back_for_real() {
    let seen = run_the_script();
    assert!(seen.is_ok(), "the script ran");
    let Ok(seen) = seen else {
        return;
    };
    let sent = &seen.sent;
    assert_eq!(sent.len(), 3, "read, write, answer: {sent:?}");
    let after_read = sent.get(1).map(String::as_str).unwrap_or_default();
    assert!(after_read.contains("‹SECRET_1›"), "{after_read}");
    assert!(
        sent.iter().all(|request| !request.contains("ghp_")),
        "{sent:?}"
    );
    assert_eq!(seen.written, TOKEN);
    for log in [&seen.logged, &seen.effects] {
        assert!(log.contains("‹SECRET_1›"), "{log}");
        assert!(!log.contains("ghp_"), "{log}");
    }
    let after_write = seen
        .sent_again
        .get(1)
        .map(String::as_str)
        .unwrap_or_default();
    assert!(after_write.contains("not a known secret"), "{after_write}");
    assert!(!seen.wrote_again);
}
