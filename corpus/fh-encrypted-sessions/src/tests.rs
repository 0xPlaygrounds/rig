use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, first_agent, log_ended, reply, run_until,
    scratch_home, scripted_models,
};
use rig_telemetry::EffectLogPlugin;
use serde_json::json;
use std::fs;

use rig_harness::host::exit::ExitPlugin;

use super::*;

const MARKER: &str = "SECRET-1234";

/// A plugin's saved state.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Notes(String);

/// `echo`, whose result and the saved notes carry the marker.
fn echo(
    called: On<ToolCalled<serde_json::Value>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success(format!("echo: {MARKER}").into()),
        );
    }
    commands
        .entity(called.agent)
        .insert(Notes(format!("noted {MARKER}")));
}

fn home(name: &str) -> Home {
    let home = scratch_home(name);
    fs::create_dir_all(home.root()).ok();
    let config = "enabled = true\nkdf_memory_kib = 64\nkdf_iterations = 1\n";
    fs::write(home.root().join("crypt.toml"), config).ok();
    home
}

/// A run of `session` given `passphrase`, as the launcher gives it.
fn start(
    home: &Home,
    session: &SessionId,
    model: &MockCompletionModel,
    passphrase: Option<&str>,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, session);
    let secrets: Secrets = passphrase
        .map(|passphrase| (SECRET, passphrase))
        .into_iter()
        .collect();
    // An interactive run with no terminal front: a headless one would
    // take `rig crypt`'s front, the first headless front offered.
    let invoked = Invoked {
        terminal: true,
        ..Invoked::default()
    };
    app.insert_resource(secrets)
        .insert_resource(invoked)
        .insert_resource(scripted_models([("test/model", model)]))
        .register_type::<Notes>()
        .add_plugins((ExitPlugin, AgentPlugin, JournalPlugin, EffectLogPlugin))
        .add_plugins(EncryptedSessionsPlugin)
        .add_open_tool("echo", "Echoes.", ToolOptions::default(), echo)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

/// Runs a turn on the first agent for `text`.
fn ask(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
    app.update();
}

/// Every file under `dir`.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// The files under `dir` holding the marker in plain bytes.
fn plain_secrets(dir: &Path) -> Vec<PathBuf> {
    files(dir)
        .into_iter()
        .filter(|path| {
            let bytes = fs::read(path).unwrap_or_default();
            bytes
                .windows(MARKER.len())
                .any(|at| at == MARKER.as_bytes())
        })
        .collect()
}

fn agent_logs(dir: &Path) -> Vec<PathBuf> {
    files(dir)
        .into_iter()
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "jsonl")
                && path.file_name().is_some_and(|name| name != "effects.jsonl")
        })
        .collect()
}

fn notices(app: &App) -> String {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .map(|notice| format!("{notice:?}\n"))
        .collect()
}

/// Why the run has no key, if it has none.
fn refusal(app: &App) -> String {
    match app
        .world()
        .get_resource::<Unlocked>()
        .map(|unlocked| &unlocked.0)
    {
        Some(Err(why)) => why.clone(),
        _ => String::new(),
    }
}

/// Whether the run was refused: no store, and an exit with code 2, which
/// ends it before anything is typed.
fn refused(app: &App) -> bool {
    let world = app.world();
    !world.contains_resource::<SessionStore>()
        && matches!(
            world.get_resource::<ExitReason>(),
            Some(ExitReason::Failed(2))
        )
}

#[test]
fn a_sealed_session_resumes_with_its_key_only() {
    let home = home("fh18-sealed");
    let key = Some("correct horse");
    let session = SessionId::generate();
    let dir = home.session(&session);
    let model =
        MockCompletionModel::from_stream_turns([calls("c1", "echo", json!({})), reply("noted")]);
    let (mut app, wakes) = start(&home, &session, &model, key);
    ask(&mut app, &wakes, &format!("Remember {MARKER}."));
    // A fork is sealed as it is written, its parent file too.
    app.world_mut().trigger(ForkSession::default());
    app.update();
    let forks: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<SessionForked>>()
        .drain()
        .collect();
    drop(app);

    let logs = agent_logs(dir.path());
    assert!(!logs.is_empty(), "the session was saved");
    let fork = forks.first().map(|fork| home.session(&fork.to));
    assert!(
        fork.as_ref()
            .is_some_and(|fork| !agent_logs(fork.path()).is_empty())
    );
    assert_eq!(
        plain_secrets(home.sessions().as_path()),
        Vec::<PathBuf>::new()
    );

    // Resumed with the key: the conversation and the plugin's state.
    let (mut app, _) = start(&home, &session, &model, key);
    let world = app.world_mut();
    let mut agents = world.query::<(&Conversation, &Notes)>();
    let restored = agents.iter(world).next();
    let text = restored.map(|(conversation, _)| format!("{:?}", conversation.messages()));
    assert!(
        text.unwrap_or_default().contains(MARKER),
        "{}",
        notices(&app)
    );
    let world = app.world_mut();
    let notes = agents.iter(world).next().map(|(_, notes)| notes.0.clone());
    assert_eq!(notes, Some(format!("noted {MARKER}")));
    drop(app);

    // A wrong passphrase: refused after the KDF, nothing written.
    let before: Vec<_> = logs.iter().map(fs::read).collect();
    let (app, _) = start(&home, &session, &model, Some("wrong horse"));
    assert!(
        refusal(&app).contains("wrong passphrase"),
        "{}",
        refusal(&app)
    );
    assert!(refused(&app));
    drop(app);
    assert_eq!(plain_secrets(dir.path()), Vec::<PathBuf>::new());
    let after: Vec<_> = logs.iter().map(fs::read).collect();
    assert_eq!(format!("{before:?}"), format!("{after:?}"));

    // One byte flipped: the log fails to load as tampered, and stays.
    let log = logs.first().cloned().unwrap_or_default();
    let mut bytes = fs::read(&log).unwrap_or_default();
    let middle = bytes.len() / 2;
    if let Some(byte) = bytes.get_mut(middle) {
        *byte ^= 1;
    }
    fs::write(&log, &bytes).ok();
    let (app, _) = start(&home, &session, &model, key);
    assert!(notices(&app).contains(TAMPERED), "{}", notices(&app));
    drop(app);
    assert_eq!(fs::read(&log).unwrap_or_default(), bytes);
}

#[test]
fn a_run_without_a_key_refuses_and_keeps_nothing() {
    let home = home("fh18-no-key");
    let session = SessionId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("noted")]);
    let (app, _) = start(&home, &session, &model, None);
    assert!(refusal(&app).contains("no key"), "{}", refusal(&app));
    assert!(refused(&app));
    drop(app);
    let dir = home.session(&session);
    assert_eq!(agent_logs(dir.path()), Vec::<PathBuf>::new());
    assert_eq!(plain_secrets(dir.path()), Vec::<PathBuf>::new());
}
