use std::sync::mpsc::Receiver;

use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, first_agent, log_ended, reply, run_until,
    scratch_home, scripted_models,
};
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

/// Encrypts every session of `home`, with a cheap KDF.
fn enable(home: &Home) {
    fs::create_dir_all(home.root()).ok();
    let config = "enabled = true\nkdf_memory_kib = 64\nkdf_iterations = 1\n";
    fs::write(home.root().join("crypt.toml"), config).ok();
}

fn home(name: &str) -> Home {
    let home = scratch_home(name);
    enable(&home);
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
    app.insert_resource(secrets)
        .insert_resource(scripted_models([("test/model", model)]))
        .register_type::<Notes>()
        .add_plugins((ExitPlugin, AgentPlugin, SessionLogPlugin, EffectLogPlugin))
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

/// Why the run's stores do not open `session`, if they do not.
fn refusal(app: &App, home: &Home, session: &SessionId) -> String {
    let stores = app.world().resource::<StoreOpener>();
    match stores.open(&home.session(session)) {
        Err(why) => why.to_string(),
        Ok(_) => String::new(),
    }
}

/// The conversation of the first agent restored, as text.
fn restored(app: &mut App) -> String {
    let world = app.world_mut();
    let mut agents = world.query::<&Conversation>();
    let text = agents
        .iter(world)
        .next()
        .map(|conversation| format!("{:?}", conversation.messages()));
    text.unwrap_or_default()
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
    drop(app);

    let logs = agent_logs(dir.path());
    assert!(!logs.is_empty(), "the session was saved");
    assert_eq!(
        plain_secrets(home.sessions().as_path()),
        Vec::<PathBuf>::new()
    );

    // Resumed with the key: the conversation and the plugin's state.
    let (mut app, _) = start(&home, &session, &model, key);
    assert!(restored(&mut app).contains(MARKER), "{}", notices(&app));
    let world = app.world_mut();
    let mut agents = world.query::<&Notes>();
    let notes = agents.iter(world).next().map(|notes| notes.0.clone());
    assert_eq!(notes, Some(format!("noted {MARKER}")));
    drop(app);

    // A wrong passphrase: refused after the KDF, nothing written.
    let before: Vec<_> = logs.iter().map(fs::read).collect();
    let (app, _) = start(&home, &session, &model, Some("wrong horse"));
    let why = refusal(&app, &home, &session);
    assert!(why.contains("wrong passphrase"), "{why}");
    assert!(refused(&app));
    drop(app);
    assert_eq!(plain_secrets(dir.path()), Vec::<PathBuf>::new());
    let after: Vec<_> = logs.iter().map(fs::read).collect();
    assert_eq!(format!("{before:?}"), format!("{after:?}"));

    // One byte flipped: the log fails to load as tampered, so the run is
    // refused, and the log stays.
    let log = logs.first().cloned().unwrap_or_default();
    let mut bytes = fs::read(&log).unwrap_or_default();
    let middle = bytes.len() / 2;
    if let Some(byte) = bytes.get_mut(middle) {
        *byte ^= 1;
    }
    fs::write(&log, &bytes).ok();
    let (app, _) = start(&home, &session, &model, key);
    assert!(refused(&app));
    let stores = app.world().resource::<StoreOpener>();
    let read = stores.open(&dir).and_then(|store| Session::read(&*store));
    let why = read.err().map(|failure| failure.to_string());
    assert!(
        why.as_deref().is_some_and(|why| why.contains(TAMPERED)),
        "{why:?}"
    );
    drop(app);
    assert_eq!(fs::read(&log).unwrap_or_default(), bytes);
}

#[test]
fn a_run_without_a_key_refuses_and_keeps_nothing() {
    let home = home("fh18-no-key");
    let session = SessionId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("noted")]);
    let (app, _) = start(&home, &session, &model, None);
    let why = refusal(&app, &home, &session);
    assert!(why.contains("no key"), "{why}");
    assert!(refused(&app));
    drop(app);
    let dir = home.session(&session);
    assert_eq!(agent_logs(dir.path()), Vec::<PathBuf>::new());
    assert_eq!(plain_secrets(dir.path()), Vec::<PathBuf>::new());
}

#[test]
fn a_plain_session_is_refused_until_migrate_seals_it() {
    let home = scratch_home("fh18-migrate");
    let session = SessionId::generate();
    let dir = home.session(&session);
    let model = MockCompletionModel::from_stream_turns([reply("noted")]);
    // Saved before encryption was on: plain.
    let (mut app, wakes) = start(&home, &session, &model, None);
    ask(&mut app, &wakes, &format!("Remember {MARKER}."));
    drop(app);
    assert!(!plain_secrets(dir.path()).is_empty());

    // Now encrypted: the plain session is refused, not read as tampered
    // nor written sealed over.
    enable(&home);
    let key = Some("correct horse");
    let before: Vec<_> = files(dir.path()).iter().map(fs::read).collect();
    let (app, _) = start(&home, &session, &model, key);
    let why = refusal(&app, &home, &session);
    assert!(why.contains("rig crypt migrate"), "{why}");
    assert!(refused(&app));
    drop(app);
    // Its plain files, the text log among them, stay in its directory.
    let after: Vec<_> = files(dir.path()).iter().map(fs::read).collect();
    assert_eq!(format!("{before:?}"), format!("{after:?}"));

    // `rig crypt migrate` seals it in place; it resumes with the key.
    let config = CryptConfig::read(&home).unwrap_or_default();
    let crypt = unlock(&home, &config, key);
    let done = crypt.and_then(|crypt| migrate(&home, Path::new(""), &crypt));
    assert_eq!(
        done.as_deref(),
        Ok("Encrypted 1 sessions; 0 running elsewhere were left plain.")
    );
    assert_eq!(plain_secrets(dir.path()), Vec::<PathBuf>::new());
    let (mut app, _) = start(&home, &session, &model, key);
    assert!(restored(&mut app).contains(MARKER), "{}", notices(&app));
}

/// A migration runs through the sealed store's own reads and writes: the
/// migrated session reads with the key, and the copy kept of it as it was
/// is sealed like the rest, no plain byte of it in the store beneath.
#[test]
fn a_migrated_session_keeps_its_copy_sealed() -> Result<(), Box<dyn std::error::Error>> {
    use rig_harness::rig_memory::migration::{KEPT, testing};
    use rig_harness::rig_memory::{Files, MemoryStore, Session, Store};
    let crypt = Crypt::derive(
        b"synthetic",
        b"0123456789abcdef",
        Kdf {
            memory_kib: 64,
            iterations: 1,
        },
    )?;
    let raw = Arc::new(MemoryStore::default());
    let sealed = CryptStore::new(raw.clone(), crypt);
    testing::write_fixture(&sealed)?;
    let original = sealed.read_log("agent")?;
    testing::CHAIN.migrate(&sealed, &|_, _| Ok(()))?;
    Session::read(&sealed)?;
    let kept = format!("{KEPT}{}/logs/agent.jsonl", testing::SYNTHETIC);
    let plain = b"a synthetic session";
    let mut stored = Vec::new();
    for name in raw.files()? {
        stored.push(raw.read_file(&name)?);
    }
    for name in raw.logs()? {
        stored.push(raw.read_log(&name)?);
    }
    let leaked = stored
        .iter()
        .any(|bytes| bytes.windows(plain.len()).any(|at| at == plain));
    if sealed.read_file(&kept)? != original || raw.read_file(&kept)? == original || leaked {
        return Err(io::Error::other("the kept copy is not the sealed original").into());
    }
    Ok(())
}
