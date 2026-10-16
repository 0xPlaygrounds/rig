use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Conversation, Entry, Halt};
use rig_core::message::{ImageMediaType, UserContent};
use serde_json::{Map, Value, json};

use super::{Unread, Upgrade, Upgrades, Versioned};
use crate::format::LogLines;
use crate::{
    Files, MemoryStore, Repair, SESSION, Session, SessionMeta, Store, SummaryState, VERSION,
    Writer, apply_summary,
};

/// The format of the oldest fixture, whose header a test makes newer.
const OLDEST_FIXTURE: u32 = 2;

/// Where the frozen sessions every older format wrote are kept.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// A session of every record type, written by this build's writer: two
/// agents (one spawned by the other, its log starting with a reset), a
/// retired one, the session's own value, an image, a summary with its
/// aside, nested branches, a halt, a saved value of a deleted type and one
/// of a type no build knows, a listing, and a torn tail.
fn every_record() -> MemoryStore {
    let store = Arc::new(MemoryStore::default());
    let Ok((writer, _)) = Writer::open(store.clone()) else {
        return MemoryStore::default();
    };
    writer.start();
    let state = |log: &str, key: &str, value: Value| {
        writer.save_value(log, key, Some(value.into()), true);
    };
    let mut main = Conversation::new();
    main.push(Message::User {
        content: vec![
            UserContent::text("look"),
            UserContent::image_base64("cG5n", Some(ImageMediaType::PNG), None),
        ],
    });
    main.push(Message::assistant("a cat"));
    main.push(Entry::new(Message::user("report")).by(Author::plugin("ci")));
    main.push(Message::assistant("noted"));
    let summary = SummaryState {
        summary: "The user showed a cat.".to_owned(),
        tracked: Vec::new(),
    };
    let upto = main.id(2);
    apply_summary(&mut main, upto, &summary);
    main.push(Message::user("go on"));
    main.halt(Halt::Stopped);
    main.push(Message::user("go on, then"));
    main.push(Message::assistant("going"));
    main.push(Message::assistant("another way"));
    main.push(Message::user("deeper"));
    let note = Aside::new("note", Author::plugin("btw"), json!({ "text": "hi" }));
    main.add_aside(note);
    writer.save("main", &mut main);
    state("main", "rig_ecs::agent::ToolAccess", json!("All"));
    state("main", "rig::model", json!("ollama/deepseek-v4-flash"));
    state(
        "main",
        "rig_compaction::Summarized",
        json!({ "summary": "The user showed a cat.", "tracked": [] }),
    );
    state("main", "a_plugin::Gone", json!(1));
    state("main", "a_plugin::Removed", json!(2));
    writer.save_value("main", "a_plugin::Removed", None, true);
    state(SESSION, "a_plugin::Counter", json!(3));

    writer.open_log("sub", Some("main"));
    let mut first = Conversation::from_messages([Message::user("a first try")]);
    writer.save("sub", &mut first);
    let mut sub = Conversation::from_messages([Message::user("Fix the parser.")]);
    writer.save("sub", &mut sub);
    sub.push(Message::assistant("Fixed."));
    writer.save("sub", &mut sub);
    state("sub", "rig::title", json!("Fix the parser"));

    writer.open_log("gone", Some("main"));
    let mut gone = Conversation::from_messages([Message::user("never mind")]);
    writer.save("gone", &mut gone);
    writer.retire("gone");
    writer.flush().expect("flushed");

    let meta = SessionMeta {
        title: "look".to_owned(),
        updated: 1_791_779_409_181,
        ..SessionMeta::default()
    };
    let torn = b"{\"seq\":99,\"n\":";
    let written = meta.write(&*store);
    let written = written.and_then(|()| store.append_log("main", torn));
    assert!(written.is_ok());
    Arc::unwrap_or_clone(store)
}

/// What a reader must find in `session`, as JSON: each agent's
/// conversation (entries, saved asides, halt), saved values, parent,
/// retirement and active title, and the session's own values.
fn summary(session: &Session) -> Value {
    let mut agents = Map::new();
    for agent in &session.agents {
        let summary = json!({
            "conversation": agent.conversation,
            "state": agent.saved,
            "parent": agent.parent,
            "retired": agent.retired,
            "title": title(&agent.conversation),
        });
        agents.insert(agent.id.clone(), summary);
    }
    json!({ "agents": agents, "state": session.saved })
}

/// The title the fixtures' readers gave the active line: the value of its
/// `rig.branch.active` aside, else `main`.
fn title(conversation: &rig_core::conversation::Conversation) -> String {
    let active = conversation
        .asides()
        .find(|(_, aside)| aside.kind == "rig.branch.active");
    let title = active.and_then(|(_, aside)| aside.value.as_str().map(str::to_owned));
    title.unwrap_or_else(|| "main".to_owned())
}

/// Writes `store` into `dir` as a `DirStore` lays it out: `logs/`,
/// `blobs/` and `files/`.
fn dump(store: &MemoryStore, dir: &Path) -> std::io::Result<()> {
    let write = |path: PathBuf, bytes: &[u8]| {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)
    };
    for log in store.logs()? {
        write(
            dir.join("logs").join(format!("{log}.jsonl")),
            &store.read_log(&log)?,
        )?;
    }
    for blob in crate::blobs::blob_names(store)? {
        let bytes = store.read_file(&format!("blobs/{blob}"))?;
        write(dir.join("blobs").join(&blob), &bytes)?;
    }
    for file in store.files()? {
        write(dir.join("files").join(&file), &store.read_file(&file)?)?;
    }
    Ok(())
}

/// The fixture in `dir` as a store in memory.
fn load(dir: &Path) -> std::io::Result<MemoryStore> {
    let store = MemoryStore::default();
    let mut found = vec![dir.to_path_buf()];
    while let Some(path) = found.pop() {
        for entry in fs::read_dir(&path)? {
            let path = entry?.path();
            if path.is_dir() {
                found.push(path);
                continue;
            }
            let Ok(name) = path.strip_prefix(dir) else {
                continue;
            };
            let name = name.to_string_lossy().replace('\\', "/");
            let bytes = fs::read(&path)?;
            if let Some(log) = name.strip_prefix("logs/") {
                let log = log.strip_suffix(".jsonl").unwrap_or(log);
                store.append_log(log, &bytes)?;
            } else if let Some(blob) = name.strip_prefix("blobs/") {
                store.write_file(&format!("blobs/{blob}"), &bytes)?;
            } else if let Some(file) = name.strip_prefix("files/") {
                store.write_file(file, &bytes)?;
            }
        }
    }
    Ok(store)
}

/// Freezes a session written by this build's writer as
/// `fixtures/rig-session-<VERSION>/`, with what a reader must find in it
/// (`expected.json`), unless it is there: run it once before a format's
/// version moves on, and commit what it wrote. Fixtures are never edited.
#[test]
#[ignore = "writes the fixture of this format; run once before bumping it"]
fn write_the_fixture_of_this_format() {
    let dir = fixtures().join(format!("rig-session-{VERSION}"));
    if dir.exists() {
        return;
    }
    let store = every_record();
    let session = Session::read(&store).expect("it reads");
    let expected = serde_json::to_vec_pretty(&summary(&session)).expect("json");
    dump(&store, &dir).expect("written");
    fs::write(dir.join("expected.json"), expected).expect("written");
}

fn label(old: Value) -> Result<Value, String> {
    Ok(json!([old, ""]))
}

fn refuse(_: Value) -> Result<Value, String> {
    Err("no".to_owned())
}

/// A value is read through every step from its own version; a newer one, a
/// failing step and version 0 are set aside with the reason; a plugin
/// aside's envelope reads back at its version.
#[test]
fn saved_values_and_asides_read_through_their_steps() {
    const STEPS: &[Upgrade] = &[label, label];
    let mut upgrades = Upgrades::default();
    upgrades.add("k", STEPS);
    upgrades.add("broken", &[refuse]);
    assert_eq!(upgrades.version("k"), 3);
    assert_eq!(upgrades.version("unknown"), 1);

    let saved = |v, value| Versioned { v, value };
    assert_eq!(
        upgrades.upgrade("k", saved(1, json!(7))),
        Ok(json!([[7, ""], ""]))
    );
    assert_eq!(upgrades.upgrade("k", saved(3, json!(7))), Ok(json!(7)));
    assert_eq!(
        upgrades.upgrade("k", saved(4, json!(7))),
        Err(Unread::Newer { v: 4, reads: 3 })
    );
    assert_eq!(
        upgrades.upgrade("broken", saved(1, json!(7))),
        Err(Unread::Step {
            from: 1,
            why: "no".to_owned()
        })
    );
    assert!(upgrades.upgrade("k", saved(0, json!(7))).is_err());

    let at_two = saved(2, json!({"v": 9, "value": 1})).to_aside_value();
    let aside = Aside::new("k", Author::rig(), at_two);
    assert_eq!(
        upgrades.read_aside(&aside),
        Ok(json!([{"v": 9, "value": 1}, ""]))
    );
    assert_eq!(saved(1, json!(5)).to_aside_value(), json!(5));
}

/// Each frozen fixture, by name, as a store in memory, with what a reader
/// must find in it.
fn every_fixture() -> Vec<(String, MemoryStore, Value)> {
    let mut found = Vec::new();
    let dirs = fs::read_dir(fixtures()).expect("the fixtures");
    for dir in dirs {
        let dir = dir.expect("a fixture").path();
        let name = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        let expected = fs::read(dir.join("expected.json")).expect("its expected.json");
        let expected = serde_json::from_slice(&expected).expect("json");
        found.push((
            name.unwrap_or_default(),
            load(&dir).expect("it loads"),
            expected,
        ));
    }
    assert!(!found.is_empty(), "the fixtures are checked in");
    found
}

/// Every byte `store` holds: its logs, blobs and files, by name.
fn bytes(store: &MemoryStore) -> Vec<(String, Vec<u8>)> {
    let mut all = Vec::new();
    for log in store.logs().expect("logs") {
        all.push((format!("logs/{log}"), store.read_log(&log).expect("a log")));
    }
    for blob in crate::blobs::blob_names(store).expect("blobs") {
        let name = format!("blobs/{blob}");
        let bytes = store.read_file(&name).expect("a blob");
        all.push((name, bytes));
    }
    for file in store.files().expect("files") {
        all.push((
            format!("files/{file}"),
            store.read_file(&file).expect("a file"),
        ));
    }
    all.sort();
    all
}

/// Every session an older rig saved reads with this one as it read then,
/// writing nothing; the first write sets its torn tail aside and leaves
/// every log in its own format, the same session.
#[test]
fn every_fixture_reads_and_is_written_on_in_its_own_format() {
    for (name, store, expected) in every_fixture() {
        let before = bytes(&store);
        let session = Session::read(&store).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(summary(&session), expected, "{name}");
        assert_eq!(bytes(&store), before, "{name}: a read writes nothing");

        let torn: Vec<Vec<u8>> = store
            .logs()
            .expect("logs")
            .into_iter()
            .map(|log| {
                LogLines::parse(store.read_log(&log).expect("a log"))
                    .tail()
                    .to_vec()
            })
            .filter(|tail| !tail.is_empty())
            .collect();
        let (_, opened) = Writer::open(Arc::new(store.clone())).expect("it opens");
        let set_aside = opened.repairs.iter().filter(|repair| {
            matches!(
                repair,
                Repair::TornTail {
                    set_aside: Some(_),
                    ..
                }
            )
        });
        assert_eq!(
            set_aside.count(),
            torn.len(),
            "{name}: {:?}",
            opened.repairs
        );
        let header = format!(r#""format":"{}""#, name.replace("session-", "session/"));
        for log in store.logs().expect("logs") {
            let bytes = store.read_log(&log).expect("a log");
            let kept = bytes.windows(header.len()).any(|w| w == header.as_bytes());
            assert!(kept, "{name}: {log} stays in its format");
        }
        let written = Session::read(&store).expect("it reads");
        assert_eq!(summary(&written), expected, "{name}");
        let recovered: Vec<Vec<u8>> = store
            .files()
            .expect("files")
            .iter()
            .filter(|file| file.starts_with("recovered/"))
            .map(|file| store.read_file(file).expect("a file"))
            .collect();
        assert_eq!(recovered, torn, "{name}: the torn tail is kept");
        // Written once: a second open changes nothing.
        let once = bytes(&store);
        let _ = Writer::open(Arc::new(store.clone())).expect("it opens");
        assert_eq!(bytes(&store), once, "{name}");
    }
}

#[test]
fn a_session_of_a_newer_format_is_refused_by_every_writer_unchanged() {
    for (name, store, _) in every_fixture() {
        let main = store.read_log("main").expect("the main log");
        let main = String::from_utf8(main).expect("text");
        let newer = format!("rig-session/{}", VERSION + 1);
        let main = main.replacen(&format!("rig-session/{}", OLDEST_FIXTURE), &newer, 1);
        store.replace_log("main", main.as_bytes()).expect("written");
        let before = bytes(&store);
        let newer = |refused: Result<(), crate::SessionError>| {
            let refused = refused.expect_err("refused");
            assert!(
                refused.to_string().contains("a newer version of rig"),
                "{name}: {refused}"
            );
        };
        newer(Writer::open(Arc::new(store.clone())).map(drop));
        newer(Session::default().save(&store));
        let mut conversation = Conversation::from_messages([Message::user("more")]);
        newer(crate::save(&store, "sub", &mut conversation));
        newer(crate::rewrite(&store, Author::rig(), &crate::Messages(|_| false)).map(drop));
        assert_eq!(bytes(&store), before, "{name}");
    }
}

fn double(old: Value) -> Result<Value, String> {
    let old = old.as_u64().ok_or("not a count")?;
    Ok(json!(old * 2))
}

/// A library program reads a value of a session an older rig saved, typed,
/// by its key, at its current version, after it was opened for writing; a
/// value no step reads, or of another shape, is an error, not a `None`.
#[test]
fn a_saved_value_of_an_older_session_reads_typed() {
    for (name, store, _) in every_fixture() {
        let _ = Writer::open(Arc::new(store.clone())).expect("opened");
        let session = Session::read(&store).expect("it reads");
        let mut upgrades = Upgrades::default();
        upgrades.add("a_plugin::Counter", &[double]);
        let counter = session.get::<u32>("a_plugin::Counter", &upgrades);
        assert_eq!(counter.map(Result::ok), Some(Some(6)), "{name}");
        let main = session.agent("main").expect("the main agent");
        let model = main.get::<String>("rig::model", &upgrades);
        assert_eq!(
            model.and_then(Result::ok).as_deref(),
            Some("ollama/deepseek-v4-flash")
        );
        assert!(main.get::<String>("nothing", &upgrades).is_none());
        assert!(matches!(
            main.get::<String>("a_plugin::Gone", &upgrades),
            Some(Err(super::ValueError::Parse(_)))
        ));
        upgrades.add("a_plugin::Gone", &[refuse]);
        assert!(matches!(
            main.get::<u32>("a_plugin::Gone", &upgrades),
            Some(Err(super::ValueError::Unread(Unread::Step { from: 1, .. })))
        ));
    }
}

/// A value no step reads moves to its side key the first time its key is
/// written again, and a held key is never written.
#[test]
fn a_set_aside_value_moves_to_its_side_key_and_a_held_one_is_kept() {
    let store = Arc::new(MemoryStore::default());
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    let saved = |v, value| Some(Versioned { v, value });
    writer.save_value("main", "old", saved(1, json!("unread")), true);
    writer.save_value("main", "newer", saved(9, json!("from a newer rig")), true);
    writer.flush().expect("flushed");
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    writer.shelve(
        "main",
        "old",
        Versioned {
            v: 1,
            value: json!("unread"),
        },
    );
    writer.hold("main", "newer");
    writer.save_value("main", "old", saved(2, json!("new")), true);
    writer.save_value("main", "old", saved(2, json!("newer still")), true);
    writer.save_value("main", "newer", saved(2, json!("over it")), true);
    writer.flush().expect("flushed");
    let session = Session::read(&*store).expect("read");
    let main = session.agent("main").expect("the agent");
    assert_eq!(
        main.saved,
        [
            (
                "newer".to_owned(),
                Versioned {
                    v: 9,
                    value: json!("from a newer rig")
                }
            ),
            (
                "old".to_owned(),
                Versioned {
                    v: 2,
                    value: json!("newer still")
                }
            ),
            (
                format!("{}old/v1", super::SHELVED),
                Versioned {
                    v: 1,
                    value: json!("unread")
                }
            ),
        ]
        .into()
    );
}
