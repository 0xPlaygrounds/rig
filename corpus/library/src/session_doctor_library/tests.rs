use std::collections::BTreeMap;
use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};

use rig_core::conversation::Conversation;
use rig_core::message::Message;
use rig_memory::{DirStore, SavedAgent, Session};
use serde_json::{Value, json};

use super::*;

/// `result`'s value, failing the test with its error.
fn ok<T>(result: Result<T, impl Display>) -> Option<T> {
    let result = result.map_err(|error| error.to_string());
    assert_eq!(result.as_ref().err(), None);
    result.ok()
}

/// A fresh directory beside the test binary.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("sp-session-doctor-{name}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    dir
}

/// Saves a session of one agent, `main`, to `dir`: two turns, and with
/// `longer` a third.
fn saved(dir: &Path, longer: bool) -> Option<()> {
    let mut conversation = Conversation::new();
    conversation.push(Message::user("hello"));
    conversation.push(Message::assistant("hi, what next?"));
    conversation.push(Message::user("list the files"));
    conversation.push(Message::assistant("a.txt and b.txt"));
    if longer {
        conversation.push(Message::user("tell me a joke"));
        conversation.push(Message::assistant("a joke"));
    }
    let agent = SavedAgent {
        id: "main".to_owned(),
        conversation,
        ..SavedAgent::default()
    };
    let mut session = Session {
        agents: vec![agent],
        ..Session::default()
    };
    ok(session.save(&ok(DirStore::open(dir))?))
}

/// Copies the directory `from` to `to`.
fn copy_dir(from: &Path, to: &Path) -> Option<()> {
    ok(fs::create_dir_all(to))?;
    for entry in ok(fs::read_dir(from))? {
        let path = ok(entry)?.path();
        let target = to.join(path.file_name()?);
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            ok(fs::copy(&path, &target))?;
        }
    }
    Some(())
}

/// Every file under `dir`, with its bytes.
fn snapshot(dir: &Path, into: &mut BTreeMap<PathBuf, Vec<u8>>) -> Option<()> {
    for entry in ok(fs::read_dir(dir))? {
        let path = ok(entry)?.path();
        if path.is_dir() {
            snapshot(&path, into)?;
        } else {
            into.insert(path.clone(), ok(fs::read(&path))?);
        }
    }
    Some(())
}

/// The lines of the main log of the session in `dir`.
fn main_log(dir: &Path) -> Option<Vec<String>> {
    let text = ok(fs::read_to_string(dir.join("logs/main.jsonl")))?;
    Some(text.lines().map(str::to_owned).collect())
}

/// Writes `lines` as the main log of the session in `dir`.
fn write_main_log(dir: &Path, lines: &[String]) -> Option<()> {
    ok(fs::write(
        dir.join("logs/main.jsonl"),
        lines.join("\n") + "\n",
    ))
}

/// The main log of `dir` with its header declaring `format`.
fn declare(dir: &Path, format: &str) -> Option<()> {
    let mut lines = main_log(dir)?;
    let header = lines.first_mut()?;
    let mut value: Value = ok(serde_json::from_str(header))?;
    *value.get_mut("format")? = json!(format);
    *header = value.to_string();
    write_main_log(dir, &lines)
}

/// The session in `dir` as the library reads it.
fn read(dir: &Path) -> Option<Session> {
    ok(Session::read(&ok(DirStore::open(dir))?))
}

/// The line of the main log of `dir` holding `kind`, from 0.
fn line_of(dir: &Path, kind: &str) -> Option<usize> {
    main_log(dir)?.iter().position(|line| line.contains(kind))
}

#[test]
fn sessions_are_classed_and_the_fixable_ones_copied_repaired() -> Result<(), String> {
    check_and_fix().ok_or_else(|| "the check failed".to_owned())
}

fn check_and_fix() -> Option<()> {
    let base = scratch("all");
    let sessions = base.join("sessions");
    let (clean, longer) = (sessions.join("a-clean"), sessions.join("b-longer"));
    saved(&clean, false)?;
    saved(&longer, true)?;

    // A plugin's entry the library does not define, in the middle.
    let unknown = sessions.join("c-unknown");
    copy_dir(&clean, &unknown)?;
    let mut lines = main_log(&unknown)?;
    let follows: Value = ok(serde_json::from_str(lines.get(2)?))?;
    let after = read(&clean)?.agents.first()?.conversation.id(1)?;
    let probe = json!({
        "seq": follows.get("seq")?, "n": follows.get("n")?, "t": follows.get("t")?,
        "type": "aside", "id": 7,
        "aside": {"kind": "x-test/probe", "after": after,
                  "author": {"kind": {"plugin": "x-test"}}, "value": {"probe": 1}},
    });
    lines.insert(3, probe.to_string());
    assert!(lines.len() > 5, "the probe is in the middle: {lines:?}");
    write_main_log(&unknown, &lines)?;

    // The last record cut in half.
    let torn = sessions.join("d-torn");
    copy_dir(&longer, &torn)?;
    let mut lines = main_log(&torn)?;
    let last = lines.pop()?;
    let half = last.get(..last.len() / 2)?;
    ok(fs::write(
        torn.join("logs/main.jsonl"),
        lines.join("\n") + "\n" + half,
    ))?;

    // The format of the rig before this one, and a newer one.
    let older = sessions.join("e-older");
    copy_dir(&longer, &older)?;
    declare(&older, "rig-session/3")?;

    let newer = sessions.join("f-newer");
    copy_dir(&clean, &newer)?;
    declare(&newer, "rig-session/999")?;

    // Random bytes, lines among them.
    let random = sessions.join("g-random");
    copy_dir(&clean, &random)?;
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let bytes: Vec<u8> = (0..4096)
        .map(|at| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if at % 97 == 96 {
                b'\n'
            } else {
                seed.to_le_bytes().first().copied().unwrap_or_default()
            }
        })
        .collect();
    ok(fs::write(random.join("logs/main.jsonl"), &bytes))?;

    let mut before = BTreeMap::new();
    snapshot(&sessions, &mut before)?;

    // The check alone.
    let report = ok(doctor(&sessions, None))?;
    let classes: Vec<(&str, Class)> = report
        .sessions
        .iter()
        .map(|session| (session.session.as_str(), session.class))
        .collect();
    assert_eq!(
        classes,
        [
            ("a-clean", Class::Ok),
            ("b-longer", Class::Ok),
            ("c-unknown", Class::UnknownEntries),
            ("d-torn", Class::Truncated),
            ("e-older", Class::OtherFormat),
            ("f-newer", Class::OtherFormat),
            ("g-random", Class::Unreadable),
        ]
    );
    assert_eq!(report.exit, 2);
    let text = render(&report);
    let line = text.lines().find(|line| line.starts_with("c-unknown"))?;
    assert!(line.contains("x-test/probe: 1"), "{text}");
    let line = text.lines().find(|line| line.starts_with("g-random"))?;
    assert!(line.contains("log main line 1"), "{text}");

    // The fix: copies of the fixable ones only, reading back the same.
    let out = base.join("out");
    let args = [sessions.display().to_string(), "--fix".to_owned()];
    let args = [&args[..], &[out.display().to_string()]].concat();
    let (text, code) = ok(run(&args))?;
    assert_eq!(code, 2, "{text}");
    let mut copies: Vec<String> = ok(fs::read_dir(&out))?
        .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().into_owned()))
        .collect();
    copies.sort();
    assert_eq!(copies, ["c-unknown", "d-torn"], "{text}");
    for name in &copies {
        let (original, copy) = (read(&sessions.join(name))?, read(&out.join(name))?);
        for (was, now) in original.agents.iter().zip(&copy.agents) {
            assert_eq!(was.conversation.messages(), now.conversation.messages());
        }
        assert_eq!(original.agents.len(), copy.agents.len());
        assert!(copy.repairs.is_empty(), "{name}: {:?}", copy.repairs);
        let header: Value = ok(serde_json::from_str(main_log(&out.join(name))?.first()?))?;
        assert_eq!(header.get("format")?, "rig-session/4");
    }
    assert_eq!(
        line_of(&out.join("c-unknown"), "x-test/probe"),
        Some(3),
        "the probe keeps its place"
    );
    let line = text.lines().find(|line| line.starts_with("d-torn"))?;
    assert!(line.contains("dropped"), "{text}");

    let mut after = BTreeMap::new();
    snapshot(&sessions, &mut after)?;
    assert!(before == after, "the originals are unchanged");

    // Only the clean ones.
    let only_clean = base.join("clean");
    copy_dir(&clean, &only_clean.join("a-clean"))?;
    copy_dir(&longer, &only_clean.join("b-longer"))?;
    let (json, code) = ok(run(&[
        only_clean.display().to_string(),
        "--json".to_owned(),
    ]))?;
    assert_eq!(code, 0, "{json}");
    let parsed: Value = ok(serde_json::from_str(&json))?;
    assert_eq!(parsed.get("totals")?, &json!({"ok": 2}));

    fs::remove_dir_all(&base).ok();
    Some(())
}
