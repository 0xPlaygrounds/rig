use std::path::Path;

use super::{Held, Record, SessionDir, Stand, find, prefixes, reap};

fn record(pid: u32, stands: bool) -> Record {
    Record {
        pid,
        vendor: "claude-code".to_owned(),
        model: "model-1".to_owned(),
        system: String::new(),
        tools: String::new(),
        session: "s1".to_owned(),
        stands: stands.then(|| Stand {
            seen: 1,
            key: "k".to_owned(),
            said: None,
        }),
    }
}

/// A directory under `root` kept with `record`, its session gone.
fn kept(root: &Path, record: &Record) -> std::path::PathBuf {
    let held = Held::default();
    let Ok(mut dir) = SessionDir::create(root, &held) else {
        return root.to_path_buf();
    };
    assert!(dir.keep(record));
    dir.path().to_path_buf()
}

#[test]
fn a_saved_session_is_found_once_and_not_while_it_is_answering_or_another_s() {
    let name = format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple());
    let root = std::env::temp_dir().join(name);
    let me = std::process::id();
    let held = Held::default();
    // A directory that was never kept goes with its session.
    let unkept = SessionDir::create(&root, &held).map(|dir| dir.path().to_path_buf());
    assert!(unkept.is_ok_and(|path| !path.exists()));

    // One whose program is answering, and one a live process works in.
    kept(&root, &record(me, false));
    kept(&root, &record(1, true));
    assert!(find(&root, &held, |_, _| true).is_none());

    let stands = kept(&root, &record(me, true));
    let found = find(&root, &held, |record, stands| {
        record.session == "s1" && stands.seen == 1
    });
    assert_eq!(found.as_ref().map(|(dir, _)| dir.path()), Some(&*stands));
    // A session works in it now: no other goes on from it.
    assert!(find(&root, &held, |_, _| true).is_none());
    drop(found);
    assert!(find(&root, &held, |_, _| false).is_none());
    assert!(stands.exists());

    // Nothing kept is removed, whoever made it.
    reap(&root);
    assert!(stands.exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_digest_of_a_history_s_start_is_the_same_whatever_follows() {
    let texts = |all: &[&str]| {
        all.iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>()
    };
    let short = prefixes(texts(&["a", "b"]));
    let long = prefixes(texts(&["a", "b", "c"]));
    let other = prefixes(texts(&["a", "x", "c"]));

    assert_eq!(short.len(), 3);
    assert_eq!(long.get(..3), Some(&*short));
    assert_eq!(other.get(1), short.get(1));
    assert_ne!(other.get(2), short.get(2));
    // Where one message ends and the next starts is part of it.
    assert_ne!(
        prefixes(texts(&["ab"])).get(1),
        prefixes(texts(&["a", "b"])).get(2)
    );
}
