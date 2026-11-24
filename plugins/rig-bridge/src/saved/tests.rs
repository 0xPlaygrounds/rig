use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::{
    MAX_AGE, RECORD, Record, Reply, SessionDir, Stand, find, prefixes, prune, prune_with, reap,
};

fn record(stands: bool) -> Record {
    Record {
        vendor: "claude-code".to_owned(),
        model: "model-1".to_owned(),
        system: String::new(),
        tools: String::new(),
        session: "s1".to_owned(),
        stands: stands.then(|| Stand {
            seen: 1,
            key: "k".to_owned(),
            said: Reply::Cut(String::new()),
        }),
    }
}

/// A directory under `root` kept with `record`, its session gone.
fn kept(root: &Path, record: &Record) -> PathBuf {
    let Ok(mut dir) = SessionDir::create(root) else {
        return root.to_path_buf();
    };
    assert!(dir.keep(record));
    dir.path().to_path_buf()
}

#[test]
fn a_saved_session_is_one_session_s_at_a_time_and_not_found_while_it_answers() {
    let name = format!("rig-bridge-test-{}", uuid::Uuid::new_v4().simple());
    let root = std::env::temp_dir().join(name);
    // A directory that was never kept goes with its session.
    let unkept = SessionDir::create(&root).map(|dir| dir.path().to_path_buf());
    assert!(unkept.is_ok_and(|path| !path.exists()));

    // One whose program is answering.
    let answering = kept(&root, &record(false));
    assert!(find(&root, |_, _| true).is_none());

    let stands = kept(&root, &record(true));
    let found = find(&root, |record, stands| {
        record.session == "s1" && stands.seen == 1
    });
    assert_eq!(found.as_ref().map(|(dir, _)| dir.path()), Some(&*stands));
    // A session works in it now: no other goes on from it.
    assert!(find(&root, |_, _| true).is_none());
    drop(found);
    assert!(find(&root, |_, _| false).is_none());
    assert!(find(&root, |_, _| true).is_some());

    // What a process that is gone left unkept, an unreadable/outdated record,
    // and any unlocked unresumable record are removed; a resumable kept
    // session stays.
    let left = root.join("rig-bridge-4294967294-left");
    assert!(std::fs::create_dir(&left).is_ok());
    let stale_schema = root.join("rig-bridge-4294967294-stale");
    assert!(std::fs::create_dir(&stale_schema).is_ok());
    assert!(std::fs::write(stale_schema.join(RECORD), b"{}").is_ok());
    reap(&root);
    assert!(!left.exists());
    assert!(!stale_schema.exists());
    assert!(!answering.exists());
    assert!(stands.exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn pruning_enforces_age_and_count_caps_and_skips_live_sessions() {
    let name = format!("rig-bridge-prune-test-{}", uuid::Uuid::new_v4().simple());
    let root = std::env::temp_dir().join(name);
    let now = SystemTime::now();

    let set_age = |dir: &Path, secs_ago: u64| {
        let updated = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join(RECORD))
            .and_then(|file| file.set_modified(now - Duration::from_secs(secs_ago)));
        assert!(updated.is_ok());
    };

    let expired = kept(&root, &record(true));
    let older = kept(&root, &record(true));
    let newer = kept(&root, &record(true));
    let newest = kept(&root, &record(true));
    let live_dir = SessionDir::create(&root);
    assert!(live_dir.is_ok());
    let Ok(mut live_dir) = live_dir else {
        return;
    };
    assert!(live_dir.keep(&record(true)));
    let live = live_dir.path().to_path_buf();

    set_age(&expired, MAX_AGE.as_secs() + 60);
    set_age(&older, 300);
    set_age(&newer, 200);
    set_age(&newest, 100);
    set_age(&live, MAX_AGE.as_secs() + 120);

    // Cap at 2 kept: `expired` goes by age, `older` goes by cap, `live` stays
    // because it holds `rig.lock`, and `newer` + `newest` stay.
    assert_eq!(prune_with(&root, now, Some(MAX_AGE), 2), 2);
    assert!(!expired.exists());
    assert!(!older.exists());
    assert!(newer.exists());
    assert!(newest.exists());
    assert!(live.exists());

    // Manual prune removes every remaining idle session, still skipping `live`.
    assert_eq!(prune(&root), 2);
    assert!(!newer.exists());
    assert!(!newest.exists());
    assert!(live.exists());
    assert_eq!(prune(&root), 0);

    drop(live_dir);
    assert_eq!(prune(&root), 1);
    assert!(!live.exists());
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
