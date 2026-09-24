//! The committed effect goldens against the migration.
//!
//! Every golden is in the current format and is exactly what the current
//! types write, so the migration leaves it untouched. Goldens change only
//! through the migration: `regenerate_goldens_from_base` rewrites each one
//! from its content at a base revision, the same function `rig-migrate`
//! applies to a file.
//!
//! ```text
//! RIG_MIGRATE_FROM=<rev> cargo test -p rig-cassette --features http \
//!     --test migrate regenerate_goldens_from_base -- --ignored
//! ```
//!
//! The `http` feature keeps JSON key order and float spelling, as the
//! `rig-migrate` command's `migrate` feature does.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};

use rig_cassette::effect_log::EffectLog;
use rig_cassette::migrate::{Migration, migrate};
use serde_json::Value;

fn effects() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/effects")
}

/// Every golden file, relative to the crate root.
fn goldens() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for directory in [effects(), effects().join("world")] {
        for entry in std::fs::read_dir(&directory).expect("the effects corpus") {
            let path = entry.expect("a corpus entry").path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn is_program_scenes(path: &Path) -> bool {
    path.to_string_lossy().ends_with(".programs.json")
}

/// A world golden's program scenes: scope → `[serving policy, checkpoint]`,
/// with every checkpoint migrated.
fn migrate_scenes(mut scenes: Value) -> (Value, bool) {
    let mut changed = false;
    for scene in scenes
        .as_object_mut()
        .expect("program scenes are a map")
        .values_mut()
    {
        let checkpoint = scene
            .get_mut(1)
            .expect("a scene is [policy, checkpoint]")
            .take();
        let (checkpoint, migration) = migrate(checkpoint).expect("the checkpoint migrates");
        changed |= migration != Migration::Current;
        scene[1] = checkpoint;
    }
    (scenes, changed)
}

fn render(value: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(value).expect("JSON renders")
    )
}

#[test]
fn the_migration_produces_the_checkpoint_format_rig_ecs_reads() {
    assert_eq!(
        rig_cassette::migrate::ECS_CHECKPOINT_FORMAT,
        rig_ecs::checkpoint::CHECKPOINT_FORMAT
    );
}

#[test]
fn every_golden_is_current_and_canonical() {
    let files = goldens();
    assert!(files.len() > 3000, "the corpus is present");
    for path in files {
        let text = std::fs::read_to_string(&path).expect("golden text");
        let value: Value = serde_json::from_str(&text).expect("golden JSON");
        if is_program_scenes(&path) {
            let (_, changed) = migrate_scenes(value);
            assert!(!changed, "{} is not current", path.display());
            continue;
        }
        let (_, migration) = migrate(value).expect("the golden migrates");
        assert_eq!(
            migration,
            Migration::Current,
            "{} is not current",
            path.display()
        );
        let log: EffectLog = serde_json::from_str(&text).expect("the golden loads");
        assert_eq!(
            format!(
                "{}\n",
                serde_json::to_string_pretty(&log).expect("the log renders")
            ),
            text,
            "{} is not what the current types write",
            path.display()
        );
    }
}

#[test]
#[ignore = "rewrites the corpus; run explicitly with RIG_MIGRATE_FROM"]
fn regenerate_goldens_from_base() {
    let base = std::env::var("RIG_MIGRATE_FROM").expect("RIG_MIGRATE_FROM names the base revision");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for path in goldens() {
        let relative = path.strip_prefix(root).expect("a corpus path");
        let output = std::process::Command::new("git")
            .arg("show")
            .arg(format!("{base}:./{}", relative.display()))
            .current_dir(root)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git show {base}:{}: {}",
            relative.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).expect("base golden JSON");
        let current = if is_program_scenes(&path) {
            migrate_scenes(value).0
        } else {
            migrate(value).expect("the base golden migrates").0
        };
        std::fs::write(&path, render(&current)).expect("the golden writes");
    }
}

#[test]
fn the_migration_produces_the_run_format_rig_agent_reads() {
    assert_eq!(
        rig_cassette::migrate::AGENT_RUN_FORMAT,
        rig_agent::run::RUN_FORMAT
    );
}
