//! Shared helpers for locating on-disk test fixtures.
//!
//! Fixtures live at the workspace-root `tests/data/`. The working directory a
//! test runs in differs by runner — `cargo test` uses the crate root while
//! `cargo nextest` uses the workspace root — so a path written relative to the
//! CWD (e.g. `"tests/data/*.epub"`) resolves under one runner but not the other,
//! letting a broken fixture path stay green in CI yet fail locally.
//!
//! Always resolve fixtures through these helpers, which anchor to the crate via
//! `CARGO_MANIFEST_DIR` and are therefore CWD-independent. A CI guard rejects
//! bare `tests/data` paths in test code to keep it that way.

use std::path::PathBuf;

/// Absolute path to `<workspace>/tests/data/<filename>`, anchored to this crate.
pub(crate) fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(filename)
}

/// Absolute glob into `<workspace>/tests/data/` for `pattern` (e.g. `"*.epub"`).
pub(crate) fn fixture_glob(pattern: &str) -> String {
    fixture_path(pattern).to_string_lossy().into_owned()
}
