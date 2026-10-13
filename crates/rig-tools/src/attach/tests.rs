use std::path::PathBuf;

use rig_core::message::UserContent;

use super::{attached_file, attachments};
use crate::workspace::Local;

/// The crate's own directory, where its manifest is.
fn base() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_text_file_is_attached_as_numbered_lines_and_typed_text_is_not_one() {
    let path = base().join("Cargo.toml").to_string_lossy().into_owned();
    let attached = attachments(&format!("look at @{path}, please"), &base(), &Local);
    assert!(attached.notes.is_empty(), "{:?}", attached.notes);
    assert_eq!(attached.files.len(), 1);
    let text = match attached.files.first().map(|(_, content)| content) {
        Some(UserContent::Text(text)) => text.text.as_str(),
        _ => "",
    };
    assert!(text.contains("     1\t[package]"), "{text}");
    let lines = std::fs::read_to_string(&path)
        .ok()
        .map(|manifest| manifest.lines().count());
    assert_eq!(
        attached_file(text).map(|(label, lines)| (label.to_owned(), Some(lines))),
        Some((path, lines))
    );
    assert_eq!(attached_file("<file path=\"x\"> is how it starts"), None);
}

/// A reference resolves from the base, once however often it is named; a
/// path with no file there is missing, and a word that only looks like a
/// name stays text.
#[test]
fn references_resolve_from_the_base_and_missing_paths_are_reported() {
    let attached = attachments(
        "@Cargo.toml and @Cargo.toml, @missing/x.rs, @gone.rs. Ask @alice.smith or @bob",
        &base(),
        &Local,
    );
    let labels: Vec<&str> = attached
        .files
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels, ["Cargo.toml"]);
    assert_eq!(attached.missing, ["missing/x.rs", "gone.rs"]);
    assert!(attached.notes.is_empty(), "{:?}", attached.notes);
}

/// A file with no extension, such as a `Makefile`, is attached when it is
/// there; the same word with no file there stays text.
#[test]
fn an_extensionless_file_that_is_there_is_attached() {
    let dir = std::env::temp_dir().join(format!("rig-tools-attach-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("Makefile"), "all:\n").ok();
    let attached = attachments("run @Makefile, ask @reviewer", &dir, &Local);
    let labels: Vec<&str> = attached
        .files
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels, ["Makefile"]);
    assert!(attached.missing.is_empty(), "{:?}", attached.missing);
    std::fs::remove_dir_all(dir).ok();
}
