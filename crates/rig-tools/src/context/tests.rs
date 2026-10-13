use std::path::PathBuf;

use super::{Instructions, Limits};
use crate::workspace::Local;

/// The limits a reader is given cut each file, and share what is left of
/// the total out from the most specific file up.
#[test]
fn the_reader_reads_as_much_as_its_limits_allow() {
    let root = std::env::temp_dir().join(format!("rig-tools-context-{}", std::process::id()));
    let inner = root.join("inner");
    std::fs::create_dir_all(&inner).ok();
    std::fs::write(root.join("AGENTS.md"), "general rules\n").ok();
    std::fs::write(inner.join("AGENTS.md"), "specific rules\n").ok();
    let limits = Limits {
        file_bytes: 8,
        total_bytes: 12,
    };
    let found = Instructions::discover_with(Vec::<PathBuf>::new(), &inner, &Local, limits);
    let texts: Vec<(&str, bool)> = found
        .files
        .iter()
        .filter(|file| file.path.starts_with(&root))
        .map(|file| (file.contents.text.as_str(), file.contents.cut()))
        .collect();
    assert_eq!(texts, [("gene", true), ("specific", true)]);
    std::fs::remove_dir_all(root).ok();
}

/// A file the total leaves no room for is still listed, as cut to
/// nothing, so the prompt names it and its reader can say so.
#[test]
fn a_file_the_total_leaves_no_room_for_is_listed_as_cut() {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe.with_file_name(format!("rig-tools-context-room-{}", std::process::id()));
    let inner = root.join("inner");
    std::fs::create_dir_all(&inner).ok();
    std::fs::write(root.join("AGENTS.md"), "general rules\n").ok();
    std::fs::write(inner.join("AGENTS.md"), "specific\n").ok();
    let limits = Limits {
        file_bytes: 9,
        total_bytes: 9,
    };
    let found = Instructions::discover_with(Vec::<PathBuf>::new(), &inner, &Local, limits);
    let texts: Vec<(&str, bool)> = found
        .files
        .iter()
        .filter(|file| file.path.starts_with(&root))
        .map(|file| (file.contents.text.as_str(), file.contents.cut()))
        .collect();
    assert_eq!(texts, [("", true), ("specific\n", false)]);
    let general = root.join("AGENTS.md");
    assert!(
        found
            .prompt_text()
            .contains(&format!("<file path=\"{}\">", general.display())),
        "{}",
        found.prompt_text()
    );
    std::fs::remove_dir_all(root).ok();
}
