use std::path::PathBuf;

use super::{SearchArgs, find, render};

/// A search's matches are data, hidden files are searched only when asked,
/// and the tool shows the same matches as text.
#[test]
fn matches_are_data_and_hidden_files_are_searched_when_asked() {
    let dir = std::env::temp_dir().join(format!("rig-tools-search-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("code.rs"), "fn main() {}\nlet secret = 1;\n").ok();
    std::fs::write(dir.join(".env"), "SECRET=2\n").ok();
    let root = dir.to_string_lossy().into_owned();
    let args = SearchArgs::new("(?i)secret").in_path(&root);

    let visible = find(&args, false).unwrap_or_default();
    assert_eq!(
        visible
            .hits
            .iter()
            .map(|hit| (hit.path.clone(), hit.line, hit.text.as_str()))
            .collect::<Vec<_>>(),
        [(dir.join("code.rs"), 2, "let secret = 1;")]
    );
    assert!(!visible.stopped);
    assert_eq!(
        render(visible),
        format!("{}:2: let secret = 1;", dir.join("code.rs").display())
    );

    let all = find(&args, true).unwrap_or_default();
    let mut paths: Vec<PathBuf> = all.hits.into_iter().map(|hit| hit.path).collect();
    paths.sort();
    assert_eq!(paths, [dir.join(".env"), dir.join("code.rs")]);

    let only_rust = find(&args.clone().glob("*.toml"), true).unwrap_or_default();
    assert_eq!(render(only_rust), "No matches.");
    assert!(find(&SearchArgs::new("("), false).is_err());
    std::fs::remove_dir_all(dir).ok();
}
