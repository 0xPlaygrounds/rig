use std::path::PathBuf;

use rig_core::tool::{FilesRead, Tool, ToolContext};

use super::{Search, SearchArgs, SearchHits, find, is_hidden, render};

/// A search's matches are data, what `skip` names is left out (hidden files
/// for the tool), and the tool shows the same matches as text.
#[test]
fn matches_are_data_and_skipped_files_are_left_out() {
    let dir = std::env::temp_dir().join(format!("rig-tools-search-{}", std::process::id()));
    std::fs::create_dir_all(dir.join(".github")).ok();
    std::fs::write(dir.join("code.rs"), "fn main() {}\nlet secret = 1;\n").ok();
    std::fs::write(dir.join(".env"), "SECRET=2\n").ok();
    std::fs::write(dir.join(".github").join("ci.yml"), "secret: 3\n").ok();
    let root = dir.to_string_lossy().into_owned();
    let args = SearchArgs::new("(?i)secret").in_path(&root);

    let visible = find(&args, is_hidden).unwrap_or_default();
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
        render(&visible),
        format!("{}:2: let secret = 1;", dir.join("code.rs").display())
    );

    let all = find(&args, |_| false).unwrap_or_default();
    let mut paths: Vec<PathBuf> = all.hits.into_iter().map(|hit| hit.path).collect();
    paths.sort();
    assert_eq!(
        paths,
        [
            dir.join(".env"),
            dir.join(".github").join("ci.yml"),
            dir.join("code.rs")
        ]
    );
    let no_env = find(&args, |path| path.ends_with(".env")).unwrap_or_default();
    assert_eq!(no_env.hits.len(), 2);

    let only_toml = find(&args.clone().glob("*.toml"), |_| false).unwrap_or_default();
    assert_eq!(render(&only_toml), "No matches.");
    assert!(find(&SearchArgs::new("("), is_hidden).is_err());
    std::fs::remove_dir_all(dir).ok();
}

/// The tool publishes what it found beside the text, and the files it
/// found matches in, each once, by absolute path.
#[test]
fn the_tool_publishes_its_hits_and_the_files_it_read() {
    let dir = std::env::temp_dir().join(format!("rig-tools-hits-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("a.rs"), "todo one\ntodo two\n").ok();
    std::fs::write(dir.join("b.rs"), "done\n").ok();
    let args = SearchArgs::new("todo").in_path(dir.to_string_lossy());

    let mut context = ToolContext::new();
    let text = futures::executor::block_on(Tool::call(&Search::default(), &mut context, args))
        .unwrap_or_default();
    let hits = context
        .result::<SearchHits>()
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(hits.0.hits.len(), 2);
    assert_eq!(render(&hits.0), text);
    let read = context
        .result::<FilesRead>()
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(read.0, [dir.join("a.rs").to_string_lossy()]);

    let mut context = ToolContext::new();
    let none = SearchArgs::new("nowhere").in_path(dir.to_string_lossy());
    let text = futures::executor::block_on(Tool::call(&Search::default(), &mut context, none));
    assert_eq!(text.ok().as_deref(), Some("No matches."));
    assert_eq!(context.result::<FilesRead>().ok().flatten(), None);
    std::fs::remove_dir_all(dir).ok();
}
