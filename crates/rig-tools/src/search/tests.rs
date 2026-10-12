use rig_core::tool::{FileRead, FilesRead, Tool, ToolContext, content_hash};

use super::{Search, SearchArgs, SearchHits, find, render};
use crate::workspace::Local;

/// A search's matches are data, hidden files and those outside the glob are
/// left out, and the tool shows the same matches as text.
#[test]
fn matches_are_data_and_hidden_files_are_left_out() {
    let dir = std::env::temp_dir().join(format!("rig-tools-search-{}", std::process::id()));
    std::fs::create_dir_all(dir.join(".github")).ok();
    std::fs::write(dir.join("code.rs"), "fn main() {}\nlet secret = 1;\n").ok();
    std::fs::write(dir.join(".env"), "SECRET=2\n").ok();
    std::fs::write(dir.join(".github").join("ci.yml"), "secret: 3\n").ok();
    let root = dir.to_string_lossy().into_owned();
    let args = SearchArgs::new("(?i)secret").in_path(&root);

    let visible = find(&Local, &args).unwrap_or_default();
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

    let only_toml = find(&Local, &args.clone().glob("*.toml")).unwrap_or_default();
    assert_eq!(render(&only_toml), "No matches.");
    assert!(find(&Local, &SearchArgs::new("(")).is_err());
    assert!(find(&Local, &args.glob("[")).is_err());
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
    let a = crate::fs::resolve(Some(&dir), "a.rs");
    let a = FileRead::new(a.to_string_lossy(), content_hash(b"todo one\ntodo two\n"));
    assert_eq!(read.0, [a]);

    let mut context = ToolContext::new();
    let none = SearchArgs::new("nowhere").in_path(dir.to_string_lossy());
    let text = futures::executor::block_on(Tool::call(&Search::default(), &mut context, none));
    assert_eq!(text.ok().as_deref(), Some("No matches."));
    assert_eq!(context.result::<FilesRead>().ok().flatten(), None);
    std::fs::remove_dir_all(dir).ok();
}

/// A search from a base names its hits by their paths below it, as the
/// model would name them to the tools on that base.
#[test]
fn a_search_from_a_base_names_hits_below_it() {
    let dir = std::env::temp_dir().join(format!("rig-tools-base-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("src")).ok();
    std::fs::write(dir.join("src").join("a.rs"), "todo one\n").ok();
    let search = Search {
        base: Some(dir.clone()),
        ..Search::default()
    };
    let mut context = ToolContext::new();
    let text =
        futures::executor::block_on(Tool::call(&search, &mut context, SearchArgs::new("todo")));
    assert_eq!(text.ok().as_deref(), Some("src/a.rs:1: todo one"));
    std::fs::remove_dir_all(dir).ok();
}
