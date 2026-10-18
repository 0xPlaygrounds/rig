use rig_core::tool::{FileRead, FilesRead, Tool, ToolContext, ToolErrorKind, content_hash};

use super::{Search, SearchArgs, SearchHits, find, render};
use crate::workspace::Local;

/// A search's matches are data, hidden files and those outside the glob are
/// left out, the tool shows the same matches as text, and a missing path is
/// not found rather than empty.
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
    let missing = SearchArgs::new("secret").in_path(dir.join("missing").to_string_lossy());
    assert_eq!(
        find(&Local, &missing).err().map(|error| error.kind()),
        Some(ToolErrorKind::NotFound)
    );
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

/// A search names each hit as `read` on the same base takes it: below the
/// base for a relative path, absolute for an absolute one, also when that
/// path is under the base.
#[test]
fn a_search_names_hits_as_read_takes_them() {
    let dir = std::env::temp_dir().join(format!("rig-tools-base-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("src")).ok();
    std::fs::write(dir.join("src").join("a.rs"), "todo one\n").ok();
    let search = Search {
        base: Some(dir.clone()),
        ..Search::default()
    };
    let text = |args| {
        futures::executor::block_on(Tool::call(&search, &mut ToolContext::new(), args))
            .unwrap_or_default()
    };
    assert_eq!(text(SearchArgs::new("todo")), "src/a.rs:1: todo one");
    assert_eq!(
        text(SearchArgs::new("todo").in_path("src")),
        "src/a.rs:1: todo one"
    );
    let absolute = dir.join("src");
    assert_eq!(
        text(SearchArgs::new("todo").in_path(absolute.to_string_lossy())),
        format!("{}:1: todo one", absolute.join("a.rs").display())
    );
    std::fs::remove_dir_all(dir).ok();
}

/// With `files`, a search names each file with a match once; with an
/// empty pattern, every file, empty and binary ones too, read or not.
#[test]
fn a_search_for_files_lists_their_paths() {
    let dir = std::env::temp_dir().join(format!("rig-tools-files-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("src")).ok();
    std::fs::write(dir.join("src").join("a.rs"), "todo one\ntodo two\n").ok();
    std::fs::write(dir.join("src").join("b.rs"), "").ok();
    std::fs::write(dir.join("logo.png"), [0xff, 0xfe]).ok();
    let search = Search {
        base: Some(dir.clone()),
        ..Search::default()
    };
    let mut context = ToolContext::new();
    let found = futures::executor::block_on(Tool::call(
        &search,
        &mut context,
        SearchArgs::new("todo").files(),
    ));
    assert_eq!(found.ok().as_deref(), Some("src/a.rs"));
    let read = context.result::<FilesRead>().ok().flatten();
    assert_eq!(read.map(|read| read.0.len()), Some(1));
    let mut context = ToolContext::new();
    let all = futures::executor::block_on(Tool::call(
        &search,
        &mut context,
        SearchArgs::new("").files(),
    ));
    let mut all: Vec<String> = all.unwrap_or_default().lines().map(str::to_owned).collect();
    all.sort();
    assert_eq!(all, ["logo.png", "src/a.rs", "src/b.rs"]);
    assert_eq!(context.result::<FilesRead>().ok().flatten(), None);
    std::fs::remove_dir_all(dir).ok();
}

/// Whatever line endings a file has, a search finds every line the pattern
/// matches, as if each line were matched alone: anchors at a later line's
/// start and end (also before `\r\n`), text anchors and flags the pattern
/// turns off, and a `.` over a `\r` inside a line.
#[test]
fn every_matching_line_is_found_whatever_the_line_ends() {
    let dir = std::env::temp_dir().join(format!("rig-tools-lines-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let cases = [
        ("first\nsecond\n", "^second", 2),
        ("first\r\nsecond\r\nthird\r\n", "second$", 2),
        ("first\r\nsecond\r\n", r"^second\z", 2),
        ("first\nsecond\n", r"\Asecond", 2),
        ("first\nsecond\n", "(?-m)^second$", 2),
        ("first\na\rb\n", "a.b", 2),
    ];
    for (index, (text, pattern, line)) in cases.into_iter().enumerate() {
        let file = dir.join(format!("{index}.txt"));
        std::fs::write(&file, text).ok();
        let args = SearchArgs::new(pattern).in_path(file.to_string_lossy());
        let found = find(&Local, &args).unwrap_or_default();
        let lines: Vec<usize> = found.hits.iter().map(|hit| hit.line).collect();
        assert_eq!(lines, [line], "{pattern:?} in {text:?}");
    }
    std::fs::remove_dir_all(dir).ok();
}
