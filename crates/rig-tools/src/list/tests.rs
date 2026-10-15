use rig_core::tool::{OutputCut, Tool, ToolContext, ToolErrorKind};

use super::{FilesListed, List, ListArgs};

/// At depth 1 a listing shows the directory's own files and each directory
/// under it once, and leaves out hidden, `target` and ignored files; with
/// no depth it shows every file, and publishes what it showed.
#[test]
fn a_listing_shows_what_files_lists_down_to_its_depth() {
    let dir = std::env::temp_dir().join(format!("rig-tools-list-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    for sub in ["src/deep", "target/debug", ".git"] {
        std::fs::create_dir_all(dir.join(sub)).ok();
    }
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    std::fs::write(dir.join(".gitignore"), "*.log\n").ok();
    for file in [
        "README.md",
        "run.log",
        "src/lib.rs",
        "src/deep/mod.rs",
        "target/debug/app",
    ] {
        std::fs::write(dir.join(file), "x").ok();
    }
    let list = List {
        base: Some(dir.clone()),
        confine: true,
        ..List::default()
    };
    let run = |args: ListArgs| {
        let mut context = ToolContext::new();
        let text = futures::executor::block_on(list.call(&mut context, args));
        (text.unwrap_or_default(), context)
    };

    let (shallow, context) = run(ListArgs::new(".").depth(1));
    assert_eq!(shallow, "README.md\nsrc/\n");
    let listed: Option<FilesListed> = context.result().ok().flatten();
    assert_eq!(
        listed.map(|listed| (listed.root, listed.entries)),
        Some((
            dir.to_string_lossy().into_owned(),
            vec!["README.md".to_owned(), "src/".to_owned()]
        ))
    );
    assert_eq!(context.result::<OutputCut>().ok().flatten(), None);

    let (all, _) = run(ListArgs::new("src"));
    assert_eq!(all, "deep/mod.rs\nlib.rs\n");
    let (refused, _) = run(ListArgs::new(".."));
    assert_eq!(refused, "");
    std::fs::remove_dir_all(dir).ok();
}

/// Hidden files are listed when asked for, or when the glob names a
/// hidden path, and ignored files and `.git` stay out either way.
#[test]
fn hidden_files_are_listed_when_asked_for_or_named() {
    let dir = std::env::temp_dir().join(format!("rig-tools-hidden-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    for sub in [".github/workflows", ".git", "src"] {
        std::fs::create_dir_all(dir.join(sub)).ok();
    }
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    std::fs::write(dir.join(".gitignore"), ".env\n").ok();
    for file in [
        ".github/workflows/ci.yml",
        ".git/HEAD",
        ".env",
        "src/lib.rs",
    ] {
        std::fs::write(dir.join(file), "x").ok();
    }
    let list = List {
        base: Some(dir.clone()),
        ..List::default()
    };
    let run = |args: ListArgs| {
        futures::executor::block_on(list.call(&mut ToolContext::new(), args)).unwrap_or_default()
    };

    assert_eq!(run(ListArgs::new(".")), "src/lib.rs\n");
    assert_eq!(
        run(ListArgs::new(".").hidden()),
        ".github/workflows/ci.yml\n.gitignore\nsrc/lib.rs\n"
    );
    assert_eq!(
        run(ListArgs::new(".").glob(".github/**")),
        ".github/workflows/ci.yml\n"
    );
    std::fs::remove_dir_all(dir).ok();
}

/// A path that does not exist is not found, not an empty listing.
#[test]
fn a_missing_path_is_not_found() {
    let missing = std::env::temp_dir().join(format!("rig-tools-gone-{}", std::process::id()));
    let listed = futures::executor::block_on(List::default().call(
        &mut ToolContext::new(),
        ListArgs::new(missing.to_string_lossy()),
    ));
    assert_eq!(
        listed.err().map(|error| error.kind()),
        Some(ToolErrorKind::NotFound)
    );
}
