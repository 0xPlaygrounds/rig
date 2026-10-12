use rig_core::tool::{OutputCut, Tool, ToolContext};

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
