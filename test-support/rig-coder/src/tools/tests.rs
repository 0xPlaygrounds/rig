use std::sync::Arc;

use super::*;

fn text(output: ToolResult) -> String {
    match output {
        Ok(output) => output.as_text().unwrap_or_default().to_string(),
        Err(error) => format!("error: {error}"),
    }
}

fn workspace() -> (tempdir::Dir, Arc<PathBuf>) {
    let dir = tempdir::Dir::new();
    let root = Arc::new(dir.path().to_path_buf());
    (dir, root)
}

/// A scratch directory removed on drop.
mod tempdir {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    pub struct Dir(PathBuf);

    impl Dir {
        pub fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rig-coder-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _created = std::fs::create_dir_all(&path);
            Self(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _removed = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[tokio::test]
async fn edit_requires_a_unique_match_and_shows_the_region() {
    let (_dir, root) = workspace();
    let written = write_file(
        root.clone(),
        WriteArgs {
            path: "a/b.txt".into(),
            content: "one\ntwo\ntwo\nthree\n".into(),
        },
    )
    .await;
    assert!(text(written).starts_with("wrote 18 bytes"));

    let ambiguous = edit_file(
        root.clone(),
        EditArgs {
            path: "a/b.txt".into(),
            old_string: "two".into(),
            new_string: "2".into(),
            replace_all: false,
        },
    )
    .await;
    assert!(text(ambiguous).contains("matches 2 times"));

    let edited = text(
        edit_file(
            root.clone(),
            EditArgs {
                path: "a/b.txt".into(),
                old_string: "two\nthree".into(),
                new_string: "3".into(),
                replace_all: false,
            },
        )
        .await,
    );
    assert!(edited.contains("     3\t3"), "{edited}");

    let read = text(
        read_file(
            root,
            ReadArgs {
                path: "a/b.txt".into(),
                offset: Some(2),
                limit: Some(1),
            },
        )
        .await,
    );
    assert!(read.starts_with("     2\ttwo\n"), "{read}");
    assert!(read.contains("showing lines 2-2 of 3"), "{read}");
}

#[tokio::test]
async fn bash_reports_exit_code_and_bounds_output() {
    let (_dir, root) = workspace();
    let output = text(
        bash(
            root.clone(),
            BashArgs {
                command: "head -c 100000 /dev/zero | tr '\\0' x; echo; exit 3".into(),
                timeout_secs: None,
            },
        )
        .await,
    );
    assert!(output.starts_with("exit code: 3\n"), "{:?}", output.lines().next());
    assert!(output.contains("bytes omitted"));
    assert!(output.len() < OUTPUT_LIMIT + 200);
}

#[tokio::test]
async fn bash_does_not_wait_for_background_children() {
    let (_dir, root) = workspace();
    let started = std::time::Instant::now();
    let output = text(
        bash(
            root,
            BashArgs {
                command: "sleep 30 & echo started".into(),
                timeout_secs: Some(20),
            },
        )
        .await,
    );
    assert!(output.contains("started"), "{output}");
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[tokio::test]
async fn bash_timeout_kills_the_command() {
    let (_dir, root) = workspace();
    let output = text(
        bash(
            root,
            BashArgs {
                command: "echo before; sleep 30".into(),
                timeout_secs: Some(1),
            },
        )
        .await,
    );
    assert!(output.starts_with("timed out after 1s"), "{output}");
    assert!(output.contains("before"));
}

#[tokio::test]
async fn grep_and_list_honour_the_workspace() {
    let (_dir, root) = workspace();
    for (path, content) in [("src/lib.rs", "fn alpha() {}\n"), ("notes.md", "alpha\n")] {
        let _written = write_file(
            root.clone(),
            WriteArgs {
                path: path.into(),
                content: content.into(),
            },
        )
        .await;
    }
    let listed = text(
        list_files(
            root.clone(),
            ListArgs {
                path: None,
                max_depth: None,
            },
        )
        .await,
    );
    assert_eq!(listed, "notes.md\nsrc/\nsrc/lib.rs");
    let found = text(
        grep(
            root,
            GrepArgs {
                pattern: "alpha".into(),
                path: None,
                glob: Some("*.rs".into()),
                case_insensitive: false,
            },
        )
        .await,
    );
    assert_eq!(found, "src/lib.rs:1: fn alpha() {}");
}
