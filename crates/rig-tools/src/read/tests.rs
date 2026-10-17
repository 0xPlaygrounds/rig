use rig_core::tool::{
    FileRead, FilesRead, KeptEnd, LinesShown, OutputCut, Tool, ToolContext, content_hash,
};

use super::{Read, ReadArgs};
use crate::fs::resolve;

/// Reading a file publishes it by its resolved path, with the hash of the
/// bytes read and the lines shown, and a cut when lines after those were
/// left out; a failed read publishes nothing.
#[test]
fn a_read_publishes_the_file_it_read() {
    let read_tool = Read::default();
    let path = std::env::temp_dir().join(format!("rig-tools-read-{}.txt", std::process::id()));
    std::fs::write(&path, "one\ntwo\n").ok();
    let path = path.to_string_lossy().into_owned();

    let mut context = ToolContext::new();
    let read = Tool::call(&read_tool, &mut context, ReadArgs::new(&path).lines(2, 1));
    let shown = futures::executor::block_on(read).unwrap_or_default();
    assert!(shown.starts_with("     2\ttwo\n"), "{shown}");
    let published = context.result::<FilesRead>().ok().flatten();
    assert_eq!(
        published,
        Some(FilesRead(vec![
            FileRead::new(
                resolve(None, &path).to_string_lossy(),
                content_hash(b"one\ntwo\n")
            )
            .with_lines(LinesShown {
                first: 2,
                count: 1,
                total: 2
            })
        ]))
    );
    assert_eq!(context.result::<OutputCut>().ok().flatten(), None);

    let mut context = ToolContext::new();
    let first = Tool::call(&read_tool, &mut context, ReadArgs::new(&path).lines(1, 1));
    let shown = futures::executor::block_on(first).unwrap_or_default();
    assert_eq!(
        shown,
        "     1\tone\n[1 more lines; read on with offset 2]\n"
    );
    assert_eq!(
        context.result::<OutputCut>().ok().flatten(),
        Some(OutputCut {
            kept: KeptEnd::Start,
            kept_lines: 1,
            total_lines: 2,
            spill: None
        })
    );

    let mut context = ToolContext::new();
    let missing = Tool::call(
        &read_tool,
        &mut context,
        ReadArgs::new(format!("{path}.gone")),
    );
    assert!(futures::executor::block_on(missing).is_err());
    assert_eq!(context.result::<FilesRead>().ok().flatten(), None);
    std::fs::remove_file(path).ok();
}

/// A read from past a file's last line says so and how long the file is,
/// rather than coming back empty.
#[test]
fn an_offset_past_the_end_says_so() {
    let shown = ReadArgs::new("a.txt")
        .lines(5, 10)
        .show("one\ntwo\nthree\n");
    assert_eq!(shown, "[offset 5 is past the end: the file has 3 lines]\n");
}
