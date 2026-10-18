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
    assert!(shown.starts_with("two\n"), "{shown}");
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
    assert_eq!(shown, "one\n[1 more lines; read on with offset 2]\n");
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

/// A file of long lines reads as far as one of short lines: its lines are
/// cut to fit the bytes a read shows, and a note says so and how to see
/// more of each, while a read of a few of them shows them whole.
#[test]
fn long_lines_are_cut_to_fit_rather_than_stopping_early() {
    let long = "x".repeat(640);
    let text: String = (0..475).map(|n| format!("{n} {long}\n")).collect();
    let shown = ReadArgs::new("data.txt").show(&text);
    let lines: Vec<&str> = shown.lines().collect();
    assert_eq!(lines.len(), 476, "{:?}", lines.last());
    assert!(lines.first().is_some_and(|line| line.starts_with("0 x") && line.ends_with(" […]")));
    assert!(lines.get(474).is_some_and(|line| line.starts_with("474 x")));
    let note = lines.last().copied().unwrap_or_default();
    assert!(
        note.starts_with("[475 lines were cut at ")
            && note.ends_with(" characters to fit 50 KB; a smaller limit shows more of each]"),
        "{note}"
    );
    assert!(shown.len() <= crate::MAX_BYTES, "{}", shown.len());
    let few = ReadArgs::new("data.txt").lines(1, 20).show(&text);
    assert!(few.starts_with(&format!("0 {long}\n1 {long}\n")), "{few:.80}");
    assert!(few.ends_with("[455 more lines; read on with offset 21]\n"), "{few}");
}

/// A file of ordinary lines that does not fit keeps its lines whole and
/// says the read stopped at its bytes.
#[test]
fn a_read_that_stops_at_its_bytes_says_so() {
    let text: String = (0..3000).map(|n| format!("let line_{n:04} = {n:030};\n")).collect();
    let shown = ReadArgs::new("big.rs").show(&text);
    let last = shown.lines().last().unwrap_or_default();
    assert!(
        last.ends_with(", as this read reached its 50 KB]") && last.contains("more lines; read on"),
        "{last}"
    );
    assert!(!shown.contains("[…]"));
}
