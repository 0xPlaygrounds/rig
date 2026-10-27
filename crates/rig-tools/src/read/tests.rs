use base64::Engine;
use rig_core::message::{DocumentSourceKind, ImageMediaType, ToolResultContent};
use rig_core::tool::{
    FileRead, FilesRead, KeptEnd, LinesShown, OutputCut, Tool, ToolContext, ToolExecutionError,
    ToolOutput, content_hash,
};

use super::{Read, ReadArgs};
use crate::fs::resolve;

/// What `read` answers for `args`, and the context it published to.
fn call(args: ReadArgs) -> (Result<ToolOutput, ToolExecutionError>, ToolContext) {
    let mut context = ToolContext::new();
    let read = futures::executor::block_on(Tool::call(&Read::default(), &mut context, args));
    (read, context)
}

/// The text `read` shows for `args`, empty when it fails.
fn text(args: ReadArgs) -> String {
    let (read, _) = call(args);
    read.ok()
        .and_then(|shown| shown.as_text().map(str::to_owned))
        .unwrap_or_default()
}

/// Reading a file publishes it by its resolved path, with the hash of the
/// bytes read and the lines shown, and a cut when lines after those were
/// left out; a failed read publishes nothing.
#[test]
fn a_read_publishes_the_file_it_read() {
    let path = std::env::temp_dir().join(format!("rig-tools-read-{}.txt", std::process::id()));
    std::fs::write(&path, "one\ntwo\n").ok();
    let path = path.to_string_lossy().into_owned();

    let (shown, context) = call(ReadArgs::new(&path).lines(2, 1));
    let shown = shown
        .ok()
        .and_then(|shown| shown.as_text().map(str::to_owned));
    assert!(shown.is_some_and(|shown| shown.starts_with("two\n")));
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

    let (shown, context) = call(ReadArgs::new(&path).lines(1, 1));
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("one\n[1 more lines; read on with offset 2]\n")
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

    let (missing, context) = call(ReadArgs::new(format!("{path}.gone")));
    assert!(missing.is_err());
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
    assert!(
        lines
            .first()
            .is_some_and(|line| line.starts_with("0 x") && line.ends_with(" […]"))
    );
    assert!(lines.get(474).is_some_and(|line| line.starts_with("474 x")));
    let note = lines.last().copied().unwrap_or_default();
    assert!(
        note.starts_with("[475 lines were cut at ")
            && note.ends_with(" characters to fit 50 KB; a smaller limit shows more of each]"),
        "{note}"
    );
    assert!(shown.len() <= crate::MAX_BYTES, "{}", shown.len());
    let few = ReadArgs::new("data.txt").lines(1, 20).show(&text);
    assert!(
        few.starts_with(&format!("0 {long}\n1 {long}\n")),
        "{few:.80}"
    );
    assert!(
        few.ends_with("[455 more lines; read on with offset 21]\n"),
        "{few}"
    );
}

/// A file of ordinary lines that does not fit keeps its lines whole and
/// says the read stopped at its bytes.
#[test]
fn a_read_that_stops_at_its_bytes_says_so() {
    let text: String = (0..3000)
        .map(|n| format!("let line_{n:04} = {n:030};\n"))
        .collect();
    let shown = ReadArgs::new("big.rs").show(&text);
    let last = shown.lines().last().unwrap_or_default();
    assert!(
        last.ends_with(", as this read reached its 50 KB]") && last.contains("more lines; read on"),
        "{last}"
    );
    assert!(!shown.contains("[…]"));
}

/// An image named and shaped as one comes back as a line saying what it
/// is and the image itself, published by the hash of its bytes; one larger
/// than a provider takes is refused, and a file named as an image that is
/// not one is read as any other file.
#[test]
fn an_image_is_read_as_an_image() {
    let dir = std::env::temp_dir().join(format!("rig-tools-read-image-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let dir = resolve(None, &dir.to_string_lossy());
    let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
    let at = |name: &str| dir.join(name).to_string_lossy().into_owned();
    std::fs::write(at("logo.png"), &png).ok();
    let mut large = png.clone();
    large.resize(5 * 1024 * 1024 + 1, 0);
    std::fs::write(at("large.png"), &large).ok();
    std::fs::write(at("notes.png"), "not an image\n").ok();

    let (shown, context) = call(ReadArgs::new(at("logo.png")));
    let content = shown.map(ToolOutput::into_content).unwrap_or_default();
    assert_eq!(content.len(), 2, "{content:?}");
    let line = content.first().and_then(ToolResultContent::as_text);
    assert_eq!(
        line,
        Some(format!("{}: a PNG image of 1 KB", at("logo.png")).as_str())
    );
    let image = match content.get(1) {
        Some(ToolResultContent::Image(image)) => Some(image),
        _ => None,
    };
    assert_eq!(
        image.and_then(|image| image.media_type.clone()),
        Some(ImageMediaType::PNG)
    );
    let data = base64::engine::general_purpose::STANDARD.encode(&png);
    assert_eq!(
        image.map(|image| &image.data),
        Some(&DocumentSourceKind::Base64(data))
    );
    assert_eq!(
        context.result::<FilesRead>().ok().flatten(),
        Some(FilesRead(vec![FileRead::new(
            at("logo.png"),
            content_hash(&png)
        )]))
    );

    let (refused, _) = call(ReadArgs::new(at("large.png")));
    let refused = refused
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        refused.ends_with(
            "large.png: it is an image of 5121 KB, more than the 5120 KB a provider takes"
        ),
        "{refused}"
    );
    assert_eq!(text(ReadArgs::new(at("notes.png"))), "not an image\n");
    std::fs::remove_dir_all(dir).ok();
}

/// A file that is not UTF-8 text is refused with what to do about it, not
/// the decoder's words.
#[test]
fn a_binary_file_is_refused_as_not_text() {
    let path = std::env::temp_dir().join(format!("rig-tools-read-{}.bin", std::process::id()));
    std::fs::write(&path, b"\xff\xfe\0\x01").ok();
    let path = resolve(None, &path.to_string_lossy());
    let path = path.to_string_lossy().into_owned();
    let (refused, _) = call(ReadArgs::new(&path));
    let refused = refused
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains(&format!(
            "{path} is not a text file (not UTF-8); `file '{path}'` in `shell` shows what it is"
        )),
        "{refused}"
    );
    std::fs::remove_file(path).ok();
}

/// The note on a line cut short names it and the byte its rest starts at,
/// and gives a command that shows the rest in lines `shell` keeps whole,
/// starting exactly where the read stopped.
#[cfg(unix)]
#[test]
fn a_cut_line_can_be_read_on_in_shell() {
    let dir = std::env::temp_dir().join(format!("rig-tools-read-it's-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let dir = resolve(None, &dir.to_string_lossy());
    let path = dir.join("long.txt");
    let ascii = "x".repeat(30_000);
    let wide = "é".repeat(2_500);
    let three = "日".repeat(2_500);
    std::fs::write(&path, format!("short\n{ascii}\n{wide}\n{three}\n")).ok();
    let path = path.to_string_lossy().into_owned();
    let quoted = format!("'{}'", path.replace('\'', "'\\''"));

    let shown = text(ReadArgs::new(&path).lines(2, 1));
    let note = shown.lines().nth(1).unwrap_or_default();
    assert_eq!(
        note,
        format!(
            "[1 line was cut at 2000 characters; line 2 shows 2000 of its 30000 bytes: in \
             `shell`, `sed -n '2p;2q' {quoted} | cut -b 2001-12000 | fold -b -w 2000` shows the \
             next 10000 bytes, and `cut -b 12001-22000` the ones after]"
        )
    );
    let shown = text(ReadArgs::new(&path).lines(3, 2));
    let note = shown.lines().nth(2).unwrap_or_default();
    assert!(
        note.starts_with(
            "[2 lines were cut at 2000 characters; the first, line 3, shows 4000 of its 5000 bytes"
        ) && note
            .ends_with("| cut -b 4001-5000 | fold -b -w 2000` shows the remaining 1000 bytes]"),
        "{note}"
    );

    let rest = |note: &str| {
        let command = note.split('`').nth(3).unwrap_or_default();
        let ran = futures::executor::block_on(
            crate::Shell::default().run(crate::ShellArgs::new(command)),
        );
        let output = ran.map(|ran| ran.output).unwrap_or_default();
        crate::split_status(&output).0.to_owned()
    };
    let next = rest(note);
    assert_eq!(next, format!("{}\n", "é".repeat(500)));
    let ascii_note = text(ReadArgs::new(&path).lines(2, 1));
    let next = rest(ascii_note.lines().nth(1).unwrap_or_default());
    assert!(next.lines().all(|line| line.len() == 2000), "{next:.100}");
    assert_eq!(next.lines().collect::<String>(), "x".repeat(10_000));
    let three_note = text(ReadArgs::new(&path).lines(4, 1));
    let next = rest(three_note.lines().nth(1).unwrap_or_default());
    assert_eq!(next, format!("{}\n", "日".repeat(500)));
    std::fs::remove_dir_all(dir).ok();
}
