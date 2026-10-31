use unicode_width::UnicodeWidthStr as _;

use std::path::PathBuf;

use rig_core::tool::{Scope, Tool, Touches};

use super::{
    Edit, EditArgs, OutputLimit, Read, ReadArgs, Search, SearchArgs, Shell, ShellArgs, Write,
    WriteArgs, excerpt, shorten,
};

/// A cut keeps whole lines from the end, within both limits, and says how
/// many it left out; a last line longer than the byte limit leaves nothing.
#[test]
fn a_cut_keeps_whole_lines_from_the_end() {
    let text = "one\ntwo\nthree";
    let roomy = OutputLimit {
        lines: 10,
        bytes: 100,
    };
    assert_eq!(roomy.tail(text), (text, 0));
    assert_eq!(roomy.tail(""), ("", 0));
    let two = OutputLimit {
        lines: 2,
        bytes: 100,
    };
    assert_eq!(two.tail(text), ("two\nthree", 1));
    let narrow = OutputLimit {
        lines: 10,
        bytes: 3,
    };
    assert_eq!(narrow.tail(text), ("", 3));
    assert_eq!(OutputLimit::default().tail("a\n"), ("a\n", 0));
}

/// A shortened text fits its width in terminal columns, the ellipsis
/// included, however wide its characters are.
#[test]
fn shorten_fits_the_width() {
    for text in [
        "plain ascii text",
        "日本語のテキストです",
        "mixed 漢字 and ascii",
    ] {
        for width in 0..24 {
            let short = shorten(text, width);
            assert!(short.width() <= width, "{short:?} in {width}");
            assert!(text.starts_with(short.trim_end_matches('…')));
        }
    }
    assert_eq!(shorten("日本語", 5), "日本…");
}

/// Each call's footprint names the path it reads or writes, resolved
/// against the tool's base; a search without a path reads the base, and a
/// shell call may change anything.
#[test]
fn each_call_names_what_it_touches() {
    let base = PathBuf::from("/nowhere/work");
    let at = |path: &str| Scope::Paths(vec![base.join(path)]);
    let read = Read {
        base: Some(base.clone()),
        confine: false,
        seen: crate::Seen::default(),
        ..Default::default()
    };
    let search = Search {
        base: Some(base.clone()),
        confine: false,
        ..Default::default()
    };
    let edit = Edit {
        base: Some(base.clone()),
        confine: false,
        seen: crate::Seen::default(),
        ..Default::default()
    };
    let write = Write {
        base: Some(base.clone()),
        confine: false,
        ..Default::default()
    };
    assert_eq!(
        read.footprint(&ReadArgs::new("src/../a.rs")),
        Touches::Reads(at("a.rs"))
    );
    assert_eq!(
        search.footprint(&SearchArgs::new("x").in_path("src")),
        Touches::Reads(at("src"))
    );
    assert_eq!(
        search.footprint(&SearchArgs::new("x")),
        Touches::Reads(Scope::Paths(vec![base.clone()]))
    );
    assert_eq!(
        edit.footprint(&EditArgs::new("a.rs", [])),
        Touches::Writes(at("a.rs"))
    );
    assert_eq!(
        write.footprint(&WriteArgs::new("/elsewhere/b.rs", "")),
        Touches::Writes(Scope::Paths(vec![PathBuf::from("/elsewhere/b.rs")]))
    );
    assert_eq!(
        Shell::default().footprint(&ShellArgs::new("ls")),
        Touches::Writes(Scope::Anything)
    );
}

/// A cut line whose span the cut left out says where it is: a deleted end
/// after a byte, and a span far into a long line with a command whose
/// window starts about 5,000 bytes before it.
#[test]
fn a_cut_names_a_span_it_left_out() {
    let line = "0".repeat(20_000);
    let deleted = excerpt(&line, Some(4000..4000), 300);
    assert_eq!(
        deleted.mark("the change", None),
        " [… cut at byte 300 of 20000; the change is after byte 4000]"
    );
    let far = excerpt(&line, Some(19_000..19_006), 300);
    let source = super::Source {
        path: "'/a.txt'",
        line: 7,
    };
    assert_eq!(
        far.mark("the match", Some(source)),
        " [… cut at byte 300 of 20000; the match is at bytes 19001-19006: in `shell`, `sed -n \
         '7p;7q' '/a.txt' | cut -b 14001-20000 | LC_ALL=C fold -b -w 2000` shows bytes \
         14001-20000]"
    );
}
