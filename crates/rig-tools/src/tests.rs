use unicode_width::UnicodeWidthStr as _;

use super::{Cut, shorten};

/// A cut keeps whole lines from the end, within both limits, and says how
/// many it left out; a last line longer than the byte limit leaves nothing.
#[test]
fn a_cut_keeps_whole_lines_from_the_end() {
    let text = "one\ntwo\nthree";
    let roomy = Cut {
        lines: 10,
        bytes: 100,
    };
    assert_eq!(roomy.tail(text), (text, 0));
    assert_eq!(roomy.tail(""), ("", 0));
    let two = Cut {
        lines: 2,
        bytes: 100,
    };
    assert_eq!(two.tail(text), ("two\nthree", 1));
    let narrow = Cut {
        lines: 10,
        bytes: 3,
    };
    assert_eq!(narrow.tail(text), ("", 3));
    assert_eq!(Cut::default().tail("a\n"), ("a\n", 0));
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
