use super::Cut;

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
