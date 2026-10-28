use rig_core::tool::{Hunk, HunkLine, Lines};

use super::*;

const EDIT: &str = "Edited a.rs: line 10.
--- a.rs
+++ a.rs
@@ -9,3 +9,3 @@
 fn main() {
-    let last = end + 1;
+    let last = end;
 }
";

#[test]
fn each_line_has_its_numbers_and_its_band() {
    let theme = Theme::on((0, 0, 0), true);
    let Some(lines) = unified_lines(EDIT, &theme) else {
        return;
    };
    let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts.get(4..).unwrap_or_default(),
        [
            "     9  9   fn main() {",
            "    10    -     let last = end + 1;",
            "       10 +     let last = end;",
            "    11 11   }",
        ]
    );
    // a.rs is Rust: its code is in the terminal's palette.
    let same = lines
        .get(4)
        .map(|line| line.spans.clone())
        .unwrap_or_default();
    assert!(
        same.iter().skip(1).any(|span| span.style.fg.is_some()),
        "{same:?}"
    );
    let bands: Vec<_> = lines.iter().map(|line| line.style.bg).collect();
    assert_eq!(
        bands.get(5..7),
        Some(&[Some(theme.removed), Some(theme.added)][..])
    );
}

#[test]
fn only_the_words_a_line_changed_are_marked() {
    let theme = Theme::on((0, 0, 0), true);
    let lines = unified_lines(EDIT, &theme).unwrap_or_default();
    let marked = |at: usize, shade| {
        let spans = lines
            .get(at)
            .map(|line| line.spans.clone())
            .unwrap_or_default();
        let marked = spans.iter().filter(|span| span.style.bg == Some(shade));
        marked
            .map(|span| span.content.to_string())
            .collect::<String>()
    };
    assert_eq!(marked(5, theme.removed_word), "end + 1;");
    assert_eq!(marked(6, theme.added_word), "end;");
}

#[test]
fn a_removed_line_like_a_file_header_stays_a_removed_line() {
    let diff = "@@ -1,2 +1,1 @@\n--- not a file\n kept\n";
    let lines = unified_lines(diff, &Theme::default()).unwrap_or_default();
    let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts.get(1).map(String::as_str),
        Some("    1   - -- not a file")
    );
}

#[test]
fn a_changes_lines_are_its_hunks_in_its_files_colors() {
    let change = FileChange::modified("/p/a.rs", 1, 1).with_hunks(vec![Hunk {
        old: Lines { start: 9, count: 2 },
        new: Lines { start: 9, count: 2 },
        lines: vec![
            HunkLine::Same("fn main() {\n".into()),
            HunkLine::Removed("    let last = end + 1;\n".into()),
            HunkLine::Added("    let last = end;\n".into()),
        ],
    }]);
    let lines = change_lines(&change, &Theme::on((0, 0, 0), true));
    let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert!(
        texts
            .iter()
            .all(|text| !text.contains("---") && !text.contains("+++")),
        "{texts:?}"
    );
    assert_eq!(
        texts.get(1..).unwrap_or_default(),
        [
            "     9  9   fn main() {",
            "    10    -     let last = end + 1;",
            "       10 +     let last = end;",
        ]
    );
    // a.rs is Rust: its code is in the terminal's palette.
    let same = lines
        .get(1)
        .map(|line| line.spans.clone())
        .unwrap_or_default();
    assert!(
        same.iter().skip(1).any(|span| span.style.fg.is_some()),
        "{same:?}"
    );
}

#[test]
fn a_hunk_cut_short_ends_at_its_first_line_that_cannot_be_its() {
    let ends = [
        ("AFTER the diff", "    AFTER the diff"),
        ("\tindented", "        indented"),
        (
            "[… 40 lines cut (lines 13-52) …]",
            "    [… 40 lines cut (lines 13-52) …]",
        ),
        ("[exit code 0]", "    [exit code 0]"),
    ];
    for (end, drawn) in ends {
        let output = format!(
            "@@ -1,19 +1,57 @@\n use std::fs;\n{end}\n+literal\n@@ -30,1 +60,1 @@\n kept\n"
        );
        let lines = unified_lines(&output, &Theme::default()).unwrap_or_default();
        let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
        assert_eq!(
            texts.get(2..).unwrap_or_default(),
            [
                drawn,
                "    +literal",
                "    @@ -30,1 +60,1 @@",
                "    30 60   kept"
            ],
            "{end}"
        );
    }
}

#[test]
fn a_line_fits_its_hunk_only_while_its_side_has_lines_left() {
    let hunks = [
        ("@@ -1,3 +1,2 @@\n-gone\n same\n+one\n+two\n", "    +two"),
        ("@@ -1,1 +1,2 @@\n-old\n-extra\n", "    -extra"),
        ("@@ -1,1 +1,2 @@\n-old\n context\n", "     context"),
    ];
    for (hunk, last) in hunks {
        let lines = unified_lines(hunk, &Theme::default()).unwrap_or_default();
        let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
        assert_eq!(texts.last().map(String::as_str), Some(last), "{hunk}");
    }
}

#[test]
fn a_blank_line_is_context_only_when_its_hunk_goes_on_after_it() {
    // A blank line as `git -c diff.suppressBlankEmpty=true diff` writes it,
    // and one echoed after a hunk cut short.
    let diffs = [
        (
            "@@ -1,3 +1,3 @@\n first\n\n-old\n+new\n",
            &[
                "    1 1   first",
                "    2 2   ",
                "    3   - old",
                "      3 + new",
            ][..],
        ),
        (
            "@@ -1 +1,3 @@\n\n+x\n+y\n",
            &["    1 1   ", "      2 + x", "      3 + y"][..],
        ),
        (
            "@@ -1,2 +1 @@\n\n-gone\n",
            &["    1 1   ", "    2   - gone"][..],
        ),
        (
            "@@ -1,3 +1,3 @@\n\n\n\nafter\n",
            &["    1 1   ", "    2 2   ", "    3 3   ", "    after"][..],
        ),
        (
            "@@ -1,2 +1,2 @@\n first\n\n\nafter\n",
            &["    1 1   first", "    2 2   ", "    ", "    after"][..],
        ),
        (
            "@@ -1,19 +1,57 @@\n first\n\nafter blank\n",
            &["    1 1   first", "    ", "    after blank"][..],
        ),
    ];
    for (diff, drawn) in diffs {
        let lines = unified_lines(diff, &Theme::default()).unwrap_or_default();
        let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
        assert_eq!(texts.get(1..).unwrap_or_default(), drawn, "{diff}");
    }
}

#[test]
fn a_line_number_past_the_largest_stays_there() {
    let diff = format!("@@ -{},2 +1 @@\n-x\n-y\n", usize::MAX);
    let lines = unified_lines(&diff, &Theme::default()).unwrap_or_default();
    let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
    let largest = usize::MAX.to_string();
    let gutter = format!("    {largest} {}", " ".repeat(largest.len()));
    assert_eq!(
        texts.get(1..).unwrap_or_default(),
        [format!("{gutter} - x"), format!("{gutter} - y")]
    );
}

#[test]
fn a_missing_newline_note_keeps_its_hunk_open() {
    let diff = "@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n";
    let lines = unified_lines(diff, &Theme::default()).unwrap_or_default();
    let texts: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts.get(1..).unwrap_or_default(),
        [
            "    1   - old",
            "    \\ No newline at end of file",
            "      1 + new"
        ]
    );
}
