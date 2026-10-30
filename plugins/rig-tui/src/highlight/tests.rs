use super::*;

#[test]
fn code_is_drawn_in_the_terminals_palette_and_plain_text_is_not_highlighted() {
    let Some(lines) = highlight("fn main() {\n    let x = 1;\n}\n", Language::File("a.rs")) else {
        return;
    };
    assert_eq!(lines.len(), 3);
    let colors: Vec<Color> = lines
        .iter()
        .flatten()
        .filter_map(|span| span.style.fg)
        .collect();
    assert!(!colors.is_empty());
    assert!(
        colors
            .iter()
            .all(|color| matches!(color, Color::Indexed(_)))
    );
    let text: String = lines
        .first()
        .into_iter()
        .flatten()
        .map(|span| span.content.as_ref())
        .collect();
    assert_eq!(text, "fn main() {");
    assert!(highlight("hello", Language::File("notes.txt")).is_none());
    assert!(highlight("ls -la", Language::Named("bash")).is_some());
}

/// `count` lines of `language`'s code made from `line`, numbered: a text
/// long enough to be kept.
fn long(line: &str, count: usize) -> String {
    (0..count)
        .map(|n| line.replace('#', &n.to_string()))
        .collect()
}

/// What the kept texts are, oldest first.
fn kept() -> Vec<String> {
    KEPT.with_borrow(|kept| kept.iter().map(|text| text.text.clone()).collect())
}

/// Highlights `code` as `path`'s language, and checks it against a fresh
/// highlight of the whole text.
fn same_as_fresh(code: &str, path: &str) {
    let fresh = syntax(Language::File(path)).and_then(|syntax| fresh(code, syntax));
    assert!(fresh.is_some(), "{path} has no grammar");
    assert_eq!(highlight(code, Language::File(path)), fresh, "{code:?}");
}

#[test]
fn a_growing_text_is_highlighted_as_the_whole_text_is() {
    KEPT.with_borrow_mut(Vec::clear);
    // A string and a comment opened in one piece and closed in a later
    // one, and CRLF lines with tabs and wide text.
    let cases = [
        (
            "a.py",
            format!(
                "doc = \"\"\"\n{}\"\"\"\nx = 1\n",
                long("still in the string #\n", 300)
            ),
        ),
        (
            "a.rs",
            format!("/*\n{}*/\nfn main() {{}}\n", long("  comment # ✓\n", 400)),
        ),
        ("a.py", long("\tx_# = \"é\"  # note\r\n", 300)),
    ];
    for (path, text) in cases {
        let cuts = (1..text.len()).step_by(997).chain([text.len()]);
        for cut in cuts.filter(|cut| text.is_char_boundary(*cut)) {
            let piece = text.get(..cut).unwrap_or_default();
            same_as_fresh(piece, path);
            if piece.len() >= KEPT_FROM {
                let kept = kept();
                let last = kept.last().map(String::as_str).unwrap_or_default();
                assert!(!last.is_empty() && piece.starts_with(last), "{kept:?}");
            }
        }
    }
}

#[test]
fn a_line_split_between_its_cr_and_its_lf_goes_on_right() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = long("\tx_# = \"é\"  # note\r\n", 400);
    let cr = text
        .get(KEPT_FROM..)
        .and_then(|rest| rest.find("\r\n"))
        .map_or(0, |at| KEPT_FROM + at);
    for end in [cr, cr + 1, cr + 2, text.len()] {
        same_as_fresh(text.get(..end).unwrap_or_default(), "a.py");
    }
}

#[test]
fn the_lines_kept_are_not_highlighted_again() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = long("x_# = 1\n", 600);
    same_as_fresh(&text, "a.py");
    let marked = vec![Span::raw("kept")];
    KEPT.with_borrow_mut(|kept| {
        if let Some(first) = kept.last_mut().and_then(|text| text.lines.first_mut()) {
            *first = marked.clone();
        }
    });
    let grown = highlight(&format!("{text}y = 2\n"), Language::File("a.py")).unwrap_or_default();
    assert_eq!(grown.first(), Some(&marked));
    assert_eq!(grown.len(), 601);
}

#[test]
fn a_rewritten_text_or_another_language_is_highlighted_afresh() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = long("x_# = 1  # one\n", 400);
    same_as_fresh(&text, "a.py");
    same_as_fresh(&format!("\"\"\"{text}"), "a.py");
    same_as_fresh(&format!("{text}y = 2\n"), "a.rb");
    same_as_fresh(&format!("{text}y = 2\n"), "a.py");
}

#[test]
fn texts_growing_in_turn_each_go_on_from_their_own() {
    KEPT.with_borrow_mut(Vec::clear);
    let one = format!("s = '''\n{}", long("in a string #\n", 400));
    let two = long("t_# = 2\n", 600);
    for cut in (KEPT_FROM..one.len().min(two.len())).step_by(1499) {
        same_as_fresh(one.get(..cut).unwrap_or_default(), "a.py");
        same_as_fresh(two.get(..cut).unwrap_or_default(), "a.py");
    }
    assert_eq!(kept().len(), 2, "{:?}", kept());
}

#[test]
fn the_kept_texts_stay_within_their_bytes_oldest_dropped_first() {
    KEPT.with_borrow_mut(Vec::clear);
    let python = syntax(Language::File("a.py"));
    assert!(python.is_some(), "no grammar for Python");
    let Some(syntax) = python else {
        return;
    };
    // Stand-ins said to take half the bound each, then one past it.
    let stand_in = |text: &str, bytes| Kept {
        syntax,
        text: text.to_owned(),
        lines: Vec::new(),
        state: HighlightLines::new(syntax, &THEME).state(),
        bytes,
    };
    KEPT.with_borrow_mut(|kept| {
        kept.extend(["one\n", "two\n", "three\n"].map(|text| stand_in(text, KEPT_BYTES / 2)));
    });
    let text = long("x_# = 1\n", 600);
    same_as_fresh(&text, "a.py");
    assert_eq!(kept(), ["three\n", text.as_str()]);
    KEPT.with_borrow_mut(|kept| kept.insert(0, stand_in("big\n", KEPT_BYTES + 1)));
    same_as_fresh(&format!("{text}y = 2\n"), "a.py");
    assert_eq!(kept().len(), 2);
    assert!(!kept().contains(&"big\n".to_owned()));
}

#[test]
fn the_kept_texts_stay_within_their_count() {
    KEPT.with_borrow_mut(Vec::clear);
    for n in 0..KEPT_COUNT + 3 {
        same_as_fresh(&long(&format!("v{n}_# = 1\n"), 500), "a.py");
    }
    assert_eq!(kept().len(), KEPT_COUNT);
    let bytes: usize = KEPT.with_borrow(|kept| kept.iter().map(|text| text.bytes).sum());
    assert!(bytes <= KEPT_BYTES, "{bytes}");
}
