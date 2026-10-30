use super::*;

/// `count` lines made from `line`, each `#` numbered: a text long enough
/// to be kept.
fn long(line: &str, count: usize) -> String {
    (0..count)
        .map(|n| line.replace('#', &n.to_string()))
        .collect()
}

/// `text`'s lines colored from its first, by the grammar of `path`.
fn whole(text: &str, path: &str) -> Vec<Vec<Span<'static>>> {
    let syntax = syntax(Language::File(path));
    assert!(syntax.is_some(), "{path} has no grammar");
    let Some(syntax) = syntax else {
        return Vec::new();
    };
    let mut parser = HighlightLines::new(syntax, &THEME);
    let lines = LinesWithEndings::from(text).map(|line| spans(&mut parser, line));
    lines.map(Option::unwrap_or_default).collect()
}

/// Each line of `code`, colored in the order of `order`, then checked
/// against the whole text colored from its first line.
fn colored_as_whole(code: &Code, path: &str, order: impl IntoIterator<Item = usize>) {
    let whole = whole(&code.0.text, path);
    for index in order {
        assert_eq!(code.spans(index).as_ref(), whole.get(index), "line {index}");
    }
    for (index, line) in whole.iter().enumerate() {
        assert_eq!(code.spans(index).as_ref(), Some(line), "line {index}");
    }
}

/// How many of `code`'s lines are colored, past those kept apart.
fn colored(code: &Code) -> usize {
    code.colors().lines.len()
}

/// The texts of the kept codes, oldest first.
fn kept() -> Vec<String> {
    KEPT.with_borrow(|kept| kept.iter().map(|code| code.0.text.clone()).collect())
}

#[test]
fn code_is_drawn_in_the_terminals_palette_and_plain_text_is_not_colored() {
    let code = Code::new("fn main() {\n    let x = 1;\n}\n", Language::File("a.rs"));
    assert_eq!(code.len(), 3);
    let lines: Vec<_> = (0..code.len()).filter_map(|at| code.spans(at)).collect();
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
    let line = code.line(0, vec![Span::raw("> ")]);
    assert_eq!(line.to_string(), "> fn main() {");
    assert_eq!(line.colored().to_string(), "> fn main() {");
    let notes = Code::new("hello", Language::File("notes.txt"));
    assert!(!notes.has_grammar());
    assert_eq!(notes.spans(0), None);
    assert!(Code::new("ls -la", Language::Named("bash")).has_grammar());
}

#[test]
fn a_block_colored_in_any_order_is_colored_as_the_whole_of_it_is() {
    KEPT.with_borrow_mut(Vec::clear);
    // A string and a comment that span lines, CRLF lines with tabs and
    // wide text, and a last line with no end.
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
            format!("/*\n{}*/\nfn main() {{}}", long("  comment # ✓\n", 40)),
        ),
        ("a.py", long("\tx_# = \"é\"  # note\r\n", 300)),
    ];
    for (path, text) in cases {
        let last = text.lines().count() - 1;
        colored_as_whole(
            &Code::made(&text, syntax(Language::File(path))),
            path,
            0..=last,
        );
        let code = Code::made(&text, syntax(Language::File(path)));
        colored_as_whole(&code, path, [last, 0, last / 2]);
    }
}

#[test]
fn only_the_lines_drawn_and_those_above_them_are_colored_each_once() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = long("x_# = 1\n", 1000);
    let (code, other) = (
        Code::new(&text, Language::File("a.py")),
        Code::new(&format!("# other\n{text}"), Language::File("a.py")),
    );
    assert!(code.spans(500).is_some());
    assert_eq!(colored(&code), 501);
    assert_eq!(colored(&other), 0);
    // A line colored is not colored again.
    let marked = vec![Span::raw("kept")];
    if let Some(first) = code.colors().lines.first_mut() {
        *first = marked.clone();
    }
    assert!(code.spans(20).is_some());
    assert_eq!(code.spans(0), Some(marked));
    assert_eq!(colored(&code), 501);
}

#[test]
fn a_growing_text_goes_on_from_the_colored_lines_of_the_one_it_extends() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = format!(
        "s = '''\n{}'''\nx = 1\n",
        long("\tin the string # ✓\r\n", 400)
    );
    let mut before = 0;
    let cuts = (KEPT_FROM..text.len()).step_by(997).chain([text.len()]);
    for cut in cuts.filter(|cut| text.is_char_boundary(*cut)) {
        let piece = text.get(..cut).unwrap_or_default();
        let code = Code::new(piece, Language::File("a.py"));
        assert!(colored(&code) >= before, "nothing taken at {cut}");
        // The end on screen, as a file being written streams in.
        let last = code.len().saturating_sub(1);
        colored_as_whole(&code, "a.py", [last]);
        before = colored(&code);
        assert_eq!(kept().last().map(String::as_str), Some(piece));
    }
    // Cut between a line's CR and its LF, then on.
    let cr = text.get(KEPT_FROM..).and_then(|rest| rest.find("\r\n"));
    let cr = cr.map_or(0, |at| KEPT_FROM + at);
    for end in [cr, cr + 1, cr + 2] {
        let code = Code::new(text.get(..end).unwrap_or_default(), Language::File("a.py"));
        colored_as_whole(&code, "a.py", [code.len().saturating_sub(1)]);
    }
}

#[test]
fn an_unfinished_last_line_is_not_taken_by_a_text_that_ends_otherwise() {
    KEPT.with_borrow_mut(Vec::clear);
    let lines = long("x_# = 1\n", 600);
    let open = Code::new(&format!("{lines}s = '''abc"), Language::File("a.py"));
    colored_as_whole(&open, "a.py", [open.len() - 1]);
    let closed = Code::new(&format!("{lines}# abc"), Language::File("a.py"));
    assert_eq!(colored(&closed), 600);
    colored_as_whole(&closed, "a.py", [closed.len() - 1]);
}

#[test]
fn a_rewritten_text_or_another_language_is_colored_afresh_and_the_same_is_shared() {
    KEPT.with_borrow_mut(Vec::clear);
    let text = long("x_# = 1  # one\n", 400);
    let code = Code::new(&text, Language::File("a.py"));
    colored_as_whole(&code, "a.py", [399]);
    let rewritten = Code::new(&format!("\"\"\"{text}"), Language::File("a.py"));
    assert_eq!(colored(&rewritten), 0);
    colored_as_whole(&rewritten, "a.py", [400]);
    let ruby = Code::new(&format!("{text}y = 2\n"), Language::File("a.rb"));
    assert_eq!(colored(&ruby), 0);
    colored_as_whole(&ruby, "a.rb", [400]);
    let again = Code::new(&text, Language::File("a.py"));
    assert!(Arc::ptr_eq(&again.0, &code.0));
    assert_eq!(colored(&again), 400);
}

#[test]
fn texts_growing_in_turn_each_go_on_from_their_own() {
    KEPT.with_borrow_mut(Vec::clear);
    let one = format!("s = '''\n{}", long("in a string #\n", 400));
    let two = long("t_# = 2\n", 600);
    let mut before = [0, 0];
    for cut in (KEPT_FROM..one.len().min(two.len())).step_by(1499) {
        for (text, before) in [&one, &two].into_iter().zip(&mut before) {
            let code = Code::new(text.get(..cut).unwrap_or_default(), Language::File("a.py"));
            assert!(colored(&code) >= *before, "nothing taken at {cut}");
            colored_as_whole(&code, "a.py", [code.len() - 1]);
            *before = colored(&code);
        }
    }
    assert_eq!(kept().len(), 2, "{:?}", kept());
}

#[test]
fn the_kept_codes_stay_within_their_bounds_oldest_dropped_first() {
    KEPT.with_borrow_mut(Vec::clear);
    let python = syntax(Language::File("a.py"));
    assert!(python.is_some(), "no grammar for Python");
    // Stand-ins, each with a text of half the bound, then one past it.
    let stand_in = |name: &str, bytes: usize| {
        let text = format!("{name}\n{}", "#".repeat(bytes));
        Code::made(&text, python)
    };
    KEPT.with_borrow_mut(|kept| {
        kept.extend(["one", "two", "three"].map(|name| stand_in(name, KEPT_BYTES / 2)));
    });
    let text = long("x_# = 1\n", 600);
    Code::new(&text, Language::File("a.py"));
    let starts = || {
        kept()
            .iter()
            .map(|text| text.chars().take(5).collect::<String>())
            .collect::<Vec<_>>()
    };
    assert_eq!(starts(), ["three", "x_0 ="]);
    // A code that alone passes the bound is not kept, and leaves the
    // others be.
    let big = format!("{}\n", "x".repeat(KEPT_BYTES));
    Code::new(&big, Language::File("a.py"));
    assert_eq!(starts(), ["three", "x_0 ="]);
    for n in 0..KEPT_COUNT + 3 {
        Code::new(&long(&format!("v{n}_# = 1\n"), 500), Language::File("a.py"));
    }
    assert_eq!(kept().len(), KEPT_COUNT);
}

#[test]
fn a_transcript_colors_only_the_code_on_screen_and_draws_it_colored() {
    use crate::Highlight;
    use crate::transcript::{Part, Scroll, Transcript};
    use crate::wrap::wrap;
    KEPT.with_borrow_mut(Vec::clear);
    let line = "x_# = 'a string long enough to wrap at the width of the pane'\n";
    let top = Code::new(&long(line, 300), Language::File("a.py"));
    let bottom = Code::new(&long(&format!("y{line}"), 50), Language::File("a.py"));
    fn lines(code: &Code) -> impl Iterator<Item = LookLine> + '_ {
        (0..code.len()).map(|at| code.line(at, vec![Span::raw("  ")]))
    }
    let all: Vec<LookLine> = lines(&top).chain(lines(&bottom)).collect();
    let width = 40;
    let parts = [Part::rows(&all, &[], width)];
    let lit = [Highlight {
        text: "'a string".to_owned(),
        style: Style::new().reversed(),
        match_case: false,
        spans: Vec::new(),
    }];
    for highlights in [&[][..], &lit] {
        // An odd height, so the screen starts on a line's second row.
        let visible =
            Transcript::default().visible(&parts, (40, 11), &mut Scroll::default(), highlights);
        assert_eq!(colored(&top), 0);
        assert_eq!(colored(&bottom), 50);
        let rows: Vec<Line<'static>> = lines(&bottom)
            .flat_map(|line| wrap(&line.colored(), width))
            .map(|(row, _)| row)
            .collect();
        let shown = rows
            .get(rows.len() - visible.rows.len()..)
            .unwrap_or_default();
        let text =
            |rows: &[Line<'static>]| rows.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(text(&visible.rows), text(shown));
        let spans = visible.rows.iter().flat_map(|row| &row.spans);
        assert!(spans.clone().any(|span| span.style.fg.is_some()));
        if highlights.is_empty() {
            assert_eq!(visible.rows, shown);
        } else {
            // Each match is lit over its colors, and only the match.
            let reversed = |span: &&Span<'_>| {
                span.style
                    .add_modifier
                    .contains(ratatui::style::Modifier::REVERSED)
            };
            let mut matched = 0;
            for row in &visible.rows {
                let lit: Vec<&Span<'_>> = row.spans.iter().filter(reversed).collect();
                let text: String = lit.iter().map(|span| &*span.content).collect();
                assert!(text.is_empty() || text == "'a string", "{row:?}");
                assert!(lit.iter().all(|span| span.style.fg.is_some()), "{row:?}");
                matched += usize::from(!text.is_empty());
            }
            assert!(matched > 0);
        }
    }
}

#[test]
fn a_code_the_grammar_failed_on_draws_plain_and_hands_nothing_on() {
    KEPT.with_borrow_mut(Vec::clear);
    let lines = long("x_# = 1\n", 600);
    let code = Code::new(&format!("{lines}y = "), Language::File("a.py"));
    colored_as_whole(&code, "a.py", [code.len() - 1]);
    code.colors().fail();
    assert_eq!(code.spans(0), None);
    let grown = Code::new(&format!("{lines}y = 2\n"), Language::File("a.py"));
    assert_eq!(colored(&grown), 0);
    colored_as_whole(&grown, "a.py", [grown.len() - 1]);
}
