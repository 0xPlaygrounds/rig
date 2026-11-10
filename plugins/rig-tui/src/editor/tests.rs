use super::*;

/// The text of each row past its two-column prompt.
fn texts(layout: &Layout) -> Vec<String> {
    let rows = layout.rows.iter().map(|row| row.to_string());
    rows.map(|row| row.chars().skip(2).collect()).collect()
}

#[test]
fn a_long_prompt_breaks_at_spaces_with_the_cursor_where_it_is_drawn() {
    let words = [
        "please",
        "read",
        "the",
        "wrapping",
        "code\tand",
        "explain",
        "👩\u{200d}💻",
        "why",
    ];
    let mut editor = Editor::default();
    editor.set(format!("{} ", words.join(" ")).repeat(3));
    for width in 12..40 {
        let text = editor.text().to_owned();
        let ends = text
            .grapheme_indices(true)
            .map(|(at, grapheme)| at + grapheme.len());
        for cursor in std::iter::once(0).chain(ends) {
            editor.set_cursor(cursor);
            let layout = editor.layout(width);
            let rows = texts(&layout);
            assert!(
                layout
                    .rows
                    .iter()
                    .all(|row| row.width() <= usize::from(width)),
                "{rows:?}"
            );
            // Every word is whole in one row: none is split.
            let pieces = rows.iter().flat_map(|row| row.split_whitespace());
            assert!(
                pieces.clone().all(|piece| words
                    .iter()
                    .any(|word| word.split('\t').any(|part| part == piece))),
                "{rows:?}"
            );
            // The cursor sits after the text before it: its row holds the
            // end of that text, at its column.
            let before = text.get(..cursor).unwrap_or_default().replace('\t', " ");
            let row = rows.get(layout.cursor_row).map_or("", String::as_str);
            let column = layout.cursor_column - 2;
            let drawn: String = row
                .graphemes(true)
                .scan(0, |at, grapheme| {
                    *at += grapheme.width();
                    Some((*at, grapheme))
                })
                .take_while(|(at, _)| *at <= column)
                .map(|(_, grapheme)| grapheme)
                .collect();
            assert!(
                before.trim_end().ends_with(drawn.trim()),
                "{cursor} at {width}: {drawn:?} in {rows:?}"
            );
            // Past a row's end only in the spaces its break dropped.
            let after_row = drawn.width() == row.width() && before.ends_with(' ');
            assert!(
                drawn.width() == column || after_row,
                "{cursor} at {width}: {rows:?}"
            );
        }
    }
}

#[test]
fn a_narrow_key_row_keeps_esc_and_says_it_dropped_some() {
    let running = hint(true, false);
    let all = fit_keys(running, 500);
    assert!(!all.contains('…'), "{all}");
    for width in 30..all.width() {
        let row = fit_keys(running, width);
        assert!(row.width() <= width, "{row}");
        assert!(
            row.contains("Esc stops") && row.trim_end().ends_with('…'),
            "{row}"
        );
    }
}
