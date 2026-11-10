use super::*;

/// The text of each row past its two-column prompt.
fn texts(layout: &Layout) -> Vec<String> {
    let rows = layout.rows.iter().map(|row| row.to_string());
    rows.map(|row| row.chars().skip(2).collect()).collect()
}

/// `text` without its blanks.
fn solid(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .collect()
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
        "abcdefghij",
        "e\u{301}",
    ];
    let mut editor = Editor::default();
    editor.set(format!("{} ", words.join(" ")).repeat(3));
    let text = editor.text().to_owned();
    let ends = text
        .grapheme_indices(true)
        .map(|(at, grapheme)| at + grapheme.len());
    let cursors: Vec<usize> = std::iter::once(0).chain(ends).collect();
    for width in 12..40 {
        for &cursor in &cursors {
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
            // The caret is inside the input, after exactly the text before
            // the cursor: the rows above it, then its row up to its column.
            assert!(
                layout.cursor_column < usize::from(width),
                "{cursor} at {width}: {rows:?}"
            );
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
            let above = rows.get(..layout.cursor_row).unwrap_or_default().concat();
            let before = text.get(..cursor).unwrap_or_default();
            assert_eq!(
                solid(&above) + &solid(&drawn),
                solid(before),
                "{cursor} at {width}: {rows:?}"
            );
            // Past a row's text only in the spaces its break dropped.
            let after_row = drawn.width() == row.width() && before.ends_with(' ');
            assert!(
                drawn.width() == column || after_row,
                "{cursor} at {width}: {rows:?}"
            );
        }
    }
}

#[test]
fn a_cursor_in_the_space_after_a_full_row_stays_inside_the_input() {
    let mut editor = Editor::default();
    editor.set("abcdefghij xyz");
    editor.set_cursor(10);
    let layout = editor.layout(12);
    assert!(layout.cursor_column < 12, "{}", layout.cursor_column);
    editor.set("abcdefghij\nxyz");
    editor.set_cursor(10);
    let layout = editor.layout(12);
    assert!(layout.cursor_column < 12, "{}", layout.cursor_column);
}

#[test]
fn the_cursor_never_lands_inside_a_grapheme() {
    let mut editor = Editor::default();
    editor.set("e\u{301} 👩\u{200d}💻x");
    editor.home();
    editor.word_right();
    assert_eq!(editor.cursor(), "e\u{301}".len());
    editor.backspace();
    assert_eq!(editor.text(), " 👩\u{200d}💻x");
    editor.set_cursor(" 👩".len());
    assert_eq!(editor.cursor(), 1);
    editor.delete();
    assert_eq!(editor.text(), " x");
    // Deleting between two pieces that join into one grapheme leaves the
    // cursor after it.
    let joins: [(&str, usize, fn(&mut Editor)); 6] = [
        ("e\n\u{301}", 2, Editor::backspace),
        ("e\n\u{301}", 1, Editor::delete),
        ("🇺x🇸", "🇺x".len(), Editor::backspace),
        ("🇺x🇸", "🇺".len(), Editor::delete),
        ("👩\u{200d}x💻", "👩\u{200d}x".len(), Editor::backspace),
        ("e\nab\u{301}", 2, Editor::delete_word_after),
    ];
    for (text, at, edit) in joins {
        editor.set(text);
        editor.set_cursor(at);
        edit(&mut editor);
        let edited = editor.text();
        let ends = edited
            .grapheme_indices(true)
            .map(|(at, grapheme)| at + grapheme.len());
        let boundaries: Vec<usize> = std::iter::once(0).chain(ends).collect();
        assert!(
            boundaries.contains(&editor.cursor()),
            "{text:?} -> {edited:?} at {}",
            editor.cursor()
        );
    }
    editor.set("a\r\nb\rc");
    assert_eq!(editor.text(), "a\nb\nc");
    editor.set("\u{301}");
    editor.home();
    editor.insert("e");
    assert_eq!(editor.cursor(), "e\u{301}".len());
}

#[test]
fn a_narrow_key_row_keeps_what_matters_most_and_says_it_dropped_some() {
    for keys in [hint(true, false), hint(false, false)] {
        let all = fit_keys(keys, 500);
        let names: Vec<&str> = keys.iter().map(|(name, _)| *name).collect();
        assert_eq!(all, format!(" {} ", names.join(" · ")));
        let most = keys.iter().map(|(_, keep)| *keep).max().unwrap_or_default();
        let top = keys.iter().find(|(_, keep)| *keep == most);
        let top = top.map_or("", |(name, _)| *name);
        for width in format!(" {top} · … ").width()..all.width() {
            let row = fit_keys(keys, width);
            assert!(row.width() <= width, "{row}");
            assert!(row.trim_end().ends_with('…'), "{row}");
            // Nothing dropped matters more than something kept, and what is
            // kept keeps its order.
            let kept: Vec<&str> = row
                .trim()
                .split(" · ")
                .filter(|item| *item != "…")
                .collect();
            let rank = |item: &&str| {
                keys.iter()
                    .find(|(name, _)| name == item)
                    .map(|(_, keep)| *keep)
            };
            let lowest = kept.iter().filter_map(rank).min().unwrap_or_default();
            for (name, keep) in keys {
                assert!(
                    *keep < lowest || kept.contains(name),
                    "{name} at {width}: {row}"
                );
            }
            let order: Vec<&str> = names
                .iter()
                .copied()
                .filter(|name| kept.contains(name))
                .collect();
            assert_eq!(kept, order, "{width}");
        }
    }
    assert!(fit_keys(hint(true, false), 15).contains("Esc stops"));
}

#[test]
fn a_history_entry_with_a_carriage_return_is_walked_like_any_other() {
    let history = ["old".to_owned(), "a\r\nb".to_owned(), "new".to_owned()];
    let mut editor = Editor::default();
    editor.set("draft");
    editor.home();
    editor.up(&history);
    editor.up(&history);
    assert!(editor.recalled());
    editor.up(&history);
    assert_eq!(editor.text(), "old");
    editor.down(&history);
    editor.down(&history);
    editor.down(&history);
    assert_eq!(editor.text(), "draft");
}
