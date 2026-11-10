use unicode_width::UnicodeWidthStr;

use super::*;

/// A terminal `width` columns wide, without colors.
fn at(width: usize) -> Terminal {
    Terminal {
        width: Some(width),
        color: false,
    }
}

/// `text` without its SGR escapes.
fn visible(text: &str) -> String {
    let mut shown = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            }
            c => shown.push(c),
        }
    }
    shown
}

/// Words of 1 to 12 letters from a fixed seed, at least `chars` long in
/// all, one space between each.
fn paragraph(seed: u64, chars: usize) -> String {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut words: Vec<String> = Vec::new();
    while words.iter().map(|word| word.len() + 1).sum::<usize>() < chars {
        let len = 1 + usize::try_from(next() % 12).unwrap_or(0);
        let letters =
            (0..len).filter_map(|_| char::from_u32(u32::try_from(97 + next() % 26).ok()?));
        words.push(letters.collect());
    }
    words.join(" ")
}

#[test]
fn a_paragraph_breaks_only_at_spaces_and_fits_the_width() {
    for seed in 0..20 {
        let text = paragraph(seed, 300);
        for width in [20, 50, 80, 140] {
            let drawn = visible(&at(width).markdown(&text));
            let rows: Vec<&str> = drawn.lines().collect();
            assert!(
                rows.iter().all(|row| row.width() <= width),
                "{width}: {rows:#?}"
            );
            // Each break replaced exactly one space, so rejoining gives the text.
            assert_eq!(rows.join(" "), text, "{width}");
        }
    }
}

#[test]
fn code_lines_are_never_wrapped_and_a_token_wider_than_the_row_ends() {
    let long = format!(
        "let config = load(\"{}\").expect(\"loads\");",
        "a/".repeat(80)
    );
    let url = format!("https://example.com/{}", "segment-".repeat(30));
    let text = format!("Run it:\n\n```rust\n{long}\n\tindented();\n```\n\n    {long}\n\nSee {url}");
    for width in [1, 20, 80] {
        let drawn = visible(&at(width).markdown(&text));
        let rows: Vec<&str> = drawn.lines().collect();
        // The fenced and the indented block, each line whole.
        let whole = rows.iter().filter(|row| row.trim_start() == long).count();
        assert_eq!(whole, 2, "{width}: {rows:#?}");
        assert!(rows.iter().any(|row| row.trim_start() == "indented();"));
        // The URL is broken across rows, every character kept.
        assert!(drawn.replace('\n', "").contains(&url), "{width}");
        let code = |row: &str| [long.as_str(), "indented();"].contains(&row.trim_start());
        let wide: Vec<_> = rows
            .iter()
            .filter(|row| row.width() > width && !code(row))
            .collect();
        assert!(wide.is_empty(), "{width}: {wide:#?}");
    }
}

#[test]
fn styles_are_escapes_and_no_color_keeps_only_the_modifiers() {
    let colored = Terminal {
        width: Some(80),
        color: true,
    };
    let text = "# Title\n\nSome **bold**, a [link](https://example.com) and `code`.\x1b[2J";
    let drawn = colored.markdown(text);
    assert!(drawn.contains("\x1b[1m"), "{drawn:?}");
    assert!(drawn.contains(";36m"), "{drawn:?}");
    // The model's own escapes are not passed to the terminal.
    assert!(!drawn.contains("\x1b[2J"), "{drawn:?}");
    let plain = at(80).markdown(text);
    assert!(
        plain.contains("\x1b[1m") && !plain.contains("36m"),
        "{plain:?}"
    );
    assert_eq!(visible(&drawn), visible(&plain));
    assert!(visible(&plain).contains("`code`"), "{plain:?}");
}

#[test]
fn a_call_shows_its_summary_and_never_its_other_arguments() {
    use rig_harness::rig_core::message::{CallId, LocalCallId, ToolFunction, ToolName};
    let shown = |name: &str, arguments: serde_json::Value| match ToolName::new(name) {
        Ok(name) => {
            let function = ToolFunction::new(name, arguments);
            let call = ToolCall::new(CallId::Local(LocalCallId::new()), function);
            visible(&at(80).call(&call))
        }
        Err(_) => String::new(),
    };
    let read = serde_json::json!({ "path": "src/lib.rs", "limit": 5 });
    assert_eq!(shown("read", read), "· read src/lib.rs");
    let shell = serde_json::json!({ "command": "cargo test\necho done", "timeout": 9 });
    assert_eq!(shown("shell", shell), "· $ cargo test\n    echo done");
    let other = serde_json::json!({ "url": "https://x", "token": "secret" });
    assert_eq!(shown("fetch", other), "· fetch");
    let long = serde_json::json!({ "command": "x".repeat(200) });
    assert!(shown("shell", long).lines().all(|row| row.width() <= 80));
}

#[test]
fn a_list_item_that_starts_with_code_keeps_its_marker() {
    for (text, marker) in [
        ("1. ```\n   cargo build\n   ```\n2. Run it.\n", "1. "),
        ("- ```\n  x\n  ```\n", "• "),
    ] {
        let drawn = visible(&at(80).markdown(text));
        assert!(drawn.starts_with(marker), "{text:?}: {drawn}");
    }
}

#[test]
fn a_shell_call_shows_its_whole_command_wrapped_at_spaces() {
    use rig_harness::rig_core::message::{CallId, LocalCallId, ToolFunction, ToolName};
    let lines = [
        paragraph(1, 150),
        "cd crates/rig-markdown".to_owned(),
        paragraph(2, 90),
    ];
    let command = lines.join("\n");
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let function = ToolFunction::new(name, serde_json::json!({ "command": command }));
    let call = ToolCall::new(CallId::Local(LocalCallId::new()), function);
    let shown = visible(&at(80).call(&call));
    let rows: Vec<&str> = shown.lines().collect();
    assert!(rows.iter().all(|row| row.width() <= 80), "{rows:#?}");
    assert!(!shown.contains('…'), "{shown}");
    // Every word of every line, whole and in order, after the `$`.
    let words: Vec<&str> = shown.split_whitespace().collect();
    let expected: Vec<&str> = ["·", "$"]
        .into_iter()
        .chain(command.split_whitespace())
        .collect();
    assert_eq!(words, expected);
    // Each later line of the command starts a row of its own.
    let starts = |line: &str| {
        line.split_whitespace()
            .next()
            .unwrap_or_default()
            .to_owned()
    };
    for line in lines.iter().skip(1) {
        let first = starts(line);
        assert!(
            rows.iter().any(|row| row.trim_start().starts_with(&first)),
            "{first}: {rows:#?}"
        );
    }
}
