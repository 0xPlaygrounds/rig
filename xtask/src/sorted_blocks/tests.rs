use super::check_source;

fn failures_for(source: &str) -> Vec<String> {
    let mut failures = Vec::new();
    check_source("f", source, &mut failures);
    failures
}

#[test]
fn sorted_toml_block_passes() {
    let source =
        "# sorted: start\na = 1\nb = { x = [\n  \"z\",\n] }\n# note\nc = 3\n# sorted: end\n";
    assert!(failures_for(source).is_empty());
}

#[test]
fn out_of_order_toml_entry_is_named_with_both_lines() {
    let source = "# sorted: start\nb = 1\na = 2\n# sorted: end\n";
    let failures = failures_for(source);
    assert_eq!(failures, vec!["f:3: `a` sorts before `b` (line 2)"]);
}

#[test]
fn multi_line_entries_do_not_leak_their_continuation_lines() {
    // The continuation line `"a",` would sort before `b` if it were an entry.
    let source = "# sorted: start\nb = [\n  \"a\",\n]\nc = 1\n# sorted: end\n";
    assert!(failures_for(source).is_empty());
}

#[test]
fn rust_mod_and_markdown_rows_are_entries() {
    let rust = "// sorted: start\npub mod b;\npub mod a;\n// sorted: end\n";
    assert_eq!(failures_for(rust).len(), 1);
    let md = "<!-- sorted: start -->\n| Name | X |\n| --- | --- |\n| B | 1 |\n| A | 2 |\n<!-- sorted: end -->\n";
    assert_eq!(failures_for(md), vec!["f:5: `A` sorts before `B` (line 4)"]);
}

#[test]
fn unclosed_block_is_reported_and_counted() {
    let mut failures = Vec::new();
    let blocks = check_source("f", "# sorted: start\na = 1\n", &mut failures);
    assert_eq!(blocks, 1);
    assert_eq!(failures, vec!["f:1: `sorted: start` is never closed"]);
}

#[test]
fn duplicate_keys_fail() {
    assert_eq!(
        failures_for("# sorted: start\na = 1\na = 2\n# sorted: end\n").len(),
        1
    );
}
