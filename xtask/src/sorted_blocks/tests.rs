use super::check_source;

fn failures_for(source: &str) -> Vec<String> {
    let mut failures = Vec::new();
    check_source("f", source, &mut failures);
    failures
}

#[test]
fn multi_line_entries_and_comments_are_one_entry() {
    // The continuation line `"a",` would sort before `b` if it were an entry.
    let source =
        "# sorted: start\nb = [\n  \"a\",\n]\n# note\nc = { x = [\n  \"z\",\n] }\n# sorted: end\n";
    assert!(failures_for(source).is_empty());
}

#[test]
fn out_of_order_and_duplicate_entries_name_both_lines() {
    assert_eq!(
        failures_for("# sorted: start\nb = 1\na = 2\n# sorted: end\n"),
        ["f:3: `a` sorts before `b` (line 2)"]
    );
    assert_eq!(
        failures_for("# sorted: start\na = 1\na = 2\n# sorted: end\n"),
        ["f:3: `a` sorts before `a` (line 2)"]
    );
}

#[test]
fn rust_mods_and_markdown_rows_are_entries() {
    let rust = "// sorted: start\npub mod b;\npub mod a;\n// sorted: end\n";
    assert_eq!(failures_for(rust).len(), 1);
    // The header row is skipped, and the order is case-insensitive.
    let md = "<!-- sorted: start -->\n| Name | X |\n| --- | --- |\n| ScyllaDB | 1 |\n| SQLite | 2 |\n| B | 3 |\n<!-- sorted: end -->\n";
    assert_eq!(
        failures_for(md),
        ["f:6: `B` sorts before `SQLite` (line 5)"]
    );
}

#[test]
fn unclosed_block_is_counted_and_reported() {
    let mut failures = Vec::new();
    let blocks = check_source("f", "# sorted: start\na = 1\n", &mut failures);
    assert_eq!(blocks, 1);
    assert_eq!(failures, ["f:1: `sorted: start` is never closed"]);
}
