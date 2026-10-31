use rig_core::tool::{FilesChanged, HunkLine, Tool, ToolContext};

use super::{Edit, EditArgs, Replacement, apply, batch_failure};

/// A misspelled key inside an edit, such as `replaceAll`, is refused
/// rather than ignored, which would replace one match instead of all.
#[test]
fn a_misspelled_key_inside_an_edit_is_refused() {
    let edit = serde_json::json!({"old_text": "a", "new_text": "b", "replaceAll": true});
    let args = serde_json::json!({"path": "a.rs", "edits": [edit]});
    let parsed = serde_json::from_value::<EditArgs>(args).map(|_| ());
    assert!(parsed.is_err_and(|error| error.to_string().contains("replaceAll")));
}

/// A failed edit is sent again; in a batch, every failed edit is named and
/// the whole batch is sent again.
#[test]
fn a_failure_names_every_failed_edit() {
    let single = batch_failure("a.rs", 1, &["`old_text` was not found".to_owned()]);
    assert_eq!(
        single,
        "`old_text` was not found. `a.rs` was not changed; fix the edit and send it again."
    );
    let one = batch_failure(
        "a.rs",
        3,
        &["edits[1]: `old_text` was not found".to_owned()],
    );
    assert_eq!(
        one,
        "edits[1]: `old_text` was not found. No edit was applied, so `a.rs` was not changed. \
         Send all 3 edits again, with this one fixed."
    );
    let failed = ["edits[0]: one".to_owned(), "edits[2]: two".to_owned()];
    let two = batch_failure("a.rs", 3, &failed);
    assert_eq!(
        two,
        "2 of the 3 edits failed:\nedits[0]: one.\nedits[2]: two.\nNo edit was applied, so \
         `a.rs` was not changed. Send all 3 edits again, with these fixed."
    );
}

/// `apply` makes of a file's text what the tool writes to it: here a CRLF
/// file with a byte-order mark, edited by a loose match.
#[test]
fn apply_gives_the_text_the_tool_writes() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-apply-{}.txt", std::process::id()));
    let before = "\u{feff}let a = \u{201c}x\u{201d};  \r\nlet b = 2;\r\n";
    std::fs::write(&path, before).ok();
    let path = path.to_string_lossy().into_owned();
    let edits = [Replacement::new("let a = \"x\";\n", "let a = 1;\n")];

    let applied = apply(&path, before, &edits);
    let mut context = ToolContext::new();
    let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
    let output = futures::executor::block_on(edited).unwrap_or_default();
    let written = std::fs::read_to_string(&path).ok();
    std::fs::remove_file(&path).ok();

    let applied = applied.ok();
    assert_eq!(
        applied.as_ref().map(|applied| applied.text.as_str()),
        Some("\u{feff}let a = 1;\r\nlet b = 2;\r\n")
    );
    assert_eq!(
        applied.as_ref().map(|applied| applied.text.clone()),
        written
    );
    assert_eq!(applied.as_ref().map(|applied| applied.notes.len()), Some(1));
    assert!(
        output.contains("Note: `old_text` matched only after"),
        "{output}"
    );
}

/// An edit's reply starts with the lines that now hold the changed text
/// and how far the file grew or shrank, not with the change itself, which
/// it still publishes whole for the terminal view.
#[test]
fn an_edit_answers_where_it_changed_the_file() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-where-{}.txt", std::process::id()));
    std::fs::write(&path, "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n").ok();
    let path = path.to_string_lossy().into_owned();
    let call = |edits: Vec<Replacement>| {
        let mut context = ToolContext::new();
        let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
        let reply = futures::executor::block_on(edited).unwrap_or_default();
        let said = reply.lines().next().unwrap_or_default().to_owned();
        (said, context.result::<FilesChanged>().ok().flatten())
    };

    // b becomes three lines; h goes.
    let (reply, published) = call(vec![
        Replacement::new("b\n", "b1\nb2\nb3\n"),
        Replacement::new("h\n", ""),
    ]);
    assert_eq!(
        reply,
        format!("Edited {path}: lines 2-4, line 10 (+1 line).")
    );
    assert!(!reply.contains("@@"), "{reply}");
    let hunks = published
        .map(|FilesChanged(changes)| changes)
        .unwrap_or_default();
    let lines = hunks
        .iter()
        .flat_map(|change| &change.hunks)
        .flat_map(|hunk| &hunk.lines);
    assert!(
        lines
            .clone()
            .any(|line| matches!(line, HunkLine::Same(text) if text == "a\n")),
        "the published change keeps its context: {hunks:?}"
    );
    assert!(
        lines
            .clone()
            .any(|line| matches!(line, HunkLine::Removed(text) if text == "h\n"))
    );

    let (reply, _) = call(vec![Replacement::new("j", "J").all()]);
    assert_eq!(reply, format!("Edited {path}: line 11."));
    std::fs::write(&path, "x\ny\nx\ny\nx\n").ok();
    let (reply, _) = call(vec![Replacement::new("x", "z").all()]);
    assert_eq!(
        reply,
        format!("Edited {path}: 3 replacements, lines 1, 3, 5.")
    );
    // The first place only removed a line, and the file shrank.
    std::fs::write(&path, "a\nb\nc\nd\ne\nf\n").ok();
    let (reply, _) = call(vec![
        Replacement::new("b\n", ""),
        Replacement::new("e\n", "E\n"),
    ]);
    std::fs::remove_file(&path).ok();
    assert_eq!(reply, format!("Edited {path}: line 2, 4 (-1 line)."));
}

/// After where it changed the file, an edit's reply shows those lines and
/// three around each as the file now holds them, places near each other in
/// one run, a long place by its ends, and lines that only went by the lines
/// around where they were.
#[test]
fn an_edit_shows_the_lines_around_each_change() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-around-{}.txt", std::process::id()));
    let numbered = |lines: std::ops::RangeInclusive<usize>| -> String {
        lines.map(|n| format!("{n}\n")).collect()
    };
    std::fs::write(&path, numbered(1..=100)).ok();
    let path = path.to_string_lossy().into_owned();
    let call = |edits: Vec<Replacement>| {
        let mut context = ToolContext::new();
        let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
        futures::executor::block_on(edited).unwrap_or_default()
    };

    let long: String = (0..40).map(|n| format!("new {n}\n")).collect();
    let reply = call(vec![
        Replacement::new("\n10\n", "\nten\n"),
        Replacement::new("\n12\n", "\ntwelve\n"),
        Replacement::new("\n50\n", "\n"),
        Replacement::new("\n80\n", format!("\n{long}")),
    ]);
    let middle: String = (0..12).map(|n| format!("new {n}\n")).collect();
    let end: String = (28..40).map(|n| format!("new {n}\n")).collect();
    let expected = format!(
        "Edited {path}: lines 10, 12, line 50, 79-118 (+38 lines).\n\
         [lines 7-15]\n7\n8\n9\nten\n11\ntwelve\n13\n14\n15\n\
         [lines 47-52]\n47\n48\n49\n51\n52\n53\n\
         [lines 76-121]\n77\n78\n79\n{middle}[… 16 lines]\n{end}81\n82\n83"
    );
    std::fs::remove_file(&path).ok();
    assert_eq!(reply, expected);
}

/// An `old_text` that is simply wrong is answered with the closest text in
/// the file, and no new read; only a file that changed since it was read
/// is to be read again.
#[test]
fn a_missed_old_text_shows_the_closest_text_and_asks_for_a_read_only_after_a_change() {
    let path = std::env::temp_dir().join(format!("rig-tools-missed-{}.rs", std::process::id()));
    let text = "fn main() {\n    let total = add(1, 2);\n    println!(\"{total}\");\n}\n";
    std::fs::write(&path, text).ok();
    let path = path.to_string_lossy().into_owned();
    let seen = crate::Seen::default();
    let read = crate::Read {
        seen: seen.clone(),
        ..crate::Read::default()
    };
    let edit = Edit {
        seen,
        ..Edit::default()
    };
    let call = |edits: Vec<Replacement>| {
        let args = EditArgs::new(&path, edits);
        let mut context = ToolContext::new();
        futures::executor::block_on(Tool::call(&edit, &mut context, args))
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    };
    let read_it = || {
        let args = crate::ReadArgs::new(&path);
        let mut context = ToolContext::new();
        futures::executor::block_on(Tool::call(&read, &mut context, args)).ok()
    };
    assert!(read_it().is_some());

    let missed = call(vec![
        Replacement::new("fn main() {", "fn main() -> () {"),
        Replacement::new(
            "    let total = add(1, 3);\n",
            "    let total = add(2, 3);\n",
        ),
    ]);
    assert!(
        missed.contains("edits[1]: `old_text` was not found")
            && missed.contains("copy it from the closest text in the file, shown below")
            && missed.contains(
                "The closest text to the `old_text` of edits[1] is line 2:\n    let total = add(1, 2);"
            )
            && !missed.contains("read it again"),
        "{missed}"
    );

    std::fs::write(&path, text.replace("add(1, 2)", "sum(1, 2)")).ok();
    let changed = call(vec![Replacement::new("add(1, 2)", "add(2, 3)")]);
    assert!(
        changed.contains("changed since it was last read, so read it again"),
        "{changed}"
    );
    assert!(read_it().is_some());
    let unrelated = call(vec![Replacement::new("nothing like it at all", "x")]);
    std::fs::remove_file(&path).ok();
    assert!(
        unrelated.contains("nothing in the file is close to it")
            && !unrelated.contains("read it again"),
        "{unrelated}"
    );
}

/// Edits apply in order, each to the text the ones before it left, so two
/// that touch the same lines, or a `replace_all` and an edit of one of its
/// matches, both apply. An edit whose text an earlier one took away is
/// refused naming that one.
#[test]
fn edits_apply_in_order_even_where_they_overlap() {
    let text = "fn area(w: u32, h: u32) -> u32 {\n    w * h\n}\n";
    let same_lines = [
        Replacement::new(
            "fn area(w: u32, h: u32) -> u32 {\n    w * h",
            "fn area(w: u64, h: u64) -> u64 {\n    w * h",
        ),
        Replacement::new("    w * h", "    w.saturating_mul(h)"),
    ];
    let applied = apply("a.rs", text, &same_lines).map(|applied| applied.text);
    assert_eq!(
        applied.ok().as_deref(),
        Some("fn area(w: u64, h: u64) -> u64 {\n    w.saturating_mul(h)\n}\n")
    );
    let all_then_one = [
        Replacement::new("u32", "u64").all(),
        Replacement::new("-> u64 {", "-> Option<u64> {"),
    ];
    let applied = apply("a.rs", text, &all_then_one).map(|applied| applied.replaced);
    assert_eq!(applied.ok(), Some(4));

    let taken = [
        Replacement::new("u32", "u64").all(),
        Replacement::new("-> u32 {", "-> Option<u32> {"),
    ];
    let refused = apply("a.rs", text, &taken)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        refused.starts_with(
            "edits[1]: `old_text` was in the file, but edits[0], applied before it, changed \
             that text; match the text as edits[0] leaves it, or make this change in edits[0]. \
             No edit was applied"
        ),
        "{refused}"
    );
}

/// A change past the 300 characters a reply shows of a line is not lost:
/// the cut says where the change is, and the command it gives shows it,
/// on a line of 350 characters and of 4,000.
#[cfg(unix)]
#[test]
fn a_change_past_the_cut_says_where_it_is() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-wide-{}.txt", std::process::id()));
    let cases = [
        (340, 7, "the change is at bytes 341-343", "301-350"),
        (4000, 0, "the change is at bytes 4001-4003", "301-4003"),
    ];
    for (before, after, at, window) in cases {
        let line = |middle: &str| format!("{}{middle}{}\n", "0".repeat(before), "0".repeat(after));
        std::fs::write(&path, format!("first\n{}", line("OLD"))).ok();
        let mut context = ToolContext::new();
        let args = EditArgs::new(path.to_string_lossy(), [Replacement::new("OLD", "NEW")]);
        let reply = futures::executor::block_on(Tool::call(&edit_tool, &mut context, args))
            .unwrap_or_default();
        let mark = format!(
            "{} [… cut at byte 300 of {}; {at}: in `shell`, `",
            "0".repeat(300),
            before + after + 3
        );
        let shown = reply
            .split_once(&mark)
            .and_then(|(_, rest)| rest.split_once(&format!("` shows bytes {window}]")))
            .map(|(command, _)| {
                let ran = crate::ShellArgs::new(command);
                futures::executor::block_on(crate::Shell::default().run(ran))
                    .map(|ran| ran.output)
                    .unwrap_or_default()
            });
        assert!(
            shown.is_some_and(|shown| shown.contains("NEW")),
            "{reply:.600}"
        );
    }
    // A byte-order mark before line 1 counts in its bytes.
    std::fs::write(&path, format!("\u{feff}{}OLD\n", "0".repeat(4000))).ok();
    let mut context = ToolContext::new();
    let args = EditArgs::new(path.to_string_lossy(), [Replacement::new("OLD", "NEW")]);
    let reply =
        futures::executor::block_on(Tool::call(&edit_tool, &mut context, args)).unwrap_or_default();
    std::fs::remove_file(&path).ok();
    assert!(
        reply.contains("[… cut at byte 303 of 4006; the change is at bytes 4004-4006: in `shell`")
            && reply.contains("| cut -b 304-4006 |"),
        "{reply:.600}"
    );
}

/// A loose match found at several places is not told to use `replace_all`,
/// which takes exact matches only.
#[test]
fn a_loose_ambiguous_match_is_not_sent_to_replace_all() {
    let text = "a = 1  \nb\na = 1\t\nb\n";
    let loose = apply("a.txt", text, &[Replacement::new("a = 1\nb", "a = 2\nb")]);
    let why = loose
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(why.contains("loosely matches 2 places"), "{why}");
    assert!(!why.contains("replace_all"), "{why}");
    let exact = apply("a.txt", "x\nx\n", &[Replacement::new("x", "y")]);
    let why = exact
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        why.contains("or set `replace_all` to change every one"),
        "{why}"
    );
}
