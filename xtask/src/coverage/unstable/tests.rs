use super::*;

const FILE_A: &str = "crates/rig-core/src/a.rs";

fn row(region: Region, source: &str, reason: &str) -> Unstable {
    Unstable {
        file: FILE_A.to_owned(),
        source: source.to_owned(),
        region,
        code: "return;".to_owned(),
        reason: reason.to_owned(),
    }
}

fn coverage() -> FileCoverage {
    FileCoverage {
        lines: BTreeMap::from([(1, true), (2, true), (3, false)]),
        branches: BTreeMap::from([((2, 0, 0), true), ((2, 0, 1), false)]),
    }
}

#[test]
fn rows_round_trip_sorted_and_a_bad_row_is_refused() {
    let rows = vec![
        row(Region::Branch((2, 0, 1)), "h", "a worker may answer first"),
        row(Region::Line(2), "h", "the abort may land first"),
    ];
    let text = render(&rows);
    assert_eq!(
        text,
        format!(
            "{HEADER}\n{FILE_A}\th\t2\treturn;\tthe abort may land first\n\
             {FILE_A}\th\t2.0.1\treturn;\ta worker may answer first\n"
        )
    );
    let back = parse(&text).unwrap();
    assert_eq!(back[0].region, Region::Line(2));
    assert_eq!(back[1].region, Region::Branch((2, 0, 1)));
    assert!(parse(&format!("{HEADER}\n{FILE_A}\th\t2.x\tcode\twhy")).is_err());
    assert!(parse(&format!("{HEADER}\n{FILE_A}\th\t2")).is_err());
    assert!(parse("").unwrap().is_empty());
}

#[test]
fn only_a_row_written_for_the_current_source_is_left_out() {
    let mut files = BTreeMap::from([(FILE_A.to_owned(), coverage())]);
    let rows = [
        row(Region::Line(2), "now", "why"),
        row(Region::Branch((2, 0, 0)), "now", "why"),
        row(Region::Line(1), "before", "why"),
    ];
    exclude(&mut files, &rows, |_| Some("now".to_owned()));
    let a = &files[FILE_A];
    assert_eq!(a.lines.keys().copied().collect::<Vec<_>>(), [1, 3]);
    assert_eq!(a.branches.keys().copied().collect::<Vec<_>>(), [(2, 0, 1)]);
    assert_eq!(per_file(&rows)[FILE_A], (2, 1));
}

#[test]
fn a_row_needs_a_reason_and_the_baselines_source() {
    let rows = [
        row(Region::Line(2), "h", " "),
        row(Region::Line(3), "old", "why"),
    ];
    let found = problems(&rows, |_| Some("h".to_owned()));
    assert_eq!(found.len(), 2);
    assert!(found[0].contains("2 has no reason"), "{found:?}");
    assert!(
        found[1].contains("3 was written for another baseline"),
        "{found:?}"
    );
    assert!(problems(&rows[..1], |_| Some("h".to_owned())).len() == 1);
    assert!(
        problems(&[row(Region::Line(2), "h", "why")], |_| Some(
            "h".to_owned()
        ))
        .is_empty()
    );
}

#[test]
fn the_runs_disagree_on_what_some_covered_and_not_all() {
    let union = BTreeMap::from([(FILE_A.to_owned(), coverage())]);
    let mut every = coverage();
    every.lines.insert(2, false);
    every.branches.insert((2, 0, 0), false);
    let intersection = BTreeMap::from([(FILE_A.to_owned(), every)]);
    assert_eq!(
        disagreements(&union, &intersection),
        [
            (FILE_A.to_owned(), Region::Line(2)),
            (FILE_A.to_owned(), Region::Branch((2, 0, 0)))
        ]
    );
    assert!(disagreements(&union, &union).is_empty());
}

#[test]
fn a_refresh_follows_moved_code_drops_gone_code_and_adds_new_regions() {
    let text = "fn a() {\n    return;\n}\nfn b() {\n    x();\n    return;\n}\n";
    let current = |_: &str| Some((text.to_owned(), "new".to_owned()));
    // Written when the second `return;` was on line 5: the nearest now is 6.
    let rows = [
        row(Region::Branch((5, 0, 1)), "old", "why"),
        Unstable {
            code: "gone();".to_owned(),
            ..row(Region::Line(9), "old", "why")
        },
    ];
    let found = [
        (FILE_A.to_owned(), Region::Branch((6, 0, 1))),
        (FILE_A.to_owned(), Region::Line(5)),
    ];
    let (kept, notes) = refresh(&rows, &found, current);
    assert_eq!(
        kept,
        [
            row(Region::Branch((6, 0, 1)), "new", "why"),
            Unstable {
                code: "x();".to_owned(),
                ..row(Region::Line(5), "new", "")
            }
        ]
    );
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert!(notes[0].contains("dropped") && notes[1].contains("added"));
    let (same, _) = refresh(&kept, &[], current);
    assert_eq!(same, kept);
}

#[test]
fn the_nearest_matching_line_wins_and_the_earlier_on_a_tie() {
    let text = "return;\nx\nreturn;\nx\nreturn;\n";
    assert_eq!(nearest(text, "return;", 3), Some(3));
    assert_eq!(nearest(text, "return;", 4), Some(3));
    assert_eq!(nearest(text, "missing", 4), None);
    assert_eq!(code("\t  let a =\tb;  "), "let a = b;");
}
