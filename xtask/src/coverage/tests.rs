use super::*;

fn parse(args: &[&str]) -> Result<Options> {
    Options::parse(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>())
}

#[test]
fn lines_and_shapes_run_by_default_and_mutation_is_opt_in() {
    let options = parse(&[]).unwrap();
    assert_eq!(options.parts, vec![Part::Lines, Part::Shapes]);
    assert!(!options.check);
    let options = parse(&["--check", "--mutants"]).unwrap();
    assert_eq!(
        options.parts,
        vec![Part::Lines, Part::Shapes, Part::Mutants]
    );
    assert!(options.check);
}

#[test]
fn only_selects_parts_and_mutants_adds_once() {
    assert_eq!(
        parse(&["--only", "shapes"]).unwrap().parts,
        vec![Part::Shapes]
    );
    assert_eq!(
        parse(&["--only", "mutants,lines", "--mutants"])
            .unwrap()
            .parts,
        vec![Part::Mutants, Part::Lines]
    );
    assert!(parse(&["--only", "regions"]).is_err());
}

#[test]
fn a_baseline_intersects_three_runs_and_a_check_runs_once() {
    assert_eq!(parse(&[]).unwrap().runs, 3);
    assert_eq!(parse(&["--check"]).unwrap().runs, 1);
    assert_eq!(parse(&["--check", "--runs", "2"]).unwrap().runs, 2);
}

#[test]
fn numeric_options_must_be_positive() {
    assert_eq!(parse(&["--sample", "8"]).unwrap().sample, Some(8));
    assert_eq!(parse(&["--jobs", "3"]).unwrap().jobs, 3);
    for args in [["--sample", "0"], ["--jobs", "x"], ["--runs", "-1"]] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
    assert!(parse(&["--sample"]).is_err());
}

#[test]
fn per_test_stands_alone() {
    assert!(parse(&["--per-test"]).unwrap().per_test);
    assert!(parse(&["--per-test", "--check"]).is_err());
    assert!(parse(&["--per-test", "--only", "lines"]).is_err());
    assert!(parse(&["--frobnicate"]).is_err());
}

#[test]
fn every_part_has_its_own_baseline_file() {
    let files: Vec<&str> = [Part::Lines, Part::Shapes, Part::Mutants]
        .into_iter()
        .map(Part::file)
        .collect();
    assert_eq!(files, ["lines.tsv", "shapes.tsv", "mutants.tsv"]);
}
