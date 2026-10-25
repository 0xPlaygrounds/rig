use super::words;

#[test]
fn a_line_splits_into_program_and_arguments() {
    let (program, args) = words(Some("code --wait"));
    assert_eq!(
        (program, args.collect::<Vec<_>>()),
        ("code", vec!["--wait"])
    );
    for blank in [None, Some(""), Some("  ")] {
        let (program, mut args) = words(blank);
        assert_eq!((program, args.next()), ("vi", None), "{blank:?}");
    }
}
