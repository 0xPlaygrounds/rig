use super::indent;

#[test]
fn reported_lines_are_indented_under_their_package() {
    assert_eq!(indent(&["a", "b"]), "      a\n      b");
}
