use super::*;

fn deleted(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[test]
fn a_test_fn_goes_with_its_attributes_docs_and_comments() {
    let source = "\
use super::x;

// What the next cell pins.
/// The cell.
#[tokio::test]
async fn gone() {
    x();
}

#[test]
fn kept() {}
";
    let (edited, removed) =
        remove_tests(source, "openai::f", &deleted(&["openai::f::gone"])).expect("valid Rust");
    assert_eq!(removed, 1);
    assert_eq!(edited, "use super::x;\n\n#[test]\nfn kept() {}\n");
}

#[test]
fn a_row_goes_and_a_matrix_left_without_rows_goes_whole() {
    let source = r#"crate::matrix::native_matrix! {
    wrapper: with_x, wire: w, run: r;
    /// First.
    #[tokio::test]
    one: ("a/one", CELL, "g_one");
    #[tokio::test]
    two: ("a/two", CELL, "g_two");
}

crate::matrix::case_matrix! {
    family: faults_case;
    #[tokio::test]
    three: SCRIPTED => "g_three";
}
"#;
    let (edited, removed) = remove_tests(
        source,
        "openai::f",
        &deleted(&["openai::f::one", "openai::f::three"]),
    )
    .expect("valid Rust");
    assert_eq!(removed, 2);
    assert_eq!(
        edited,
        r#"crate::matrix::native_matrix! {
    wrapper: with_x, wire: w, run: r;
    #[tokio::test]
    two: ("a/two", CELL, "g_two");
}

"#
    );
}

#[test]
fn a_test_in_an_inline_module_is_found_by_its_full_name() {
    let source = "mod inner {\n    #[test]\n    fn gone() {}\n}\n";
    let (edited, removed) =
        remove_tests(source, "openai::f", &deleted(&["openai::f::inner::gone"]))
            .expect("valid Rust");
    assert_eq!(removed, 1);
    assert_eq!(edited, "mod inner {\n}\n");
    let (_, removed) =
        remove_tests(source, "openai::f", &deleted(&["openai::f::gone"])).expect("valid Rust");
    assert_eq!(removed, 0, "a name outside the module is another test");
}

#[test]
fn a_doc_table_row_naming_only_deleted_tests_goes() {
    let source = "\
//! | cell | pins |
//! |---|---|
//! | `gone` | one |
//! | `kept` | two |
//! | `gone`, `kept` | both |

#[test]
fn gone() {}

#[test]
fn kept() {}
";
    let (edited, removed) =
        remove_tests(source, "openai::f", &deleted(&["openai::f::gone"])).expect("valid Rust");
    assert_eq!(removed, 1);
    assert_eq!(
        edited,
        "\
//! | cell | pins |
//! |---|---|
//! | `kept` | two |
//! | `gone`, `kept` | both |

#[test]
fn kept() {}
"
    );
}
