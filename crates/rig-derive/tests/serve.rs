//! The bus's diagnostics name what a handler author implements.

#![allow(clippy::expect_used)]

#[test]
fn a_type_that_is_not_a_serve_is_told_to_implement_serve() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/serve/fail_not_a_serve.rs");
}

/// The expected output names `Serve` and nothing the author cannot
/// implement: the crate-private boxed trait, the sealed family trait.
#[test]
fn the_diagnostic_recommends_serve_and_nothing_private() {
    let stderr = std::fs::read_to_string("tests/ui/serve/fail_not_a_serve.stderr")
        .expect("the case's expected output is checked in");
    assert!(stderr.contains("Serve"), "{stderr}");
    assert!(
        !stderr.contains("`Handler`") && !stderr.contains("Served"),
        "the diagnostic must not recommend a trait the author cannot implement:\n{stderr}"
    );
}
