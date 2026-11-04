//! Compile-time diagnostics of the derive and attribute macros. Every
//! `tests/ui/*/pass_*.rs` case must compile, and every `fail_*.rs` case must
//! fail with its checked-in `.stderr`. Invalid macro input is a compile error,
//! never silently ignored or resolved.

#[test]
fn ui_cases() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/*/pass_*.rs");
    tests.compile_fail("tests/ui/*/fail_*.rs");
}
