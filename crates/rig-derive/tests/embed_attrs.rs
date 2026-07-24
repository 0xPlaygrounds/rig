//! `#[derive(Embed)]` attribute conflicts on a single field are compile
//! errors, never silently resolved.

#[test]
fn conflicting_embed_attributes_are_rejected() {
    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/embed_attrs/fail_duplicate_embed_with.rs");
}
