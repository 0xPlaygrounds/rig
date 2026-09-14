//! `#[rig_tool(...)]` arguments follow a strict grammar: unknown or duplicate
//! entries and names that match no parameter are compile errors, never
//! silently ignored.

#[test]
fn invalid_tool_arguments_are_rejected() {
    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/tool_args/fail_params_unknown_param.rs");
    tests.compile_fail("tests/ui/tool_args/fail_required_unknown_param.rs");
    tests.compile_fail("tests/ui/tool_args/fail_required_option_param.rs");
    tests.compile_fail("tests/ui/tool_args/fail_params_non_string.rs");
    tests.compile_fail("tests/ui/tool_args/fail_duplicate_name.rs");
    tests.compile_fail("tests/ui/tool_args/fail_duplicate_param_description.rs");
}
