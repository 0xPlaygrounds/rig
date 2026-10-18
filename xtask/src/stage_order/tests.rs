use super::*;

fn parse(source: &str) -> syn::File {
    syn::parse_file(source).expect("test source parses")
}

#[test]
fn a_crate_with_stage_systems_must_check_their_order_in_a_test() {
    let shipped = parse(
        "fn build(app: &mut App) {
            app.add_systems(Update, tick)
                .add_systems(rig_ecs::AgentPass, gate.in_set(Stage::Gate));
        }",
    );
    let elsewhere = parse("fn build(app: &mut App) { app.add_systems(Update, tick); }");
    let checked = parse(
        "#[test]
        fn stages_are_ordered_with_the_defaults() {
            rig_defaults_test_support::assert_stages_ordered(Plan);
        }",
    );
    let replacing = parse(
        "#[test]
        fn it_replaces_the_default() {
            rig_defaults_test_support::assert_stages_ordered_replacing::<Default, _>(Plan);
        }",
    );
    // Named only in a comment, a string or an unrelated position.
    let mentioned = parse(
        "// assert_stages_ordered(Plan)
        fn f() { let _ = \"assert_stages_ordered(Plan)\"; g(assert_stages_ordered); }",
    );

    assert!(unchecked([(&shipped, false)]));
    assert!(unchecked([(&shipped, false), (&mentioned, true)]));
    // Called only in shipped code, it is no test.
    assert!(unchecked([(&shipped, false), (&checked, false)]));
    assert!(!unchecked([(&shipped, false), (&checked, true)]));
    assert!(!unchecked([(&shipped, false), (&replacing, true)]));
    // Stage systems added only in tests, or none at all.
    assert!(!unchecked([(&elsewhere, false), (&shipped, true)]));
}
