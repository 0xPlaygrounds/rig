use super::*;

#[test]
fn only_a_deliveries_change_is_churn() {
    let base = r#"{"header":{"deliveries":[1,2],"run_spec":7},"records":[{"a":1}]}"#;
    let reordered = r#"{"header":{"deliveries":[2,1],"run_spec":7},"records":[{"a":1}]}"#;
    let real = r#"{"header":{"deliveries":[2,1],"run_spec":7},"records":[{"a":2}]}"#;
    assert!(delivery_only(base, reordered));
    assert!(!delivery_only(base, real));
    assert!(!delivery_only(base, base), "an unchanged file is not churn");
    assert!(!delivery_only(base, "not json"));
}
