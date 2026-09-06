use super::*;

#[test]
fn queue_preserves_ids_and_does_not_count_mappings_as_verified() {
    let manifest = json!({"baseline_revision":"base","scenarios":[
        {"id":"a","source":"family.rs","classification":"unclassified"},
        {"id":"b","source":"family.rs","classification":"agent","ecs":{"test":"native"}}
    ]});
    let queue = generate(&manifest, &json!({"requirements":[]})).expect("queue");
    assert_eq!(queue["counts"]["scenarios"], 2);
    assert_eq!(queue["counts"]["mapped"], 1);
    assert_eq!(queue["counts"]["historical_scoped_verified"], 0);
    assert_eq!(queue["items"].as_array().expect("items").len(), 2);
}

#[test]
fn duplicate_scenario_cannot_silently_shrink_denominator() {
    let row = json!({"id":"a","source":"family.rs","classification":"unclassified"});
    assert!(
        generate(
            &json!({"scenarios":[row.clone(),row]}),
            &json!({"requirements":[]})
        )
        .is_err()
    );
}
