use super::{CqlValue, DynamicSearchFilter, Filter, ScyllaSearchFilter, SearchFilter};
use serde_json::json;

fn after_zero() -> ScyllaSearchFilter {
    ScyllaSearchFilter::gt("created_at", CqlValue::BigInt(0))
}

/// The refusal message, or as an error the CQL the filter would send to the server.
fn refusal(filter: &ScyllaSearchFilter) -> Result<String, String> {
    Err(filter.condition().to_string())
}

/// CQL binds positionally, so the rendered condition must carry exactly one
/// `?` per parameter — including `IN`, which renders one per value.
#[test]
fn every_parameterised_operator_uses_question_mark_placeholders() {
    let filter = ScyllaSearchFilter::gte("price", CqlValue::BigInt(5)).and(
        ScyllaSearchFilter::member("id", vec![CqlValue::BigInt(1), CqlValue::BigInt(2)]),
    );

    assert_eq!(filter.condition(), "(price >= ?) AND (id IN (?, ?))");
    assert_eq!(filter.condition().matches('?').count(), 3);
    assert_eq!(filter.params().len(), 3);
}

#[test]
fn or_filters_are_refused_wherever_they_nest() {
    let refused = Ok("Cannot compile 'OR' to the backend's filter type".to_string());
    for filter in [
        after_zero().or(after_zero()),
        after_zero().and(after_zero().or(after_zero())),
        after_zero().or(after_zero()).and(after_zero()),
    ] {
        assert_eq!(refusal(&filter), refused);
    }
}

#[test]
fn not_filters_are_refused_wherever_they_nest() {
    let refused = Ok("Cannot compile 'NOT' to the backend's filter type".to_string());
    for filter in [
        after_zero().not(),
        after_zero().not().not(),
        after_zero().and(after_zero().not()),
        after_zero().not().and(after_zero()),
    ] {
        assert_eq!(refusal(&filter), refused);
    }
}

#[test]
fn dynamic_or_filters_fail_at_conversion() {
    let atom = || Filter::gt("created_at", json!(0));
    assert!(ScyllaSearchFilter::from_dynamic_filter(atom().and(atom())).is_ok());
    for filter in [atom().or(atom()), atom().and(atom().or(atom()))] {
        assert!(
            ScyllaSearchFilter::from_dynamic_filter(filter).is_err(),
            "dynamic OR converted to CQL ScyllaDB rejects"
        );
    }
}
