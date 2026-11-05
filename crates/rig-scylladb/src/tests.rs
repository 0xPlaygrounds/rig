use super::{CqlValue, ScyllaSearchFilter, SearchFilter};

/// CQL binds positionally, so the rendered condition must carry exactly one
/// `?` per parameter — including `IN`, which renders one per value.
#[test]
fn every_parameterised_operator_uses_question_mark_placeholders() {
    let filter = ScyllaSearchFilter::gte("price", CqlValue::BigInt(5))
        .and(ScyllaSearchFilter::member(
            "id",
            vec![CqlValue::BigInt(1), CqlValue::BigInt(2)],
        ))
        .and(ScyllaSearchFilter::lte(
            "created_at",
            CqlValue::BigInt(i64::MAX),
        ));

    assert_eq!(
        filter.condition(),
        "((price >= ?) AND (id IN (?, ?))) AND (created_at <= ?)"
    );
    assert_eq!(filter.condition().matches('?').count(), 4);
    assert_eq!(filter.params().len(), 4);
}

#[test]
fn unsupported_logical_operators_are_rejected_even_inside_conjunctions() {
    let atom = || ScyllaSearchFilter::gt("created_at", CqlValue::BigInt(0));
    for filter in [
        atom().or(atom()),
        atom().not(),
        atom().and(atom().or(atom())),
        atom().not().and(atom()),
        atom().not().not(),
    ] {
        assert!(matches!(
            filter.validate(),
            Err(super::FilterError::Must(_, _))
        ));
    }
    assert!(atom().validate().is_ok());
    assert!(
        atom()
            .and(ScyllaSearchFilter::lt(
                "created_at",
                CqlValue::BigInt(i64::MAX)
            ))
            .validate()
            .is_ok()
    );
}

#[test]
fn dynamic_disjunctions_fail_during_filter_conversion() {
    use super::{DynamicSearchFilter, Filter};
    use serde_json::json;

    let atom = || Filter::gt("created_at", json!(0));
    assert!(ScyllaSearchFilter::from_dynamic_filter(atom().or(atom())).is_err());
    assert!(ScyllaSearchFilter::from_dynamic_filter(atom().and(atom().or(atom()))).is_err());
    assert!(ScyllaSearchFilter::from_dynamic_filter(atom().or(atom()).and(atom())).is_err());
    assert!(ScyllaSearchFilter::from_dynamic_filter(atom().and(atom())).is_ok());
}
