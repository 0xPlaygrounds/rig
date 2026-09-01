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
        .or(ScyllaSearchFilter::ne("kind", CqlValue::Text("veg".to_string())).not());

    assert_eq!(
        filter.condition(),
        "((price >= ?) AND (id IN (?, ?))) OR (NOT (kind != ?))"
    );
    assert_eq!(filter.condition().matches('?').count(), 4);
    assert_eq!(filter.params().len(), 4);
}
