/// f32 slice -> the little-endian blob `build_search_query` now takes.
fn query_blob(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
use super::*;
use rig_core::embeddings::{EmbeddingError, EmbeddingResponse};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use std::cmp::Ordering;
use std::os::raw::c_char;
use std::sync::Once;
use tokio_rusqlite::Connection;

const SCORE_EPSILON: f64 = 1e-5;

fn test_metadata_columns() -> Vec<SqliteMetadataColumn> {
    vec![SqliteMetadataColumn {
        name: "category",
        metadata_type: SqliteMetadataType::Text,
    }]
}

fn typed_metadata_columns() -> Vec<SqliteMetadataColumn> {
    vec![
        SqliteMetadataColumn {
            name: "priority",
            metadata_type: SqliteMetadataType::Integer,
        },
        SqliteMetadataColumn {
            name: "rating",
            metadata_type: SqliteMetadataType::Float,
        },
        SqliteMetadataColumn {
            name: "published",
            metadata_type: SqliteMetadataType::Boolean,
        },
    ]
}

#[test]
fn json_column_text_decodes_to_json_object() -> anyhow::Result<()> {
    let column = Column::new("metadata", "JSON");
    let value = sqlite_column_value_to_json(
        0,
        &column,
        ValueRef::Text(br#"{"knowledge_doc_id":361,"knowledge_id":1,"user_id":1}"#),
    )?;

    let expected = serde_json::json!({
        "knowledge_doc_id": 361,
        "knowledge_id": 1,
        "user_id": 1
    });
    anyhow::ensure!(
        value == expected,
        "JSON column text should decode to a JSON object, got {value:?}"
    );

    Ok(())
}

#[test]
fn text_column_json_looking_text_stays_string() -> anyhow::Result<()> {
    let column = Column::new("metadata", "TEXT");
    let value = sqlite_column_value_to_json(
        0,
        &column,
        ValueRef::Text(br#"{"knowledge_doc_id":361,"knowledge_id":1,"user_id":1}"#),
    )?;

    let expected = serde_json::json!(r#"{"knowledge_doc_id":361,"knowledge_id":1,"user_id":1}"#);
    anyhow::ensure!(
        value == expected,
        "TEXT column should preserve JSON-looking text as a string, got {value:?}"
    );

    Ok(())
}

#[test]
fn json_column_invalid_text_returns_conversion_error() -> anyhow::Result<()> {
    let column = Column::new("metadata", "JSON");
    let err = match sqlite_column_value_to_json(0, &column, ValueRef::Text(b"not json")) {
        Ok(value) => anyhow::bail!("invalid JSON column text should fail, got {value:?}"),
        Err(err) => err,
    };

    anyhow::ensure!(
        matches!(
            err,
            rusqlite::Error::FromSqlConversionFailure(0, Type::Text, _)
        ),
        "invalid JSON column text should return a conversion error, got {err}"
    );

    Ok(())
}

#[test]
fn serde_json_value_column_value_round_trips_json_column() -> anyhow::Result<()> {
    let value = serde_json::json!({
        "knowledge_doc_id": 361,
        "knowledge_id": 1,
        "user_id": 1
    });
    anyhow::ensure!(
        value.column_type() == "JSON",
        "serde_json::Value should declare JSON column type"
    );

    let text = match value.to_sql_value() {
        Value::Text(text) => text,
        value => {
            anyhow::bail!("serde_json::Value should serialize as JSON text, got {value:?}")
        }
    };

    let column = Column::new("metadata", "JSON");
    let round_trip = sqlite_column_value_to_json(0, &column, ValueRef::Text(text.as_bytes()))?;
    anyhow::ensure!(
        round_trip == value,
        "serde_json::Value should round-trip through a JSON column, got {round_trip:?}"
    );

    Ok(())
}

fn filter_error<T: std::fmt::Debug>(
    result: Result<T, FilterError>,
    context: &str,
) -> anyhow::Result<FilterError> {
    match result {
        Ok(value) => anyhow::bail!("{context} should have failed, got {value:?}"),
        Err(err) => Ok(err),
    }
}

fn ensure_vector_store_filter_error<T: std::fmt::Debug>(
    result: Result<T, VectorStoreError>,
    context: &str,
) -> anyhow::Result<()> {
    match result {
        Err(VectorStoreError::FilterError(_)) => Ok(()),
        Err(err) => anyhow::bail!("{context} returned unexpected error: {err}"),
        Ok(value) => anyhow::bail!("{context} should have failed, got {value:?}"),
    }
}

#[test]
fn threshold_filter_uses_computed_similarity_expression() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .threshold(0.95)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &[],
        5,
    )?;

    anyhow::ensure!(
        where_clause.contains("e.embedding MATCH ?"),
        "missing vector match constraint: {where_clause}"
    );
    anyhow::ensure!(
        where_clause.contains("k = ?"),
        "missing vector k constraint: {where_clause}"
    );
    anyhow::ensure!(
        where_clause.contains("(1 - vec_distance_cosine(?1, e.embedding)) >= ?"),
        "threshold should use computed similarity expression: {where_clause}"
    );
    anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Real(0.95)),
        "unexpected threshold param: {params:?}"
    );

    Ok(())
}

#[test]
fn l2_threshold_filter_uses_l2_score_expression() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .threshold(-1.5)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::L2,
        &[],
        5,
    )?;

    anyhow::ensure!(
        where_clause.contains("(-vec_distance_l2(?1, e.embedding)) >= ?"),
        "threshold should use L2 score expression: {where_clause}"
    );
    anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Real(-1.5)),
        "unexpected threshold param: {params:?}"
    );

    Ok(())
}

#[test]
fn no_threshold_does_not_add_similarity_predicate() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &[],
        5,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(params.len() == 3, "unexpected params: {params:?}");

    Ok(())
}

#[test]
fn candidate_limit_at_k_cap_still_uses_knn_path() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &[],
        SQLITE_VEC_MAX_K,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "candidate limit at the cap should keep the KNN path: {where_clause}"
    );
    // ?1 (query vec) + MATCH (query vec) + k (candidate limit).
    anyhow::ensure!(params.len() == 3, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(2) == Some(&Value::Integer(SQLITE_VEC_MAX_K as i64)),
        "k param should be the candidate limit: {params:?}"
    );

    Ok(())
}

#[test]
fn candidate_limit_above_k_cap_falls_back_to_brute_force_scan() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &[],
        SQLITE_VEC_MAX_K + 1,
    )?;

    // Above the sqlite-vec k cap the MATCH/k KNN constraints are dropped so
    // the outer ORDER BY ... LIMIT ranks every row exactly. With no other
    // predicate the vector WHERE clause must be empty, not a bare `WHERE`.
    anyhow::ensure!(
        !where_clause.contains("MATCH") && !where_clause.contains("k = ?"),
        "brute-force scan should drop the KNN constraints: {where_clause}"
    );
    anyhow::ensure!(
        where_clause.is_empty(),
        "brute-force scan without filters should emit no WHERE clause: {where_clause:?}"
    );
    // Only ?1 (the query vector) remains; the second query vec and the k
    // param are gone, so downstream filter params stay aligned.
    anyhow::ensure!(params.len() == 1, "unexpected params: {params:?}");
    anyhow::ensure!(
        matches!(params.first(), Some(Value::Blob(_))),
        "remaining param should be the query vector: {params:?}"
    );

    Ok(())
}

#[test]
fn brute_force_scan_keeps_filter_params_aligned() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .threshold(0.95)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &[],
        SQLITE_VEC_MAX_K + 1,
    )?;

    // Dropping MATCH/k renumbers the threshold's anonymous `?` to index 2,
    // so it must bind the second params element (the query vec stays ?1).
    anyhow::ensure!(
        where_clause == "WHERE ((1 - vec_distance_cosine(?1, e.embedding)) >= ?)",
        "brute-force scan should keep native filters: {where_clause}"
    );
    anyhow::ensure!(params.len() == 2, "unexpected params: {params:?}");
    anyhow::ensure!(
        matches!(params.first(), Some(Value::Blob(_))),
        "first param should be the query vector: {params:?}"
    );
    anyhow::ensure!(
        params.get(1) == Some(&Value::Real(0.95)),
        "threshold param should follow the query vector: {params:?}"
    );

    Ok(())
}

#[test]
fn default_filter_composes_under_or_as_a_tautology() -> anyhow::Result<()> {
    let filter = SqliteSearchFilter::default().or(SqliteSearchFilter::eq(
        "category",
        serde_json::json!("docs"),
    ));

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(filter)
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;
    anyhow::ensure!(
        query.document_filter_clause == "AND ((1 = 1) OR (d.category = ?))",
        "default() under OR should render as a tautology: {}",
        query.document_filter_clause
    );

    Ok(())
}

#[test]
fn or_filter_uses_document_filter_to_preserve_boolean_semantics() -> anyhow::Result<()> {
    let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).or(
        SqliteSearchFilter::eq("title", serde_json::json!("archive")),
    );

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(filter)
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    anyhow::ensure!(
        filters.has_post_filters(),
        "OR filters should be applied after vector candidate search"
    );
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.vector_where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "OR filters should not be partially pushed into sqlite-vec: {}",
        query.vector_where_clause
    );
    anyhow::ensure!(
        query.document_filter_clause == "AND ((d.category = ?) OR (d.title = ?))",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("docs".to_string()))
            && query.params.get(4) == Some(&Value::Text("archive".to_string())),
        "unexpected OR filter params: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn indexed_filter_uses_vec0_metadata_constraint() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq(
            "category",
            serde_json::json!("docs"),
        ))
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &test_metadata_columns(),
        5,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.category = ?)",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Text("docs".to_string())),
        "unexpected filter param: {params:?}"
    );

    Ok(())
}

#[test]
fn negated_eq_filter_uses_vec0_metadata_inequality() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq("category", serde_json::json!("docs")).not())
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &test_metadata_columns(),
        5,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.category != ?)",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Text("docs".to_string())),
        "unexpected filter param: {params:?}"
    );

    Ok(())
}

#[test]
fn negated_range_comparison_uses_vec0_metadata_boundary() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::gt("priority", serde_json::json!(10)).not())
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &typed_metadata_columns(),
        5,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.priority <= ?)",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(params.len() == 4, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Integer(10)),
        "unexpected filter param: {params:?}"
    );

    Ok(())
}

#[test]
fn negated_boolean_eq_filter_uses_vec0_metadata_inequality() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq("published", serde_json::json!(true)).not())
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &typed_metadata_columns(),
        5,
    )?;

    anyhow::ensure!(
        where_clause == "WHERE e.embedding MATCH ? AND k = ? AND (e.published != ?)",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(
        params.get(3) == Some(&Value::Integer(1)),
        "unexpected boolean filter param: {params:?}"
    );

    Ok(())
}

#[test]
fn negated_between_filter_uses_document_filter() -> anyhow::Result<()> {
    let filter = SqliteSearchFilter::between("priority".to_string(), 1_i64..=10_i64).not();
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(filter)
        .build();

    let filters = render_search_filters(
        &req,
        SqliteDistanceMetric::Cosine,
        &typed_metadata_columns(),
    )?;
    anyhow::ensure!(
        filters.has_post_filters(),
        "negated range filters should be applied after vector candidate search"
    );
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.vector_where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "negated range filters should not be partially pushed into sqlite-vec: {}",
        query.vector_where_clause
    );
    anyhow::ensure!(
        query.document_filter_clause == "AND (NOT (d.priority between ? and ?))",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Integer(1))
            && query.params.get(4) == Some(&Value::Integer(10)),
        "unexpected negated between params: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn boolean_range_filter_is_rejected() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::gt(
            "published",
            serde_json::json!(false),
        ))
        .build();

    let err = filter_error(
        build_where_clause(
            &req,
            query_blob(&[1.0, 0.0]),
            SqliteDistanceMetric::Cosine,
            &typed_metadata_columns(),
            5,
        ),
        "boolean range filters",
    )?;

    anyhow::ensure!(
        err.to_string().contains("BOOLEAN"),
        "unexpected error for boolean range filter: {err}"
    );

    Ok(())
}

#[test]
fn indexed_between_filter_uses_vec0_metadata_constraints() -> anyhow::Result<()> {
    let filter = SqliteSearchFilter::between("priority".to_string(), 1_i64..=10_i64);
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(filter)
        .build();

    let (where_clause, params) = build_where_clause(
        &req,
        query_blob(&[1.0, 0.0]),
        SqliteDistanceMetric::Cosine,
        &typed_metadata_columns(),
        5,
    )?;

    anyhow::ensure!(
        where_clause
            == "WHERE e.embedding MATCH ? AND k = ? AND (e.priority >= ? AND e.priority <= ?)",
        "unexpected where clause: {where_clause}"
    );
    anyhow::ensure!(params.len() == 5, "unexpected params: {params:?}");
    anyhow::ensure!(
        params.get(3) == Some(&Value::Integer(1)) && params.get(4) == Some(&Value::Integer(10)),
        "between bounds should be bound as parameters: {params:?}"
    );

    Ok(())
}

#[test]
fn mismatched_metadata_filter_value_types_are_rejected() -> anyhow::Result<()> {
    let cases = [
        (
            SqliteSearchFilter::eq("published", serde_json::json!("true")),
            "boolean filter value",
        ),
        (
            SqliteSearchFilter::gt("priority", serde_json::json!(1.5)),
            "integer filter value",
        ),
        (
            SqliteSearchFilter::eq("category", serde_json::json!({ "name": "docs" })),
            "string filter value",
        ),
        (
            SqliteSearchFilter::between("priority".to_string(), "1".to_string()..="10".to_string()),
            "integer filter value",
        ),
    ];

    for (filter, expected) in cases {
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(5)
            .filter(filter)
            .build();

        let err = filter_error(
            build_where_clause(
                &req,
                query_blob(&[1.0, 0.0]),
                SqliteDistanceMetric::Cosine,
                &typed_metadata_columns()
                    .into_iter()
                    .chain(test_metadata_columns())
                    .collect::<Vec<_>>(),
                5,
            ),
            "mismatched metadata filter value",
        )?;

        anyhow::ensure!(
            err.to_string().contains(expected),
            "unexpected error for mismatched metadata filter value: {err}"
        );
    }

    Ok(())
}

#[test]
fn pattern_and_null_filters_use_document_filter() -> anyhow::Result<()> {
    let filter = SqliteSearchFilter::like("title".to_string(), "%O'Reilly%")
        .and(SqliteSearchFilter::glob("category".to_string(), "doc*"))
        .and(SqliteSearchFilter::is_null(
            "metadata->>'$.missing'".to_string(),
        ));
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(filter)
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    anyhow::ensure!(
        filters.has_post_filters(),
        "pattern and null filters should be applied after vector candidate search"
    );
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.vector_where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "pattern filters should not be pushed into sqlite-vec: {}",
        query.vector_where_clause
    );
    anyhow::ensure!(
        query.document_filter_clause
            == "AND (d.title like ?) AND (d.category glob ?) AND (d.metadata->>'$.missing' is null)",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("%O'Reilly%".to_string()))
            && query.params.get(4) == Some(&Value::Text("doc*".to_string())),
        "unexpected pattern filter params: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn nonindexed_filters_use_document_filter() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq("title", serde_json::json!("docs")))
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    anyhow::ensure!(
        filters.has_post_filters(),
        "non-indexed filters should be applied after vector candidate search"
    );
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.vector_where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "unexpected vector where clause: {}",
        query.vector_where_clause
    );
    anyhow::ensure!(
        query.document_filter_clause == "AND (d.title = ?)",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("docs".to_string())),
        "unexpected document filter param: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn json_metadata_expression_uses_document_filter() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq(
            "metadata->>'$.xxx'",
            serde_json::json!("vvv"),
        ))
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    anyhow::ensure!(
        filters.has_post_filters(),
        "JSON metadata expressions should be applied after vector candidate search"
    );
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.vector_where_clause == "WHERE e.embedding MATCH ? AND k = ?",
        "unexpected vector where clause: {}",
        query.vector_where_clause
    );
    anyhow::ensure!(
        query.document_filter_clause == "AND (d.metadata->>'$.xxx' = ?)",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("vvv".to_string())),
        "unexpected JSON metadata filter param: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn json_metadata_arrow_expression_binds_rhs_as_json_text() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq(
            "metadata->'$.xxx'",
            serde_json::json!("vvv"),
        ))
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.document_filter_clause == "AND (d.metadata->'$.xxx' = ?)",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("\"vvv\"".to_string())),
        "SQLite `->` should compare against JSON text: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn chained_json_metadata_expression_uses_final_operator_for_param_mode() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq(
            "metadata->'$.nested'->>'$.xxx'",
            serde_json::json!("vvv"),
        ))
        .build();

    let filters =
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns())?;
    let query = build_search_query(query_blob(&[1.0, 0.0]), filters, 5)?;

    anyhow::ensure!(
        query.document_filter_clause == "AND (d.metadata->'$.nested'->>'$.xxx' = ?)",
        "unexpected document filter clause: {}",
        query.document_filter_clause
    );
    anyhow::ensure!(
        query.params.get(3) == Some(&Value::Text("vvv".to_string())),
        "final `->>` should compare against SQL scalar text: {:?}",
        query.params
    );

    Ok(())
}

#[test]
fn unsupported_document_filter_expressions_are_rejected() -> anyhow::Result<()> {
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(5)
        .filter(SqliteSearchFilter::eq(
            "metadata) OR 1 = 1 --",
            serde_json::json!("vvv"),
        ))
        .build();

    let err = filter_error(
        render_search_filters(&req, SqliteDistanceMetric::Cosine, &test_metadata_columns()),
        "unsupported document filter expressions",
    )?;

    anyhow::ensure!(
        err.to_string()
            .contains("supported SQLite document filter expression"),
        "unexpected error for unsupported document filter expression: {err}"
    );

    Ok(())
}

#[tokio::test]
async fn live_search_orders_by_similarity_and_applies_threshold() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_search_orders_by_similarity_and_applies_threshold",
        vec![
            row("exact", "docs", "exact match", vec![1.0, 0.0]),
            row("close", "docs", "close match", vec![0.8, 0.6]),
            row("opposite", "docs", "opposite match", vec![-1.0, 0.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(3)
        .threshold(0.75)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    let exact_score = results.first().map(|(score, _, _)| *score);
    let close_score = results.get(1).map(|(score, _, _)| *score);

    anyhow::ensure!(
        ids.as_slice() == ["exact", "close"],
        "unexpected ids: {ids:?}"
    );
    anyhow::ensure!(
        exact_score
            .zip(close_score)
            .is_some_and(|(exact, close)| exact > close),
        "expected exact score to be greater than close score: {results:?}"
    );
    anyhow::ensure!(
        results.iter().all(|(score, _, _)| *score > 0.75),
        "threshold should remove low-scoring rows: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        result_ids.as_slice() == ["exact", "close"],
        "unexpected top_n_ids ids: {id_results:?}"
    );
    anyhow::ensure!(
        id_results.iter().all(|(score, _)| *score > 0.75),
        "top_n_ids threshold should remove low-scoring rows: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_reinsert_same_document_id_removes_stale_vec0_candidates() -> anyhow::Result<()> {
    register_sqlite_vec_extension();

    let conn = Connection::open(
        "file:live_reinsert_same_document_id_removes_stale_vec0_candidates?mode=memory",
    )
    .await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<TestDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store
        .add_rows(vec![row(
            "replace",
            "docs",
            "original near vector",
            vec![1.0, 0.0],
        )])
        .await?;
    vector_store
        .add_rows(vec![
            row("replace", "docs", "replacement far vector", vec![-1.0, 0.0]),
            row("fresh", "docs", "fresh near vector", vec![0.9, 0.1]),
        ])
        .await?;

    let index = vector_store.index(model);
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["fresh"],
        "stale replaced vectors should not consume sqlite-vec candidates: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["fresh"],
        "top_n_ids should not return or be starved by stale replaced vectors: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_reinsert_preserves_unrelated_multivector_embeddings() -> anyhow::Result<()> {
    register_sqlite_vec_extension();

    let conn = Connection::open(
        "file:live_reinsert_preserves_unrelated_multivector_embeddings?mode=memory",
    )
    .await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<TestDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    let multi_document = TestDocument {
        id: "multi".to_string(),
        category: "docs".to_string(),
        title: "multi-vector document".to_string(),
    };
    vector_store
        .add_rows(vec![
            (
                multi_document.clone(),
                vec![
                    Embedding {
                        document: "far chunk".to_string(),
                        vec: vec![-1.0, 0.0],
                    },
                    Embedding {
                        document: "exact chunk".to_string(),
                        vec: vec![1.0, 0.0],
                    },
                ],
            ),
            row(
                "replace",
                "docs",
                "initial replacement vector",
                vec![0.8, 0.2],
            ),
        ])
        .await?;
    vector_store
        .add_rows(vec![row(
            "replace",
            "docs",
            "replacement far vector",
            vec![-1.0, 0.0],
        )])
        .await?;

    let index = vector_store.index(model);
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .threshold(0.9)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["multi"],
        "reinsert should not delete another document's best embedding: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["multi"],
        "top_n_ids should preserve unrelated multivector embeddings after reinsert: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_multiple_embeddings_per_document_use_best_embedding() -> anyhow::Result<()> {
    let multi_document = TestDocument {
        id: "multi".to_string(),
        category: "docs".to_string(),
        title: "multi-vector document".to_string(),
    };
    let index = live_test_index(
        "live_multiple_embeddings_per_document_use_best_embedding",
        vec![
            (
                multi_document.clone(),
                vec![
                    Embedding {
                        document: "far chunk".to_string(),
                        vec: vec![-1.0, 0.0],
                    },
                    Embedding {
                        document: "exact chunk".to_string(),
                        vec: vec![1.0, 0.0],
                    },
                ],
            ),
            row("single", "docs", "single close chunk", vec![0.8, 0.6]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(2)
        .build();
    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["multi", "single"],
        "top_n should return each document once using its best embedding: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["multi", "single"],
        "top_n_ids should return each document once using its best embedding: {id_results:?}"
    );

    let threshold_req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(2)
        .threshold(1.0)
        .build();
    let threshold_results = index.top_n::<TestDocument>(threshold_req.clone()).await?;
    let threshold_ids = threshold_results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        threshold_ids.as_slice() == ["multi"],
        "threshold should include scores equal to the minimum and filter lower scores: {threshold_results:?}"
    );

    let threshold_id_results = index.top_n_ids(threshold_req).await?;
    let threshold_result_ids = threshold_id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        threshold_result_ids.as_slice() == ["multi"],
        "top_n_ids threshold should include scores equal to the minimum: {threshold_id_results:?}"
    );

    Ok(())
}

/// Regression test for issue #1904: a document owning many embeddings pushes
/// the internal candidate count past sqlite-vec's hard KNN `k = 4096` cap.
/// The search must fall back to a brute-force scan and still return the
/// exact, correctly ordered results instead of erroring with
/// "k value in knn query too large".
#[tokio::test]
async fn live_multivector_search_beyond_knn_k_cap_succeeds() -> anyhow::Result<()> {
    // Enough embeddings on one document that
    // `samples + (embedding_count - document_count)` exceeds 4096, forcing
    // the brute-force path. Before the fix this value bound the KNN `k`
    // directly and sqlite-vec rejected the query.
    let filler_chunks = (0..4100)
        .map(|i| Embedding {
            document: format!("filler chunk {i}"),
            vec: vec![0.0, 1.0],
        })
        .collect::<Vec<_>>();
    let filler_document = TestDocument {
        id: "filler".to_string(),
        category: "docs".to_string(),
        title: "many-embedding document".to_string(),
    };

    let index = live_test_index(
        "live_multivector_search_beyond_knn_k_cap_succeeds",
        vec![
            (filler_document, filler_chunks),
            row("best", "docs", "best", vec![1.0, 0.0]),
            row("mid", "docs", "mid", vec![0.5, 0.5]),
            row("worst", "docs", "worst", vec![-1.0, 0.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(3)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["best", "mid", "filler"],
        "brute-force scan should return the exact top-n past the knn k cap: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["best", "mid", "filler"],
        "top_n_ids should also brute-force past the knn k cap: {id_results:?}"
    );

    Ok(())
}

/// Regression test for issue #1904 (post-filter path): with more stored
/// embeddings than the sqlite-vec KNN cap, a filter on a non-indexed column
/// forces an exhaustive candidate scan. The brute-force fallback must both
/// avoid the `k` cap error and still find a match that ranks far below the
/// top 4096 by vector similarity.
#[tokio::test]
async fn live_post_filter_search_beyond_knn_k_cap_succeeds() -> anyhow::Result<()> {
    let mut rows = (0..4096)
        .map(|i| row(format!("noise-{i}"), "docs", "noise title", vec![1.0, 0.0]))
        .collect::<Vec<_>>();
    // The wanted document is the worst possible vector match, so it only
    // survives if candidate retrieval is exhaustive rather than capped at
    // the top 4096 by similarity.
    rows.push(row("wanted", "docs", "wanted title", vec![-1.0, 0.0]));

    let index = live_test_index("live_post_filter_search_beyond_knn_k_cap_succeeds", rows).await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "title",
            serde_json::json!("wanted title"),
        ))
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["wanted"],
        "exhaustive non-indexed filter past the knn k cap should still find the match: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["wanted"],
        "top_n_ids should also apply the exhaustive filter past the knn k cap: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_equal_score_results_are_ordered_by_document_id() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_equal_score_results_are_ordered_by_document_id",
        vec![
            row("b", "docs", "second id exact match", vec![1.0, 0.0]),
            row("a", "docs", "first id exact match", vec![1.0, 0.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(2)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["a", "b"],
        "equal-score top_n results should use document id as a stable tie-breaker: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["a", "b"],
        "equal-score top_n_ids results should use document id as a stable tie-breaker: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_common_sqlite_text_types_round_trip_in_top_n() -> anyhow::Result<()> {
    let index = live_common_type_test_index(
        "live_common_sqlite_text_types_round_trip_in_top_n",
        vec![common_type_row(
            "common",
            "varchar name",
            "clob notes",
            7,
            vec![1.0, 0.0],
        )],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .build();
    let results = index.top_n::<CommonTypeDocument>(req).await?;

    let Some((_, id, doc)) = results.first() else {
        anyhow::bail!("expected common type document result");
    };
    anyhow::ensure!(id == "common", "unexpected id: {id}");
    anyhow::ensure!(
        doc.name == "varchar name",
        "VARCHAR value should round-trip: {doc:?}"
    );
    anyhow::ensure!(
        doc.notes == "clob notes",
        "CLOB value should round-trip: {doc:?}"
    );
    anyhow::ensure!(doc.rank == 7, "NUMERIC value should round-trip: {doc:?}");

    Ok(())
}

#[tokio::test]
async fn live_json_column_structured_metadata_round_trips_in_top_n() -> anyhow::Result<()> {
    let metadata = StructuredMetadata {
        user_id: 1,
        knowledge_id: 1,
        knowledge_doc_id: 361,
    };
    let index = live_structured_json_metadata_test_index(
        "live_json_column_structured_metadata_round_trips_in_top_n",
        vec![structured_json_metadata_row(
            "structured",
            metadata.clone(),
            "metadata document",
            vec![1.0, 0.0],
        )],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .build();
    let results = index
        .top_n::<StructuredJsonMetadataDocument>(req.clone())
        .await?;

    let Some((_, id, doc)) = results.first() else {
        anyhow::bail!("expected structured JSON metadata document result");
    };
    anyhow::ensure!(id == "structured", "unexpected id: {id}");
    anyhow::ensure!(
        doc.metadata == metadata,
        "JSON column should deserialize into structured metadata: {doc:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    anyhow::ensure!(
        id_results.first().is_some_and(|(_, id)| id == "structured"),
        "top_n_ids should still return the structured metadata document id: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_text_affinity_metadata_filters_during_candidate_search() -> anyhow::Result<()> {
    let index = live_common_type_test_index(
        "live_text_affinity_metadata_filters_during_candidate_search",
        vec![
            common_type_row("nearest", "misc", "nearest excluded", 1, vec![1.0, 0.0]),
            common_type_row("docs", "docs", "docs match", 2, vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq("name", serde_json::json!("docs")))
        .build();

    let results = index.top_n::<CommonTypeDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        ids.as_slice() == ["docs"],
        "VARCHAR metadata filters should constrain sqlite-vec candidate search: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        result_ids.as_slice() == ["docs"],
        "top_n_ids should use VARCHAR metadata filters during candidate search: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_l2_metric_is_consistent() -> anyhow::Result<()> {
    let index = live_test_index_with_metric(
        "live_l2_metric_is_consistent",
        vec![
            row("exact", "docs", "exact match", vec![1.0, 0.0]),
            row("l2-close", "docs", "l2 close match", vec![1.0, 1.0]),
            row(
                "same-direction-far",
                "docs",
                "same direction far away",
                vec![10.0, 0.0],
            ),
        ],
        SqliteDistanceMetric::L2,
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(2)
        .threshold(-2.0)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    let exact_score = results
        .iter()
        .find(|(_, id, _)| id == "exact")
        .map(|(score, _, _)| *score);
    let close_score = results
        .iter()
        .find(|(_, id, _)| id == "l2-close")
        .map(|(score, _, _)| *score);

    anyhow::ensure!(
        ids.as_slice() == ["exact", "l2-close"],
        "L2 search should return the nearest L2 candidates: {results:?}"
    );
    anyhow::ensure!(
        exact_score
            .zip(close_score)
            .is_some_and(|(exact, close)| exact > close && close > -2.0),
        "expected L2 scores to be ordered and thresholded: {results:?}"
    );
    anyhow::ensure!(
        results.iter().all(|(score, _, _)| *score > -2.0),
        "threshold should be applied to L2 scores: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        result_ids.as_slice() == ["exact", "l2-close"],
        "top_n_ids should use the same L2 metric: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_indexed_filter_is_applied_during_candidate_search() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_indexed_filter_is_applied_during_candidate_search",
        vec![
            row(
                "nearest",
                "misc",
                "nearest excluded category",
                vec![1.0, 0.0],
            ),
            row("docs", "docs", "docs match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "category",
            serde_json::json!("docs"),
        ))
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        ids.as_slice() == ["docs"],
        "indexed filters should constrain sqlite-vec candidate search: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        result_ids.as_slice() == ["docs"],
        "top_n_ids should use indexed filters during candidate search: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_nonindexed_filter_is_applied_after_candidate_search() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_nonindexed_filter_is_applied_after_candidate_search",
        vec![
            row("nearest", "docs", "nearest excluded title", vec![1.0, 0.0]),
            row("wanted", "docs", "wanted title", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "title",
            serde_json::json!("wanted title"),
        ))
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["wanted"],
        "non-indexed filters should not be starved by the initial candidate limit: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["wanted"],
        "top_n_ids should apply non-indexed filters after candidate search: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_json_metadata_filter_is_applied_after_candidate_search() -> anyhow::Result<()> {
    let index = live_json_metadata_test_index(
        "live_json_metadata_filter_is_applied_after_candidate_search",
        vec![
            json_metadata_row("nearest", "docs", "skip", "nearest skipped", vec![1.0, 0.0]),
            json_metadata_row("matched", "docs", "vvv", "metadata match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "metadata->>'$.xxx'",
            serde_json::json!("vvv"),
        ))
        .build();

    let results = index.top_n::<JsonMetadataDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["matched"],
        "JSON metadata filters should not be starved by the initial candidate limit: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["matched"],
        "top_n_ids should apply JSON metadata filters after candidate search: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_json_arrow_filter_compares_against_json_text() -> anyhow::Result<()> {
    let index = live_json_metadata_test_index(
        "live_json_arrow_filter_compares_against_json_text",
        vec![
            json_metadata_row("nearest", "docs", "skip", "nearest skipped", vec![1.0, 0.0]),
            json_metadata_row("matched", "docs", "vvv", "metadata match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "metadata->'$.xxx'",
            serde_json::json!("vvv"),
        ))
        .build();

    let results = index.top_n::<JsonMetadataDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["matched"],
        "SQLite `->` JSON filters should compare against JSON text: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["matched"],
        "top_n_ids should apply SQLite `->` JSON filters: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_mixed_indexed_and_json_metadata_filters_are_applied() -> anyhow::Result<()> {
    let index = live_json_metadata_test_index(
        "live_mixed_indexed_and_json_metadata_filters_are_applied",
        vec![
            json_metadata_row(
                "nearest-docs",
                "docs",
                "skip",
                "nearest docs skipped by JSON metadata",
                vec![1.0, 0.0],
            ),
            json_metadata_row(
                "nearest-json",
                "misc",
                "vvv",
                "nearest JSON match skipped by category",
                vec![0.9, 0.1],
            ),
            json_metadata_row(
                "matched",
                "docs",
                "vvv",
                "matching category and JSON metadata",
                vec![0.0, 1.0],
            ),
        ],
    )
    .await?;

    let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).and(
        SqliteSearchFilter::eq("metadata->>'$.xxx'", serde_json::json!("vvv")),
    );
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(filter)
        .build();

    let results = index.top_n::<JsonMetadataDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["matched"],
        "indexed and JSON metadata filters should both be applied: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["matched"],
        "top_n_ids should apply both indexed and JSON metadata filters: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_negated_eq_filter_is_applied_during_candidate_search() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_negated_eq_filter_is_applied_during_candidate_search",
        vec![
            row(
                "nearest",
                "misc",
                "nearest excluded category",
                vec![1.0, 0.0],
            ),
            row("docs", "docs", "docs match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq("category", serde_json::json!("misc")).not())
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        ids.as_slice() == ["docs"],
        "negated filters should constrain sqlite-vec candidate search: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();

    anyhow::ensure!(
        result_ids.as_slice() == ["docs"],
        "top_n_ids should use negated filters during candidate search: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_top_n_reads_id_by_column_name_not_schema_position() -> anyhow::Result<()> {
    register_sqlite_vec_extension();

    let conn =
        Connection::open("file:live_top_n_reads_id_by_column_name_not_schema_position?mode=memory")
            .await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<ReorderedIdDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store
        .add_rows(vec![
            reordered_id_row("winner", "winner title", "docs", vec![1.0, 0.0]),
            reordered_id_row("other", "other title", "docs", vec![0.0, 1.0]),
        ])
        .await?;

    let index = vector_store.index(model);
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .build();

    let results = index.top_n::<ReorderedIdDocument>(req.clone()).await?;
    let Some((_, id, doc)) = results.first() else {
        anyhow::bail!("expected reordered-id result");
    };
    anyhow::ensure!(
        id == "winner",
        "top_n should return the id column, not the first schema column: {results:?}"
    );
    anyhow::ensure!(
        doc.id == "winner" && doc.title == "winner title",
        "document columns should still deserialize in schema order: {doc:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    anyhow::ensure!(
        id_results.first().map(|(_, id)| id.as_str()) == Some("winner"),
        "top_n_ids should agree with top_n id handling: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_internal_score_and_rank_column_names_do_not_shadow_search_columns()
-> anyhow::Result<()> {
    register_sqlite_vec_extension();

    let conn = Connection::open(
        "file:live_internal_score_and_rank_column_names_do_not_shadow_search_columns?mode=memory",
    )
    .await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<InternalAliasDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store
        .add_rows(vec![
            internal_alias_row(
                "winner",
                "payload score",
                "payload rank",
                "winner title",
                vec![1.0, 0.0],
            ),
            internal_alias_row(
                "other",
                "other score",
                "other rank",
                "other title",
                vec![0.0, 1.0],
            ),
        ])
        .await?;

    let index = vector_store.index(model);
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .threshold(0.9)
        .build();

    let results = index.top_n::<InternalAliasDocument>(req.clone()).await?;
    let Some((score, id, doc)) = results.first() else {
        anyhow::bail!("expected internal-alias document result");
    };

    anyhow::ensure!(id == "winner", "unexpected id: {results:?}");
    anyhow::ensure!(
        (*score - 1.0).abs() <= SCORE_EPSILON,
        "top_n should return computed score, not the document __rig_score column: {results:?}"
    );
    anyhow::ensure!(
        doc.rig_score == "payload score" && doc.rig_rank == "payload rank",
        "document columns with internal-looking names should still deserialize: {doc:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    anyhow::ensure!(
        id_results
            .first()
            .map(|(score, id)| ((*score - 1.0).abs() <= SCORE_EPSILON, id.as_str()))
            == Some((true, "winner")),
        "top_n_ids should agree with top_n despite internal-looking document columns: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_typed_columns_round_trip_and_filter_during_candidate_search() -> anyhow::Result<()> {
    let index = live_typed_test_index(
        "live_typed_columns_round_trip_and_filter_during_candidate_search",
        vec![
            typed_row(
                1,
                "misc",
                100,
                0.99,
                true,
                "nearest excluded by typed metadata",
                vec![1.0, 0.0],
            ),
            typed_row(2, "docs", 5, 0.95, true, "typed docs match", vec![0.0, 1.0]),
            typed_row(
                3,
                "docs",
                5,
                0.97,
                false,
                "unpublished docs match",
                vec![0.0, 0.9],
            ),
        ],
    )
    .await?;

    let filter = SqliteSearchFilter::lt("priority", serde_json::json!(10))
        .and(SqliteSearchFilter::gt("rating", serde_json::json!(0.9)))
        .and(SqliteSearchFilter::eq("published", serde_json::json!(true)));
    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(filter)
        .build();

    let results = index.top_n::<TypedTestDocument>(req.clone()).await?;
    anyhow::ensure!(
        results.len() == 1,
        "expected one typed document result: {results:?}"
    );

    let Some((_, id, doc)) = results.first() else {
        anyhow::bail!("expected one typed document result");
    };
    anyhow::ensure!(id == "2", "expected integer id to be returned as string");
    anyhow::ensure!(doc.id == 2, "typed integer id should round-trip: {doc:?}");
    anyhow::ensure!(
        doc.priority == 5,
        "typed integer field should round-trip: {doc:?}"
    );
    anyhow::ensure!(
        (doc.rating - 0.95).abs() < f64::EPSILON,
        "typed float field should round-trip: {doc:?}"
    );
    anyhow::ensure!(
        doc.published,
        "typed boolean field should round-trip: {doc:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["2"],
        "top_n_ids should use the same typed metadata filters: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_boolean_range_filter_is_rejected() -> anyhow::Result<()> {
    let index = live_typed_test_index(
        "live_boolean_range_filter_is_rejected",
        vec![
            typed_row(
                1,
                "misc",
                1,
                0.5,
                false,
                "nearest unpublished doc",
                vec![1.0, 0.0],
            ),
            typed_row(2, "docs", 2, 0.7, true, "published doc", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(2)
        .filter(SqliteSearchFilter::gt(
            "published",
            serde_json::json!(false),
        ))
        .build();

    ensure_vector_store_filter_error(
        index.top_n::<TypedTestDocument>(req.clone()).await,
        "top_n boolean range filter",
    )?;
    ensure_vector_store_filter_error(index.top_n_ids(req).await, "top_n_ids boolean range filter")?;

    Ok(())
}

#[tokio::test]
async fn live_mismatched_metadata_filter_value_type_is_rejected() -> anyhow::Result<()> {
    let index = live_typed_test_index(
        "live_mismatched_metadata_filter_value_type_is_rejected",
        vec![typed_row(
            1,
            "docs",
            1,
            0.95,
            true,
            "published doc",
            vec![1.0, 0.0],
        )],
    )
    .await?;

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(SqliteSearchFilter::eq(
            "published",
            serde_json::json!("true"),
        ))
        .build();

    ensure_vector_store_filter_error(
        index.top_n::<TypedTestDocument>(req.clone()).await,
        "top_n mismatched metadata filter value type",
    )?;
    ensure_vector_store_filter_error(
        index.top_n_ids(req).await,
        "top_n_ids mismatched metadata filter value type",
    )?;

    Ok(())
}

#[tokio::test]
async fn live_matches_exact_oracle_for_metrics_filters_and_thresholds() -> anyhow::Result<()> {
    let query = vec![1.0, 0.0];
    let rows = oracle_test_rows();
    let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs"))
        .and(SqliteSearchFilter::lt("priority", serde_json::json!(10)))
        .and(SqliteSearchFilter::gt("rating", serde_json::json!(0.8)))
        .and(SqliteSearchFilter::eq("published", serde_json::json!(true)));

    for distance_metric in [
        SqliteDistanceMetric::Cosine,
        SqliteDistanceMetric::L2,
        SqliteDistanceMetric::L1,
    ] {
        let threshold = oracle_threshold(distance_metric);
        let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
            .query("needle")
            .samples(u64::try_from(rows.len())?)
            .threshold(threshold)
            .filter(filter.clone())
            .build();
        let expected = exact_oracle_results(
            &rows,
            &query,
            distance_metric,
            threshold,
            rows.len(),
            |row| row.category == "docs" && row.priority < 10 && row.rating > 0.8 && row.published,
        )?;
        let test_name =
            format!("live_matches_exact_oracle_for_{distance_metric:?}").to_ascii_lowercase();
        let index = live_typed_test_index_with_metric(
            &test_name,
            sqlite_oracle_rows(&rows),
            distance_metric,
        )
        .await?;

        let results = index.top_n::<TypedTestDocument>(req.clone()).await?;
        let scored_ids = results
            .iter()
            .map(|(score, id, doc)| {
                anyhow::ensure!(
                    id == &doc.id.to_string(),
                    "top_n returned mismatched id and document: id={id}, doc={doc:?}"
                );
                Ok((*score, id.clone()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        assert_scored_ids_match(&scored_ids, &expected, distance_metric, "top_n")?;

        let id_results = index.top_n_ids(req).await?;
        assert_scored_ids_match(&id_results, &expected, distance_metric, "top_n_ids")?;
    }

    Ok(())
}

#[tokio::test]
async fn live_or_filter_preserves_mixed_document_semantics() -> anyhow::Result<()> {
    let index = live_test_index(
        "live_or_filter_preserves_mixed_document_semantics",
        vec![
            row(
                "nearest",
                "misc",
                "nearest excluded category",
                vec![1.0, 0.0],
            ),
            row("special", "misc", "special title", vec![0.9, 0.1]),
            row("docs", "docs", "far docs match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let filter = SqliteSearchFilter::eq("category", serde_json::json!("docs")).or(
        SqliteSearchFilter::eq("title", serde_json::json!("special title")),
    );

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(filter)
        .build();

    let results = index.top_n::<TestDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["special"],
        "OR filters should be applied as a whole document predicate: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["special"],
        "top_n_ids should preserve OR document semantics: {id_results:?}"
    );

    Ok(())
}

#[tokio::test]
async fn live_pattern_and_null_filters_are_applied_after_candidate_search() -> anyhow::Result<()> {
    let index = live_json_metadata_test_index(
        "live_pattern_and_null_filters_are_applied_after_candidate_search",
        vec![
            json_metadata_row("nearest", "docs", "skip", "skip this", vec![1.0, 0.0]),
            json_metadata_row("matched", "docs", "vvv", "metadata match", vec![0.0, 1.0]),
        ],
    )
    .await?;

    let filter = SqliteSearchFilter::is_null("metadata->>'$.missing'".to_string())
        .and(SqliteSearchFilter::like("title".to_string(), "metadata%"))
        .and(SqliteSearchFilter::glob("category".to_string(), "doc*"));

    let req = VectorSearchRequest::<SqliteSearchFilter>::builder()
        .query("needle")
        .samples(1)
        .filter(filter)
        .build();

    let results = index.top_n::<JsonMetadataDocument>(req.clone()).await?;
    let ids = results
        .iter()
        .map(|(_, id, _)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.as_slice() == ["matched"],
        "pattern and null filters should not be starved by the initial candidate limit: {results:?}"
    );

    let id_results = index.top_n_ids(req).await?;
    let result_ids = id_results
        .iter()
        .map(|(_, id)| id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        result_ids.as_slice() == ["matched"],
        "top_n_ids should apply pattern and null filters after candidate search: {id_results:?}"
    );

    Ok(())
}

type SqliteExtensionFn =
    unsafe extern "C" fn(*mut sqlite3, *mut *mut c_char, *const sqlite3_api_routines) -> i32;

fn register_sqlite_vec_extension() {
    static REGISTER_SQLITE_VEC: Once = Once::new();

    REGISTER_SQLITE_VEC.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    });
}

async fn live_test_index(
    name: &str,
    rows: Vec<(TestDocument, Vec<Embedding>)>,
) -> anyhow::Result<SqliteVectorIndex<TestDocument>> {
    live_test_index_with_metric(name, rows, SqliteDistanceMetric::Cosine).await
}

async fn live_test_index_with_metric(
    name: &str,
    rows: Vec<(TestDocument, Vec<Embedding>)>,
    distance_metric: SqliteDistanceMetric,
) -> anyhow::Result<SqliteVectorIndex<TestDocument>> {
    register_sqlite_vec_extension();

    let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
    let model = TestEmbeddingModel;
    let vector_store =
        SqliteVectorStore::with_distance_metric(conn, &model, distance_metric).await?;

    vector_store.add_rows(rows).await?;

    Ok(vector_store.index(model))
}

async fn live_typed_test_index(
    name: &str,
    rows: Vec<(TypedTestDocument, Vec<Embedding>)>,
) -> anyhow::Result<SqliteVectorIndex<TypedTestDocument>> {
    live_typed_test_index_with_metric(name, rows, SqliteDistanceMetric::Cosine).await
}

async fn live_typed_test_index_with_metric(
    name: &str,
    rows: Vec<(TypedTestDocument, Vec<Embedding>)>,
    distance_metric: SqliteDistanceMetric,
) -> anyhow::Result<SqliteVectorIndex<TypedTestDocument>> {
    register_sqlite_vec_extension();

    let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<TypedTestDocument> =
        SqliteVectorStore::with_distance_metric(conn, &model, distance_metric).await?;

    vector_store.add_rows(rows).await?;

    Ok(vector_store.index(model))
}

async fn live_common_type_test_index(
    name: &str,
    rows: Vec<(CommonTypeDocument, Vec<Embedding>)>,
) -> anyhow::Result<SqliteVectorIndex<CommonTypeDocument>> {
    register_sqlite_vec_extension();

    let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<CommonTypeDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store.add_rows(rows).await?;

    Ok(vector_store.index(model))
}

async fn live_json_metadata_test_index(
    name: &str,
    rows: Vec<(JsonMetadataDocument, Vec<Embedding>)>,
) -> anyhow::Result<SqliteVectorIndex<JsonMetadataDocument>> {
    register_sqlite_vec_extension();

    let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<JsonMetadataDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store.add_rows(rows).await?;

    Ok(vector_store.index(model))
}

async fn live_structured_json_metadata_test_index(
    name: &str,
    rows: Vec<(StructuredJsonMetadataDocument, Vec<Embedding>)>,
) -> anyhow::Result<SqliteVectorIndex<StructuredJsonMetadataDocument>> {
    register_sqlite_vec_extension();

    let conn = Connection::open(format!("file:{name}?mode=memory")).await?;
    let model = TestEmbeddingModel;
    let vector_store: SqliteVectorStore<StructuredJsonMetadataDocument> =
        SqliteVectorStore::new(conn, &model).await?;

    vector_store.add_rows(rows).await?;

    Ok(vector_store.index(model))
}

fn row(
    id: impl Into<String>,
    category: impl Into<String>,
    title: impl Into<String>,
    vec: Vec<f64>,
) -> (TestDocument, Vec<Embedding>) {
    let document = TestDocument {
        id: id.into(),
        category: category.into(),
        title: title.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

fn common_type_row(
    id: impl Into<String>,
    name: impl Into<String>,
    notes: impl Into<String>,
    rank: i64,
    vec: Vec<f64>,
) -> (CommonTypeDocument, Vec<Embedding>) {
    let document = CommonTypeDocument {
        id: id.into(),
        name: name.into(),
        notes: notes.into(),
        rank,
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.name,
            vec,
        }],
    )
}

fn json_metadata_row(
    id: impl Into<String>,
    category: impl Into<String>,
    xxx: impl AsRef<str>,
    title: impl Into<String>,
    vec: Vec<f64>,
) -> (JsonMetadataDocument, Vec<Embedding>) {
    let document = JsonMetadataDocument {
        id: id.into(),
        category: category.into(),
        metadata: serde_json::json!({ "xxx": xxx.as_ref() }).to_string(),
        title: title.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

fn structured_json_metadata_row(
    id: impl Into<String>,
    metadata: StructuredMetadata,
    title: impl Into<String>,
    vec: Vec<f64>,
) -> (StructuredJsonMetadataDocument, Vec<Embedding>) {
    let document = StructuredJsonMetadataDocument {
        id: id.into(),
        metadata,
        title: title.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

fn reordered_id_row(
    id: impl Into<String>,
    title: impl Into<String>,
    category: impl Into<String>,
    vec: Vec<f64>,
) -> (ReorderedIdDocument, Vec<Embedding>) {
    let document = ReorderedIdDocument {
        title: title.into(),
        id: id.into(),
        category: category.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

fn internal_alias_row(
    id: impl Into<String>,
    rig_score: impl Into<String>,
    rig_rank: impl Into<String>,
    title: impl Into<String>,
    vec: Vec<f64>,
) -> (InternalAliasDocument, Vec<Embedding>) {
    let document = InternalAliasDocument {
        id: id.into(),
        rig_score: rig_score.into(),
        rig_rank: rig_rank.into(),
        title: title.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

fn typed_row(
    id: i64,
    category: impl Into<String>,
    priority: i64,
    rating: f64,
    published: bool,
    title: impl Into<String>,
    vec: Vec<f64>,
) -> (TypedTestDocument, Vec<Embedding>) {
    let document = TypedTestDocument {
        id,
        category: category.into(),
        priority,
        rating,
        published,
        title: title.into(),
    };

    (
        document.clone(),
        vec![Embedding {
            document: document.title,
            vec,
        }],
    )
}

#[derive(Clone, Debug)]
struct OracleRow {
    document: TypedTestDocument,
    embedding: Vec<f64>,
}

#[derive(Debug)]
struct ExpectedScoredId {
    id: String,
    score: f64,
}

fn oracle_test_rows() -> Vec<OracleRow> {
    vec![
        oracle_row(1, "docs", 1, 0.95, true, "exact match", vec![1.0, 0.0]),
        oracle_row(2, "docs", 2, 0.90, true, "close match", vec![0.8, 0.6]),
        oracle_row(3, "docs", 3, 0.81, true, "borderline match", vec![0.5, 0.5]),
        oracle_row(
            4,
            "docs",
            4,
            0.70,
            true,
            "filtered by rating",
            vec![0.95, 0.05],
        ),
        oracle_row(
            5,
            "docs",
            15,
            0.99,
            true,
            "filtered by priority",
            vec![1.0, 0.0],
        ),
        oracle_row(
            6,
            "docs",
            5,
            0.99,
            false,
            "filtered by published",
            vec![1.0, 0.0],
        ),
        oracle_row(
            7,
            "misc",
            1,
            0.99,
            true,
            "filtered by category",
            vec![1.0, 0.0],
        ),
        oracle_row(8, "docs", 5, 0.95, true, "far match", vec![0.0, 1.0]),
    ]
}

fn oracle_row(
    id: i64,
    category: impl Into<String>,
    priority: i64,
    rating: f64,
    published: bool,
    title: impl Into<String>,
    embedding: Vec<f64>,
) -> OracleRow {
    OracleRow {
        document: TypedTestDocument {
            id,
            category: category.into(),
            priority,
            rating,
            published,
            title: title.into(),
        },
        embedding,
    }
}

fn sqlite_oracle_rows(rows: &[OracleRow]) -> Vec<(TypedTestDocument, Vec<Embedding>)> {
    rows.iter()
        .map(|row| {
            (
                row.document.clone(),
                vec![Embedding {
                    document: row.document.title.clone(),
                    vec: row.embedding.clone(),
                }],
            )
        })
        .collect()
}

fn oracle_threshold(distance_metric: SqliteDistanceMetric) -> f64 {
    match distance_metric {
        SqliteDistanceMetric::Cosine => 0.75,
        SqliteDistanceMetric::L2 => -0.8,
        SqliteDistanceMetric::L1 => -0.9,
    }
}

fn exact_oracle_results(
    rows: &[OracleRow],
    query: &[f64],
    distance_metric: SqliteDistanceMetric,
    threshold: f64,
    samples: usize,
    filter: impl Fn(&TypedTestDocument) -> bool,
) -> anyhow::Result<Vec<ExpectedScoredId>> {
    let mut expected = Vec::new();
    for row in rows {
        if !filter(&row.document) {
            continue;
        }

        let score = oracle_score(distance_metric, query, &row.embedding)?;
        if score >= threshold {
            expected.push(ExpectedScoredId {
                id: row.document.id.to_string(),
                score,
            });
        }
    }

    sort_expected_scores(&mut expected);
    expected.truncate(samples);
    Ok(expected)
}

fn sort_expected_scores(expected: &mut [ExpectedScoredId]) {
    expected.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.id.cmp(&rhs.id))
    });
}

fn oracle_score(
    distance_metric: SqliteDistanceMetric,
    query: &[f64],
    embedding: &[f64],
) -> anyhow::Result<f64> {
    anyhow::ensure!(
        query.len() == embedding.len(),
        "query and embedding dimensions differ: query={}, embedding={}",
        query.len(),
        embedding.len()
    );

    let query = query.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let embedding = embedding
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();

    let score = match distance_metric {
        SqliteDistanceMetric::Cosine => {
            let dot = query
                .iter()
                .zip(&embedding)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
            let embedding_norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            anyhow::ensure!(
                query_norm > 0.0 && embedding_norm > 0.0,
                "cosine oracle requires non-zero vectors"
            );
            dot / (query_norm * embedding_norm)
        }
        SqliteDistanceMetric::L2 => -query
            .iter()
            .zip(&embedding)
            .map(|(lhs, rhs)| {
                let delta = lhs - rhs;
                delta * delta
            })
            .sum::<f32>()
            .sqrt(),
        SqliteDistanceMetric::L1 => -query
            .iter()
            .zip(&embedding)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>(),
    };

    Ok(f64::from(score))
}

fn assert_scored_ids_match(
    actual: &[(f64, String)],
    expected: &[ExpectedScoredId],
    distance_metric: SqliteDistanceMetric,
    context: &str,
) -> anyhow::Result<()> {
    let actual_ids = actual.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>();
    let expected_ids = expected
        .iter()
        .map(|expected| expected.id.as_str())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        actual_ids == expected_ids,
        "{context} ids for {distance_metric:?} did not match exact oracle: actual={actual:?}, expected={expected:?}"
    );

    for ((actual_score, actual_id), expected) in actual.iter().zip(expected) {
        anyhow::ensure!(
            (actual_score - expected.score).abs() <= SCORE_EPSILON,
            "{context} score for {distance_metric:?} id `{actual_id}` did not match exact oracle: actual={actual_score}, expected={}",
            expected.score
        );
    }

    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TestDocument {
    id: String,
    category: String,
    title: String,
}

impl SqliteVectorStoreTable for TestDocument {
    fn name() -> &'static str {
        "live_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("category", "TEXT").indexed(),
            Column::new("title", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("category", Box::new(self.category.clone())),
            ("title", Box::new(self.title.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReorderedIdDocument {
    title: String,
    id: String,
    category: String,
}

impl SqliteVectorStoreTable for ReorderedIdDocument {
    fn name() -> &'static str {
        "live_reordered_id_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("title", "TEXT"),
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("category", "TEXT").indexed(),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("title", Box::new(self.title.clone())),
            ("id", Box::new(self.id.clone())),
            ("category", Box::new(self.category.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InternalAliasDocument {
    id: String,
    #[serde(rename = "__rig_score")]
    rig_score: String,
    #[serde(rename = "__rig_rank")]
    rig_rank: String,
    title: String,
}

impl SqliteVectorStoreTable for InternalAliasDocument {
    fn name() -> &'static str {
        "live_internal_alias_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("__rig_score", "TEXT"),
            Column::new("__rig_rank", "TEXT"),
            Column::new("title", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("__rig_score", Box::new(self.rig_score.clone())),
            ("__rig_rank", Box::new(self.rig_rank.clone())),
            ("title", Box::new(self.title.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CommonTypeDocument {
    id: String,
    name: String,
    notes: String,
    rank: i64,
}

impl SqliteVectorStoreTable for CommonTypeDocument {
    fn name() -> &'static str {
        "live_common_type_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("name", "VARCHAR(255)").indexed(),
            Column::new("notes", "CLOB"),
            Column::new("rank", "NUMERIC"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("name", Box::new(self.name.clone())),
            ("notes", Box::new(self.notes.clone())),
            ("rank", Box::new(self.rank)),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct JsonMetadataDocument {
    id: String,
    category: String,
    metadata: String,
    title: String,
}

impl SqliteVectorStoreTable for JsonMetadataDocument {
    fn name() -> &'static str {
        "live_json_metadata_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("category", "TEXT").indexed(),
            Column::new("metadata", "TEXT"),
            Column::new("title", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("category", Box::new(self.category.clone())),
            ("metadata", Box::new(self.metadata.clone())),
            ("title", Box::new(self.title.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct StructuredMetadata {
    user_id: i64,
    knowledge_id: i64,
    knowledge_doc_id: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StructuredJsonMetadataDocument {
    id: String,
    metadata: StructuredMetadata,
    title: String,
}

impl SqliteVectorStoreTable for StructuredJsonMetadataDocument {
    fn name() -> &'static str {
        "live_structured_json_metadata_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("metadata", "JSON"),
            Column::new("title", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            (
                "metadata",
                Box::new(serde_json::json!({
                    "user_id": self.metadata.user_id,
                    "knowledge_id": self.metadata.knowledge_id,
                    "knowledge_doc_id": self.metadata.knowledge_doc_id,
                })),
            ),
            ("title", Box::new(self.title.clone())),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TypedTestDocument {
    id: i64,
    category: String,
    priority: i64,
    rating: f64,
    published: bool,
    title: String,
}

impl SqliteVectorStoreTable for TypedTestDocument {
    fn name() -> &'static str {
        "live_typed_test_documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "INTEGER PRIMARY KEY"),
            Column::new("category", "TEXT").indexed(),
            Column::new("priority", "INTEGER").indexed(),
            Column::new("rating", "FLOAT").indexed(),
            Column::new("published", "BOOLEAN").indexed(),
            Column::new("title", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.to_string()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id)),
            ("category", Box::new(self.category.clone())),
            ("priority", Box::new(self.priority)),
            ("rating", Box::new(self.rating)),
            ("published", Box::new(self.published)),
            ("title", Box::new(self.title.clone())),
        ]
    }
}

#[derive(Clone)]
struct TestEmbeddingModel;

impl EmbeddingModel for TestEmbeddingModel {
    fn max_documents(&self) -> usize {
        16
    }

    fn ndims(&self) -> usize {
        2
    }

    async fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Ok(EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![1.0, 0.0],
                })
                .collect(),
            "mock",
        ))
    }
}
