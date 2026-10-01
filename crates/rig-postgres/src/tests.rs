use super::{PgSearchFilter, PgVectorDistanceFunction, SearchFilter, render_search_query};
use rig_core::vector_store::request::VectorSearchRequest;
use serde_json::json;

/// `gte`/`lte`/`member` previously emitted `?` placeholders while
/// `eq`/`gt`/`lt` emitted `$`; the query renumbering only rewrites `$`, so
/// any `?` would reach Postgres verbatim and break the query.
#[test]
fn every_parameterised_operator_uses_dollar_placeholders() {
    let gte = PgSearchFilter::gte("price", json!(5));
    let lte = PgSearchFilter::lte("price", json!(10));

    let (cond, values) = gte.and(lte).0.into_parts();
    assert_eq!(cond, "(price >= $) AND (price <= $)");
    assert!(!cond.contains('?'));
    assert_eq!(cond.matches('$').count(), values.len());

    let member = PgSearchFilter::member("id", vec![json!(1), json!(2)]);
    let (cond, values) = PgSearchFilter::eq("kind", json!("fruit"))
        .and(member)
        .0
        .into_parts();
    assert!(!cond.contains('?'));
    assert_eq!(cond.matches('$').count(), values.len());
}

/// The `WHERE` fragment of the inner `SELECT`, or `None` when the query has
/// no filter at all.
fn rendered_where(
    distance: PgVectorDistanceFunction,
    req: &VectorSearchRequest<PgSearchFilter>,
) -> (Option<String>, Vec<serde_json::Value>) {
    let (sql, params) = render_search_query(&distance, "documents", false, req);
    let clause = sql
        .split("FROM documents")
        .nth(1)
        .and_then(|rest| rest.split("ORDER BY id, distance").next())
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
        .map(str::to_owned);
    (clause, params)
}

fn request() -> VectorSearchRequest<PgSearchFilter> {
    VectorSearchRequest::builder().query("q").samples(5).build()
}

fn thresholded(
    threshold: f64,
    filter: Option<PgSearchFilter>,
) -> VectorSearchRequest<PgSearchFilter> {
    let builder = VectorSearchRequest::builder()
        .query("q")
        .samples(5)
        .threshold(threshold);
    match filter {
        Some(filter) => builder.filter(filter).build(),
        None => builder.build(),
    }
}

fn filtered(filter: PgSearchFilter) -> VectorSearchRequest<PgSearchFilter> {
    VectorSearchRequest::builder()
        .query("q")
        .samples(5)
        .filter(filter)
        .build()
}

#[test]
fn no_filter_renders_no_where_clause() {
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &request());
    assert_eq!(clause, None);
    assert!(params.is_empty());
}

/// rig#2376 bug 2: `"WHERE" + condition` used to render `WHEREprice >= $3`.
#[test]
fn single_condition_filter_renders_where_with_separator() {
    let req = filtered(PgSearchFilter::gte("price", json!(5)));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &req);
    assert_eq!(clause.as_deref(), Some("WHERE (price >= $3)"));
    assert_eq!(params, vec![json!(5)]);
}

/// rig#2376 bug 1: `member` used to render `id is in (...)`, which is not a
/// Postgres operator.
#[test]
fn member_filter_renders_sql_in() {
    let req = filtered(PgSearchFilter::member("id", vec![json!(1), json!(2)]));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &req);
    assert_eq!(clause.as_deref(), Some("WHERE (id IN ($3, $4))"));
    assert_eq!(params, vec![json!(1), json!(2)]);
}

/// rig#2376 bugs 3 and 4: the threshold used to render `distance > $3`
/// inside the inner `SELECT`, where `distance` is only a select-list alias,
/// and compared a *distance* with `>` although `threshold` is a minimum
/// *similarity*. Each operator gets its own similarity expression that
/// repeats the operator instead of naming the alias.
#[test]
fn threshold_renders_minimum_similarity_per_distance_function() {
    let cases = [
        (
            PgVectorDistanceFunction::Cosine,
            "WHERE (1 - (embedding <=> $1) >= $3)",
        ),
        (
            PgVectorDistanceFunction::Jaccard,
            "WHERE (1 - (embedding <%> $1) >= $3)",
        ),
        (
            PgVectorDistanceFunction::InnerProduct,
            "WHERE (-(embedding <#> $1) >= $3)",
        ),
        (
            PgVectorDistanceFunction::L2,
            "WHERE (-(embedding <-> $1) >= $3)",
        ),
        (
            PgVectorDistanceFunction::L1,
            "WHERE (-(embedding <+> $1) >= $3)",
        ),
        (
            PgVectorDistanceFunction::Hamming,
            "WHERE (-(embedding <~> $1) >= $3)",
        ),
    ];
    for (distance, expected) in cases {
        let req = thresholded(0.8, None);
        let (clause, params) = rendered_where(distance, &req);
        assert_eq!(clause.as_deref(), Some(expected));
        assert_eq!(params, vec![json!(0.8)]);
    }
}

/// The threshold binds first (`$3`) and the filter's placeholders continue
/// from `$4`, in the same order the values are pushed. The `$1` inside the
/// similarity expression is the query vector and must not be renumbered.
#[test]
fn threshold_and_compound_filter_number_parameters_in_bind_order() {
    let filter = PgSearchFilter::eq("kind", json!("fruit"))
        .and(PgSearchFilter::member("id", vec![json!(1), json!(2)]));
    let req = thresholded(0.5, Some(filter));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &req);
    assert_eq!(
        clause.as_deref(),
        Some("WHERE (1 - (embedding <=> $1) >= $3) AND ((kind = $4) AND (id IN ($5, $6)))")
    );
    assert_eq!(params, vec![json!(0.5), json!("fruit"), json!(1), json!(2)]);
}

/// The outer query still orders by raw ascending distance and binds the
/// limit as `$2`, so callers without a threshold see unchanged results.
#[test]
fn outer_query_orders_by_distance_and_limits_on_second_parameter() {
    let (sql, _) = render_search_query(&PgVectorDistanceFunction::L2, "docs", true, &request());
    let compact: String = sql.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(compact.contains("SELECT id, document, distance FROM ("));
    assert!(compact.contains("embedding <-> $1 as distance FROM docs ORDER BY id, distance"));
    assert!(compact.ends_with(") as d ORDER BY distance LIMIT $2"));
}

/// A `$` inside spliced text, such as a key, is not a placeholder. Only real
/// placeholders are numbered, so later parameters keep their positions.
#[test]
fn dollar_in_spliced_key_is_not_renumbered() {
    let filter = PgSearchFilter::eq("document->>'$price'", json!(5))
        .and(PgSearchFilter::is_null("col$1"))
        .and(PgSearchFilter::eq("kind", json!("fruit")));
    let req = thresholded(0.5, Some(filter));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &req);
    assert_eq!(
        clause.as_deref(),
        Some(
            "WHERE (1 - (embedding <=> $1) >= $3) AND \
             (((document->>'$price' = $4) AND (col$1 is null)) AND (kind = $5))"
        )
    );
    assert_eq!(params, vec![json!(0.5), json!(5), json!("fruit")]);
}

/// A `$` inside a bound `like` pattern stays in the parameter, and the
/// placeholders after it keep their numbers.
#[test]
fn dollar_in_like_pattern_is_bound_not_renumbered() {
    let filter = PgSearchFilter::like("document->>'$name'", "%$%")
        .and(PgSearchFilter::eq("kind", json!("fruit")));
    let req = thresholded(0.5, Some(filter));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &req);
    assert_eq!(
        clause.as_deref(),
        Some(
            "WHERE (1 - (embedding <=> $1) >= $3) AND \
             ((document->>'$name' like $4) AND (kind = $5))"
        )
    );
    assert_eq!(params, vec![json!(0.5), json!("%$%"), json!("fruit")]);
}

/// `like` and `similar_to` take runtime patterns and bind them unquoted.
#[test]
fn like_and_similar_to_bind_runtime_patterns() {
    let prefix = String::from("app");
    let filter = PgSearchFilter::like("name", format!("{prefix}%"))
        .or(PgSearchFilter::similar_to("name", "(pear|plum)"));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &filtered(filter));
    assert_eq!(
        clause.as_deref(),
        Some("WHERE ((name like $3) OR (name similar to $4))")
    );
    assert_eq!(params, vec![json!("app%"), json!("(pear|plum)")]);
}

/// `between` binds both bounds like `gte`/`lte`, accepts floats, and binds a
/// non-finite float as `NULL` instead of splicing `NaN` into the SQL.
#[test]
fn between_binds_its_bounds() {
    let filter = PgSearchFilter::between("price", 5..=10)
        .and(PgSearchFilter::between("score", 0.5..=f64::NAN));
    let (clause, params) = rendered_where(PgVectorDistanceFunction::Cosine, &filtered(filter));
    assert_eq!(
        clause.as_deref(),
        Some("WHERE ((price between $3 and $4) AND (score between $5 and $6))")
    );
    assert_eq!(params, vec![json!(5), json!(10), json!(0.5), json!(null)]);
}
