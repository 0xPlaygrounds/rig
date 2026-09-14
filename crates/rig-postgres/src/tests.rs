use super::{PgSearchFilter, SearchFilter};
use serde_json::json;

/// `gte`/`lte`/`member` previously emitted `?` placeholders while
/// `eq`/`gt`/`lt` emitted `$`; the query renumbering only rewrites `$`, so
/// any `?` would reach Postgres verbatim and break the query.
#[test]
fn every_parameterised_operator_uses_dollar_placeholders() {
    let gte = PgSearchFilter::gte("price", json!(5));
    let lte = PgSearchFilter::lte("price", json!(10));

    let (cond, values) = gte.and(lte).into_clause();
    assert_eq!(cond, "(price >= $) AND (price <= $)");
    assert!(!cond.contains('?'));
    assert_eq!(cond.matches('$').count(), values.len());

    let member = PgSearchFilter::member("id", vec![json!(1), json!(2)]);
    let (cond, values) = PgSearchFilter::eq("kind", json!("fruit"))
        .and(member)
        .into_clause();
    assert_eq!(cond, "(kind = $) AND (id is in ($, $))");
    assert!(!cond.contains('?'));
    assert_eq!(cond.matches('$').count(), values.len());
}
