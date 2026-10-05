use super::{Filter, SearchFilter, SqlCondition};
use serde_json::json;

type F = Filter<serde_json::Value>;

#[test]
fn gt_and_lt_compare_the_named_field() {
    let doc = json!({ "price": 10, "text": "banana" });
    assert!(F::gt("price", json!(5)).satisfies(&doc));
    assert!(!F::gt("price", json!(10)).satisfies(&doc));
    assert!(F::lt("price", json!(20)).satisfies(&doc));
    assert!(!F::lt("price", json!(10)).satisfies(&doc));
    // Missing / non-comparable fields never satisfy an ordering filter.
    assert!(!F::gt("missing", json!(1)).satisfies(&doc));
    assert!(!F::gt("text", json!(1)).satisfies(&doc));
}

#[test]
fn and_or_combine_leaf_filters() {
    let doc = json!({ "category": "fruit", "price": 10 });
    let both = F::eq("category", json!("fruit")).and(F::gt("price", json!(5)));
    assert!(both.satisfies(&doc));

    let missing_branch = F::eq("category", json!("fruit")).and(F::gt("price", json!(50)));
    assert!(!missing_branch.satisfies(&doc));

    let either = F::eq("category", json!("veg")).or(F::lt("price", json!(50)));
    assert!(either.satisfies(&doc));
}

#[test]
fn try_interpret_converts_nested_leaf_values() {
    let f: Filter<i64> =
        Filter::Eq("a".into(), 1).and(Filter::Gt("b".into(), 2).or(Filter::Lt("c".into(), 3)));
    let out: Filter<String> = f
        .try_interpret(|v| Ok::<_, std::convert::Infallible>(v.to_string()))
        .unwrap();
    match out {
        Filter::And(lhs, rhs) => {
            assert!(matches!(*lhs, Filter::Eq(ref k, ref v) if k == "a" && v == "1"));
            match *rhs {
                Filter::Or(l, r) => {
                    assert!(matches!(*l, Filter::Gt(ref k, ref v) if k == "b" && v == "2"));
                    assert!(matches!(*r, Filter::Lt(ref k, ref v) if k == "c" && v == "3"));
                }
                other => panic!("expected Or, got {other:?}"),
            }
        }
        other => panic!("expected And, got {other:?}"),
    }
}

#[test]
fn sql_condition_renders_only_constructor_placeholders() {
    let condition = SqlCondition::binary("doc->>'$a'", "=", "$", 1)
        .and(SqlCondition::raw("col$ is null").not())
        .or(SqlCondition::list("id", "IN", "$", vec![2, 3]))
        .and(SqlCondition::between("n$", "$", 4, 5))
        .and(SqlCondition::range("m", "$", 6, 7));

    assert_eq!(
        condition.render_placeholders(|i| format!("${}", i + 10)),
        "((((doc->>'$a' = $10) AND (NOT (col$ is null))) OR (id IN ($11, $12))) \
         AND (n$ between $13 and $14)) AND (m >= $15 AND m <= $16)"
    );
    assert_eq!(condition.params(), &[1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn sql_condition_keeps_placeholders_through_serde() {
    let condition = SqlCondition::binary("k$", "=", "$", 1).and(SqlCondition::list(
        "id",
        "IN",
        "$",
        Vec::<i32>::new(),
    ));
    let json = serde_json::to_value(&condition).unwrap();
    let restored: SqlCondition<i32> = serde_json::from_value(json).unwrap();

    assert_eq!(restored.condition(), "(k$ = $) AND (id IN ())");
    assert_eq!(
        restored.render_placeholders(|_| "?"),
        "(k$ = ?) AND (id IN ())"
    );
}
