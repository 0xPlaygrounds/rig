use super::*;

#[test]
fn dynamic_filter_compiles_to_native_aws_documents() {
    let filter = Filter::eq("status", serde_json::json!("ready"))
        .and(Filter::eq("tags", serde_json::json!(["rust", "ai"])))
        .and(Filter::gt("score", serde_json::json!(4.5)));

    let compiled = S3SearchFilter::from_dynamic_filter(filter)
        .expect("JSON values should compile to AWS documents");

    assert_eq!(
        document_to_json_value(compiled.inner()),
        serde_json::json!({
            "$and": [{
                "$and": [
                    { "status": { "$eq": "ready" } },
                    { "tags": { "$eq": ["rust", "ai"] } }
                ]
            }, {
                "score": { "$gt": 4.5 }
            }]
        })
    );
}

#[test]
fn extension_operators_build_the_documented_filter_shapes() {
    let number = |n| Document::Number(aws_smithy_types::Number::PosInt(n));
    let filter = S3SearchFilter::gte("score", number(5))
        .or(S3SearchFilter::lte("score", number(1)))
        .or(S3SearchFilter::exists("status"))
        .not();

    assert_eq!(
        document_to_json_value(filter.inner()),
        serde_json::json!({
            "$not": {
                "$or": [
                    { "$or": [
                        { "score": { "$gte": 5 } },
                        { "score": { "$lte": 1 } }
                    ]},
                    { "$exists": { "status": true } }
                ]
            }
        })
    );
}
