use std::sync::Arc;

use arrow_array::Array;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use serde_json::json;

use super::{embedded_row, embedding_column, filter_embeddings, rows_to_batch};

#[tokio::test]
async fn test_column_filtering() {
    let field_a = Field::new("id", DataType::Int64, false);
    let field_b = Field::new("my_bool", DataType::Boolean, false);
    let field_c = Field::new(
        "my_embeddings",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
        false,
    );
    let field_d = Field::new(
        "my_list",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 10),
        false,
    );

    let schema = Schema::new(vec![field_a, field_b, field_c, field_d]);

    let columns = filter_embeddings(&schema);

    assert_eq!(
        columns,
        vec![
            "id".to_string(),
            "my_bool".to_string(),
            "my_list".to_string()
        ]
    );
}

#[tokio::test]
async fn test_column_filtering_2() {
    let field_a = Field::new("id", DataType::Int64, false);
    let field_b = Field::new("my_bool", DataType::Boolean, false);
    let field_c = Field::new(
        "my_embeddings",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
        false,
    );
    let field_d = Field::new(
        "my_other_embeddings",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
        false,
    );

    let schema = Schema::new(vec![field_a, field_b, field_c, field_d]);

    let columns = filter_embeddings(&schema);

    assert_eq!(columns, vec!["id".to_string(), "my_bool".to_string()]);
}

fn float_list(name: &str, item: DataType, size: i32) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", item, true)), size),
        false,
    )
}

#[test]
fn embedding_column_is_the_only_float_list_or_the_named_one() {
    let id = Field::new("id", DataType::Utf8, false);
    let single = Schema::new(vec![id.clone(), float_list("vector", DataType::Float32, 2)]);
    let double = Schema::new(vec![
        id,
        float_list("title_vector", DataType::Float64, 2),
        float_list("body_vector", DataType::Float64, 2),
    ]);

    assert_eq!(
        embedding_column(&single, None).ok().as_deref(),
        Some("vector")
    );
    assert_eq!(
        embedding_column(&double, Some("body_vector"))
            .ok()
            .as_deref(),
        Some("body_vector")
    );
    assert!(embedding_column(&double, None).is_err());
    assert!(embedding_column(&double, Some("id")).is_err());
}

#[test]
fn rows_decode_against_the_table_schema() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("document", DataType::Utf8, true),
        float_list("vector", DataType::Float32, 2),
    ]));
    let rows = vec![
        embedded_row(
            &json!({ "id": "doc0", "rank": 1, "extra": true }),
            "vector",
            vec![0.5, 1.0],
        ),
        embedded_row(&json!("a bare string"), "vector", vec![0.25, 0.75]),
    ];

    let batch = rows_to_batch(schema.clone(), &rows).unwrap().unwrap();
    assert_eq!(batch.schema(), schema);
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.column(0).null_count(), 1);
    assert_eq!(batch.column(2).null_count(), 1);

    let wrong_size = [embedded_row(&json!({ "id": "doc2" }), "vector", vec![0.5])];
    assert!(rows_to_batch(schema.clone(), &wrong_size).is_err());
    assert!(rows_to_batch(schema, &[]).unwrap().is_none());
}
