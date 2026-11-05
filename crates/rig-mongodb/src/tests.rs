use super::{SearchIndex, VectorStoreError};
use mongodb::bson::{self, Bson, doc};

fn embedded_field(fields: Vec<Bson>) -> Result<String, VectorStoreError> {
    let index = bson::from_document::<SearchIndex>(doc! {
        "id": "index-1",
        "name": "vector_index",
        "type": "vectorSearch",
        "status": "READY",
        "queryable": true,
        "latestDefinition": { "fields": fields },
    })
    .map_err(VectorStoreError::datastore)?;
    index.latest_definition.embedded_field()
}

fn vector() -> Bson {
    doc! {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 3,
        "similarity": "cosine",
        "quantization": "scalar",
    }
    .into()
}

fn filter() -> Bson {
    doc! { "type": "filter", "path": "tenant" }.into()
}

#[test]
fn vector_only_index_selects_embedding_path() {
    assert_eq!(
        embedded_field(vec![vector()]).ok().as_deref(),
        Some("embedding")
    );
}

#[test]
fn filter_after_vector_does_not_require_vector_options() {
    assert_eq!(
        embedded_field(vec![vector(), filter()]).ok().as_deref(),
        Some("embedding")
    );
}

#[test]
fn filter_before_vector_is_not_selected_as_embedding_path() {
    assert_eq!(
        embedded_field(vec![filter(), vector()]).ok().as_deref(),
        Some("embedding")
    );
}

#[test]
fn unknown_field_types_do_not_hide_vector_field() {
    assert_eq!(
        embedded_field(vec![
            doc! { "type": "future", "path": "other", "setting": true }.into(),
            vector(),
        ])
        .ok()
        .as_deref(),
        Some("embedding")
    );
}

#[test]
fn index_without_vector_field_returns_explicit_error() {
    for fields in [vec![], vec![filter()]] {
        let result = embedded_field(fields);
        assert!(matches!(
            result,
            Err(VectorStoreError::DatastoreError(error))
                if error.to_string() == "No vector fields found"
        ));
    }
}
