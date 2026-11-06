use super::SearchIndex;
use mongodb::bson::{self, doc};

/// Atlas lists `filter` fields with only a type and a path, in any order
/// relative to the vector field.
#[test]
fn index_with_filter_fields_selects_its_vector_field() -> anyhow::Result<()> {
    let index: SearchIndex = bson::from_document(doc! {
        "id": "index-1",
        "name": "vector_index",
        "type": "vectorSearch",
        "status": "READY",
        "queryable": true,
        "latestDefinition": { "fields": [
            { "type": "filter", "path": "tenant" },
            { "type": "vector", "path": "embedding", "numDimensions": 3, "similarity": "cosine" },
        ] },
    })?;
    let path = index.latest_definition.vector_path()?;
    anyhow::ensure!(path == "embedding", "selected {path}");
    Ok(())
}
