use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_qdrant::QdrantFilter;

#[test]
fn qdrant_filter_is_available_from_crate_root() {
    let request = VectorSearchRequest::<QdrantFilter>::builder()
        .query("search text")
        .samples(3)
        .filter(QdrantFilter::eq("document_id", serde_json::json!("doc-1")))
        .build();

    assert!(request.filter().is_some());
}
