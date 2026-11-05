use super::*;
use std::collections::HashMap;
use vector_index::VectorSimilarityFunction;

fn index_info() -> FalkorIndex {
    FalkorIndex {
        entity_type: EntityType::Node,
        status: IndexStatus::Active,
        index_label: "Document".into(),
        fields: vec!["embedding".into()],
        field_types: HashMap::from([("embedding".into(), vec![IndexType::Vector])]),
        language: "english".into(),
        stopwords: vec![],
        info: HashMap::new(),
        options: HashMap::from([(
            "embedding".into(),
            FalkorValue::Map(HashMap::from([
                ("dimension".into(), FalkorValue::I64(3)),
                (
                    "similarityFunction".into(),
                    FalkorValue::String("euclidean".into()),
                ),
            ])),
        )]),
    }
}

#[test]
fn loads_index_metric_and_dimensions() -> anyhow::Result<()> {
    let (config, dimensions) = read_index(&[index_info()], "Document", "embedding", 3)?;
    anyhow::ensure!(dimensions == 3);
    anyhow::ensure!(config.similarity_function == VectorSimilarityFunction::Euclidean);
    anyhow::ensure!(read_index(&[index_info()], "Document", "embedding", 0).is_ok());
    Ok(())
}

#[test]
fn rejects_missing_pending_wrong_type_and_mismatched_indexes() {
    assert!(read_index(&[], "Document", "embedding", 3).is_err());
    assert!(read_index(&[index_info()], "Document", "other", 3).is_err());
    assert!(read_index(&[index_info()], "Document", "embedding", 2).is_err());
    let mut index = index_info();
    index.entity_type = EntityType::Edge;
    assert!(read_index(&[index], "Document", "embedding", 3).is_err());
    let mut index = index_info();
    index.status = IndexStatus::Pending;
    assert!(read_index(&[index], "Document", "embedding", 3).is_err());
    let mut index = index_info();
    index.field_types = HashMap::from([("embedding".into(), vec![IndexType::Range])]);
    assert!(read_index(&[index], "Document", "embedding", 3).is_err());
    let mut index = index_info();
    index.options.clear();
    assert!(read_index(&[index], "Document", "embedding", 3).is_err());
}

#[test]
fn json_conversion_preserves_types_and_rejects_integer_overflow() -> anyhow::Result<()> {
    let value = json_to_falkor(serde_json::json!({"a": [1, 1.5, true, null, "it's\\safe"]}))?;
    let decoded: Value = value.deserialize_into()?;
    anyhow::ensure!(decoded == serde_json::json!({"a": [1, 1.5, true, null, "it's\\safe"]}));
    anyhow::ensure!(json_to_falkor(serde_json::json!(u64::MAX)).is_err());
    Ok(())
}
