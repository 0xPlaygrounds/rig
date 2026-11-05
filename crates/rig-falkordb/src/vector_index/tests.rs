use super::*;
use rig_core::vector_store::request::SearchFilter;
use serde_json::json;

#[test]
fn configuration_defaults_and_identifier_validation() -> anyhow::Result<()> {
    let config = IndexConfig::default();
    anyhow::ensure!(config.node_label == "Document");
    anyhow::ensure!(config.embedding_property == "embedding");
    anyhow::ensure!(config.similarity_function == VectorSimilarityFunction::Cosine);
    anyhow::ensure!(
        insert_query(&IndexConfig::new("Movie plots").embedding_property("plot vector"))?
            == "UNWIND $items AS item CREATE (node:`Movie plots`) SET node = item.properties SET node.`plot vector` = vecf32(item.embedding)"
    );
    anyhow::ensure!(IndexConfig::new("a`b").validate().is_err());
    anyhow::ensure!(
        IndexConfig::new("Movie")
            .embedding_property("")
            .validate()
            .is_err()
    );
    Ok(())
}

#[test]
fn queries_use_similarity_thresholds_and_bound_filters() -> anyhow::Result<()> {
    let req = VectorSearchRequest::builder()
        .query("find")
        .samples(3)
        .threshold(0.8)
        .filter(FalkorDBSearchFilter::eq("category", json!("manual")))
        .build();
    let query = search_query(&IndexConfig::default(), &req, true)?;
    anyhow::ensure!(query.text.contains("WITH node, 1.0 - score AS similarity"));
    anyhow::ensure!(
        query
            .text
            .contains("WHERE similarity >= $threshold AND (node.`category` = $filter_0)")
    );
    anyhow::ensure!(
        query
            .text
            .contains("node {.*, `embedding`: null} AS document")
    );
    anyhow::ensure!(query.text.contains("ORDER BY score DESC, element_id ASC"));
    anyhow::ensure!(
        query.params
            == vec![
                ("threshold".into(), FalkorValue::F64(0.8)),
                ("filter_0".into(), FalkorValue::String("manual".into())),
            ]
    );
    let query = search_query(
        &IndexConfig::default().similarity_function(VectorSimilarityFunction::Euclidean),
        &req,
        false,
    )?;
    anyhow::ensure!(query.text.contains("1.0 / (1.0 + score) AS similarity"));
    anyhow::ensure!(!query.text.contains("AS document"));
    Ok(())
}

#[test]
fn sample_count_cannot_wrap_to_a_signed_integer() {
    let req = VectorSearchRequest::builder()
        .query("find")
        .samples(u64::MAX)
        .build();
    assert!(matches!(
        search_query(&IndexConfig::default(), &req, false),
        Err(VectorStoreError::SamplesOutOfRange {
            requested: u64::MAX,
            ..
        })
    ));
}

#[test]
fn document_preparation_preserves_fields_and_multiple_embeddings() -> anyhow::Result<()> {
    let embeddings = vec![
        Embedding {
            document: "first".into(),
            vec: vec![1.0, 0.0],
        },
        Embedding {
            document: "second".into(),
            vec: vec![0.0, 1.0],
        },
    ];
    let records = prepare_documents(
        vec![(
            json!({"id": "doc", "rating": 42, "active": true}),
            embeddings,
        )],
        2,
        "embedding",
    )?;
    anyhow::ensure!(records.len() == 2);
    let record = records
        .first()
        .and_then(FalkorValue::as_map)
        .ok_or_else(|| anyhow::anyhow!("missing record map"))?;
    let properties = record
        .get("properties")
        .and_then(FalkorValue::as_map)
        .ok_or_else(|| anyhow::anyhow!("missing properties"))?;
    anyhow::ensure!(properties.get("id") == Some(&FalkorValue::String("doc".into())));
    anyhow::ensure!(properties.get("rating") == Some(&FalkorValue::I64(42)));
    anyhow::ensure!(properties.get("active") == Some(&FalkorValue::Bool(true)));
    anyhow::ensure!(properties.get("embedded_text") == Some(&FalkorValue::String("first".into())));
    anyhow::ensure!(
        record.get("embedding")
            == Some(&FalkorValue::Array(vec![
                FalkorValue::F64(1.0),
                FalkorValue::F64(0.0),
            ]))
    );
    Ok(())
}

#[test]
fn scalar_documents_and_empty_input_are_supported() -> anyhow::Result<()> {
    let records = prepare_documents(
        vec![(
            "hello",
            vec![Embedding {
                document: "hello".into(),
                vec: vec![1.0],
            }],
        )],
        1,
        "embedding",
    )?;
    let record = records
        .first()
        .and_then(FalkorValue::as_map)
        .ok_or_else(|| anyhow::anyhow!("missing record"))?;
    let properties = record
        .get("properties")
        .and_then(FalkorValue::as_map)
        .ok_or_else(|| anyhow::anyhow!("missing properties"))?;
    anyhow::ensure!(properties.get("document") == Some(&FalkorValue::String("hello".into())));
    anyhow::ensure!(prepare_documents::<String>(vec![], 1, "embedding")?.is_empty());
    Ok(())
}

#[test]
fn wrong_embedding_dimensions_are_rejected() {
    assert!(
        prepare_documents(
            vec![(
                "doc",
                vec![Embedding {
                    document: "doc".into(),
                    vec: vec![1.0]
                },]
            )],
            2,
            "embedding"
        )
        .is_err()
    );
}
