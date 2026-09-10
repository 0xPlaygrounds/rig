use super::*;

#[test]
fn node_label_defaults_to_none_and_builder_sets_it() {
    assert_eq!(IndexConfig::new("idx").node_label, None);
    assert_eq!(
        IndexConfig::new("idx")
            .node_label("Movie")
            .node_label
            .as_deref(),
        Some("Movie"),
    );
}

#[test]
fn insert_documents_query_uses_label_else_default() {
    assert_eq!(
        insert_documents_query("Movie"),
        "UNWIND $items AS item CREATE (n:Movie) SET n = item",
    );
    assert_eq!(
        insert_documents_query(DEFAULT_NODE_LABEL),
        "UNWIND $items AS item CREATE (n:Document) SET n = item",
    );
}
