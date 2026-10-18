use super::LSHIndex;

#[test]
fn remove_undoes_insert() {
    let mut index = LSHIndex::new(3, 4, 8);
    index.insert("a", &[1.0, 0.5, 0.0]);
    index.insert("b", &[1.0, 0.5, 0.0]);
    index.remove("a", &[1.0, 0.5, 0.0]);
    assert_eq!(index.query(&[1.0, 0.5, 0.0]), vec!["b"]);
    index.remove("b", &[1.0, 0.5, 0.0]);
    assert!(index.tables.iter().all(|table| table.is_empty()));
}
