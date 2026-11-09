use super::LSHIndex;

#[test]
fn default_index_keeps_ids_until_removed_or_cleared() {
    let mut index = LSHIndex::default();
    index.insert("a", &[1.0, 0.0]);
    index.insert("b", &[0.0, 1.0]);
    let mut ids = index.query(&[1.0, 0.0]);
    ids.sort_unstable();
    assert_eq!(ids, ["a", "b"]);

    index.remove("a", &[1.0, 0.0]);
    assert_eq!(index.query(&[0.0, 1.0]), ["b"]);
    index.clear();
    assert!(index.query(&[0.0, 1.0]).is_empty());

    index.insert("c", &[1.0, 1.0]);
    assert_eq!(index.query(&[1.0, 1.0]), ["c"]);
}

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
