use super::stable_hash;

/// The hash is over the value, not the order its keys were inserted
/// in — the property `preserve_order` breaks for a raw serialization.
#[test]
fn the_hash_is_independent_of_key_order() {
    let mut ab = serde_json::Map::new();
    ab.insert("a".to_owned(), serde_json::json!(1));
    ab.insert("b".to_owned(), serde_json::json!({"d": 4, "c": 3}));
    let mut ba = serde_json::Map::new();
    ba.insert("b".to_owned(), serde_json::json!({"c": 3, "d": 4}));
    ba.insert("a".to_owned(), serde_json::json!(1));
    assert_eq!(
        stable_hash(&serde_json::Value::Object(ab)).expect("hashes"),
        stable_hash(&serde_json::Value::Object(ba)).expect("hashes")
    );
}

/// Struct fields are keys too: the hash is over the canonical form
/// of the whole value, so a struct hashes like the object it becomes.
#[test]
fn a_struct_hashes_as_its_canonical_object() {
    #[derive(serde::Serialize)]
    struct Spec {
        zeta: u8,
        alpha: &'static str,
    }
    assert_eq!(
        stable_hash(&Spec {
            zeta: 1,
            alpha: "x"
        })
        .expect("hashes"),
        stable_hash(&serde_json::json!({"alpha": "x", "zeta": 1})).expect("hashes")
    );
}
