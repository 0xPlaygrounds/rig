use super::*;

#[test]
fn an_empty_list_is_refused_by_every_constructor_and_by_deserialize() {
    assert!(NonEmpty::<u8>::from_vec(Vec::new()).is_none());
    assert_eq!(NonEmpty::<u8>::try_from(Vec::new()), Err(Empty));
    let error = serde_json::from_str::<NonEmpty<u8>>("[]").expect_err("an empty array is refused");
    assert!(error.to_string().contains("at least one item"), "{error}");
}

#[test]
fn a_list_serializes_as_a_plain_array() {
    let items = NonEmpty::of(1, [2, 3]);
    let encoded = serde_json::to_string(&items).expect("serializes");
    assert_eq!(encoded, "[1,2,3]");
    let decoded: NonEmpty<i32> = serde_json::from_str(&encoded).expect("loads");
    assert_eq!(decoded, items);
}

#[test]
fn retain_reports_a_list_it_emptied() {
    let items = NonEmpty::of(1, [2, 3]);
    assert_eq!(
        items.clone().retain(|item| *item > 1),
        Some(NonEmpty::of(2, [3]))
    );
    assert_eq!(items.retain(|_| false), None);
}
