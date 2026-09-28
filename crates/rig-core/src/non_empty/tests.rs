use super::{Empty, NonEmpty};

#[test]
fn an_empty_list_is_refused() {
    assert_eq!(NonEmpty::<u8>::from_vec(Vec::new()), Err(Empty));
    assert!(serde_json::from_str::<NonEmpty<u8>>("[]").is_err());
}

#[test]
fn it_round_trips_as_a_plain_list() {
    let items = NonEmpty::with_rest(1, [2, 3]);
    let json = serde_json::to_string(&items).expect("serializes");
    assert_eq!(json, "[1,2,3]");
    assert_eq!(
        serde_json::from_str::<NonEmpty<u8>>(&json).expect("parses"),
        items
    );
}

#[test]
fn filter_never_yields_an_empty_list() {
    let items = NonEmpty::with_rest(1, [2, 3]);
    assert_eq!(
        items
            .clone()
            .filter(|item| *item > 1)
            .map(NonEmpty::into_vec),
        Some(vec![2, 3])
    );
    assert_eq!(items.filter(|item| *item > 3), None);
}

#[test]
fn first_and_last_need_no_option() {
    let mut items = NonEmpty::new("a");
    items.push("b");
    assert_eq!((*items.first(), *items.last()), ("a", "b"));
}
