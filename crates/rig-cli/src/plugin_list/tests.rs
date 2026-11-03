use super::*;

#[test]
fn the_key_of_a_list_is_stable_and_tells_lists_apart() {
    let one = ["a::A".to_owned(), "b::B".to_owned()];
    let other = ["b::B".to_owned(), "a::A".to_owned()];
    assert_eq!(key(&one), key(&one.clone()));
    assert_ne!(key(&one), key(&other));
    assert_eq!(key(&one).len(), 16);
}
