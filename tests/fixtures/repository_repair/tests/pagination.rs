use page_window::page;

#[test]
fn ordinary_pages_preserve_order() {
    assert_eq!(page(&[10, 20, 30, 40], 1, 2), &[20, 30]);
    assert_eq!(page(&[10, 20, 30, 40], 3, 2), &[40]);
}

#[test]
fn zero_limit_returns_no_items() {
    assert!(page(&[10, 20, 30], 1, 0).is_empty());
}

#[test]
fn an_offset_past_the_end_returns_an_empty_page() {
    assert!(page(&[10, 20, 30], 5, 2).is_empty());
}
