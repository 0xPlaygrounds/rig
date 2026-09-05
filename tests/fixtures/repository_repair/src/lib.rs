/// Return a borrowed page of at most `limit` items starting at `offset`.
/// Offsets beyond the end and zero limits return an empty page.
pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {
    let end = (offset + limit).min(items.len());
    &items[offset..end]
}
