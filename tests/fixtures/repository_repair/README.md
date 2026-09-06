# Page window

This dependency-free library selects one page from a slice. Run its tests with
`cargo test --offline --locked`.

`page(items, offset, limit)` returns up to `limit` consecutive items, starting at
`offset`. Preserve input order and return a borrowed slice. An offset at or past
the end returns an empty page, as does a zero limit. All `usize` offsets and
limits are valid; the function must not panic or wrap arithmetic at large values.

The tests currently fail. Repair the implementation while preserving this
contract. Add focused regression coverage in `tests/regression.rs` before
changing `src/lib.rs`. Keep the existing tests and package configuration intact.
