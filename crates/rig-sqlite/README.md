<div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://www.sqlite.org/images/sqlite370_banner.gif">
        <source media="(prefers-color-scheme: light)" srcset="https://www.sqlite.org/images/sqlite370_banner.gif">
        <img src="https://www.sqlite.org/images/sqlite370_banner.gif" width="200" alt="SQLite logo">
    </picture>
</div>

<br><br>

## Rig-SQLite

This companion crate implements a Rig vector store based on SQLite.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-sqlite = "0.2.6"
rig-core = "0.37.0"
```

You can also run `cargo add rig-sqlite rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Important Note

Before using the SQLite vector store, you must [initialize the SQLite vector extension](https://alexgarcia.xyz/sqlite-vec/rust.html). Add this code before creating your connection:

```rust
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

unsafe {
    sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
}
```

## Filtering JSON Metadata

SQLite filters can target document-table columns that store JSON text. Use
SQLite's JSON extraction operators in the filter key:

```rust
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_sqlite::SqliteSearchFilter;

let req = VectorSearchRequest::builder()
    .query("release notes")
    .samples(5)
    .filter(SqliteSearchFilter::eq(
        "metadata->>'$.source'",
        serde_json::json!("docs"),
    ))
    .build();
```

Use `->>` when you want SQLite to compare a JSON value as a SQL scalar, such as
text, number, or boolean. Use `->` when you want to compare against JSON text;
Rig serializes the right-hand side as JSON for that form.

JSON metadata filters are applied after sqlite-vec candidate search because
they reference the document table, not sqlite-vec metadata columns. This keeps
results correct, including with `samples(1)`, but requires exhaustive candidate
retrieval. For frequently-used scalar filters, prefer storing the value in a
regular column marked with `Column::indexed()` so sqlite-vec can apply it during
candidate search.
