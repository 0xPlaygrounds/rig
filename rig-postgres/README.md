<div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://www.postgresql.org/media/img/about/press/elephant.png">
        <source media="(prefers-color-scheme: light)" srcset="https://www.postgresql.org/media/img/about/press/elephant.png">
        <img src="https://www.postgresql.org/media/img/about/press/elephant.png" width="200" alt="SQLite logo">
    </picture>
</div>

<br><br>

## Rig-postgres

This companion crate implements a Rig vector store based on PostgreSQL.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-core = "0.4.0"
rig-postgres = "0.1.0"
```

You can also run `cargo add rig-core rig-postgres` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Important Note

The crate utilizes the [pgvector](https://github.com/pgvector/pgvector) PostgreSQL extension. It will automatically install it in your database if it's not yet present.
