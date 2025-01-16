<div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source srcset="https://www.postgresql.org/media/img/about/press/elephant.png">
        <img src="https://www.postgresql.org/media/img/about/press/elephant.png" width="200" alt="Postgres logo">
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

## PostgreSQL usage

The crate utilizes the [pgvector](https://github.com/pgvector/pgvector) extension, which is available for PostgreSQL version 13 and later. Use any of the [official](https://www.postgresql.org/download/) or alternative methods to install psql. The `pgvector` extension will be automatically installed by the crate if it's not present yet.

The crate relies on [`tokio-postgres`](https://docs.rs/tokio-postgres/latest/tokio_postgres/index.html) to manage its communication with the database. You can connect to a DB using any of the [supported methods](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING). See the [`/examples`](./examples) folder for usage examples.
