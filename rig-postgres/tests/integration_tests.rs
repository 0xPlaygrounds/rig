use sqlx::postgres::PgPoolOptions;
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};

const POSTGRES_PORT: u16 = 5432;

#[tokio::test]
async fn vector_search_test() {
    // Setup a local postgres container for testing. NOTE: docker service must be running.
    let container = GenericImage::new("pgvector/pgvector", "pg17")
        .with_wait_for(WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .with_exposed_port(POSTGRES_PORT.tcp())
        .with_env_var("POSTGRES_USER", "postgres")
        .with_env_var("POSTGRES_PASSWORD", "postgres")
        .with_env_var("POSTGRES_DB", "rig")
        .start()
        .await
        .expect("Failed to start postgres with pgvector container");

    let host = container.get_host().await.unwrap().to_string();
    let port = container
        .get_host_port_ipv4(POSTGRES_PORT)
        .await
        .expect("Error getting docker port");

    println!("Container started on host:port {}:{}", host, port);

    // connect to Postgres
    let pg_pool = PgPoolOptions::new()
        .max_connections(50)
        .idle_timeout(std::time::Duration::from_secs(5))
        .connect(&format!(
            "postgres://postgres:postgres@{}:{}/rig",
            host, port
        ))
        .await
        .expect("Failed to create postgres pool");

    // run migrations on Postgres
    sqlx::migrate!("./tests/migrations")
        .run(&pg_pool)
        .await
        .expect("Failed to run migrations");

    println!("Connected to postgres");
}
