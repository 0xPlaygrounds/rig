//! Runnable test-support consumer; invoke with `cargo run -p rig --example ecs-consumer -- plan`.

#[path = "../common/cassettes.rs"]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unreachable,
    reason = "reuse the repository's assertion-based test cassette engine"
)]
mod cassettes;
#[path = "mod.rs"]
mod consumer;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    // Cassette assertions can contain raw wire payloads. The runner reports
    // the selected case without exposing a panic's request/header values.
    std::panic::set_hook(Box::new(|_| {
        eprintln!("consumer task panicked; see the selected case in the matrix report");
    }));
    match consumer::runner::run(std::env::args().skip(1)).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let message = error.to_string();
            if consumer::artifacts::validate_text(std::path::Path::new("stderr"), &message).is_ok()
            {
                eprintln!("{message}");
            } else {
                eprintln!("consumer failed; diagnostic withheld by artifact redaction checks");
            }
            std::process::ExitCode::FAILURE
        }
    }
}
