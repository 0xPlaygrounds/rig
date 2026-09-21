//! Host assembly of an SDK-backed model: no core provider reference or ECS materializer.
//!
//! Requires Google Application Default Credentials and GOOGLE_CLOUD_PROJECT.
//! This example demonstrates construction/registration only; it sends no completion.
//! A live host must keep its Tokio runtime alive and provide that runtime's context
//! when polling SDK requests on ECS worker threads. Vertex currently supports unary
//! completions only: do not request streaming from this model.
//!
//! Live resume reconstructs the model from host launch configuration and calls
//! `Handlers::restore_erased` against the checkpoint's saved key instead of ordinary
//! registration. Replay installs replay handlers and never constructs this client.

use rig_core::{
    driver::CompletionProvider,
    serve::{ErasedHandler, adapters::CompletionAdapter},
};
use rig_ecs::bus::{BusPlugin, Handlers};
use rig_vertexai::{Client, completion::GEMINI_2_5_FLASH};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::from_env()?;
    // SDK initialization belongs to the host, outside any ECS system borrow.
    client.inner().await?;
    let model = client.completion(GEMINI_2_5_FLASH);
    let handler = ErasedHandler::new(CompletionAdapter::new(GEMINI_2_5_FLASH, model));
    let mut app = bevy_app::App::new();
    app.add_plugins(BusPlugin::default());
    Handlers::with(app.world_mut(), |handlers| {
        handlers.register_erased("model:default", handler)
    })??;
    Ok(())
}
