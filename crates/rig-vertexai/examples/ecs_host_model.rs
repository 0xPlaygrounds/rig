//! Live, opt-in Vertex completion through ECS (requires ADC and GOOGLE_CLOUD_PROJECT).
//! Running this example invokes a billable service; compiling it does not.
//! The host retains and drives the SDK/credential runtime while ECS owns each
//! operation future. No provider construction data is inserted into the world.

use bevy_app::App;
use rig_core::{
    completion::CompletionRequestBuilder,
    driver::CompletionProvider,
    effect::{EffectKind, HandlerDescriptor, family},
    serve::{Dispatch, ErasedHandler, Reply, Serve, adapters::CompletionAdapter},
};
use rig_ecs::bus::{EffectOutcome, Handlers, PendingEffect};

// A host-local adapter, not another scheduler. Enter on every poll, including
// polls performed by ECS workers. No detached operation task is spawned.
struct Hosted {
    handler: ErasedHandler,
    runtime: tokio::runtime::Handle,
}

impl Serve for Hosted {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        self.handler.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let mut operation = self.handler.handle(kind, dispatch);
        std::future::poll_fn(|cx| {
            let _entered = self.runtime.enter();
            operation.as_mut().poll(cx)
        })
        .await
    }
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    let client = {
        let _entered = runtime.enter();
        rig_vertexai::Client::from_env()?
    };
    // Complete SDK preparation before borrowing the execution world. A host
    // with an already-prepared PredictionService can inject that instead.
    runtime.block_on(client.inner())?;
    let model = client.completion(rig_vertexai::completion::GEMINI_2_5_FLASH_LITE);
    let handler = Hosted {
        handler: ErasedHandler::new(CompletionAdapter::new("vertex", model)),
        runtime: runtime.handle().clone(),
    };
    let mut app = App::new();
    app.add_plugins(rig_ecs::RigPlugin::default());
    app.finish();
    app.cleanup();
    Handlers::with(app.world_mut(), |h| h.register("model", handler))??;
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: CompletionRequestBuilder::unbound("Say hello briefly.").build(),
                stream: false, // Vertex streaming is explicitly unsupported.
            },
        ))
        .id();
    let started = std::time::Instant::now();
    let result = loop {
        app.update();
        if let Some(outcome) = app.world().get::<EffectOutcome>(effect) {
            break outcome.0.as_ref().map(|_| ()).map_err(Clone::clone);
        }
        if started.elapsed() > std::time::Duration::from_secs(60) {
            break Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Internal,
                "host deadline exceeded",
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    // Stop admitting work; drop the world/operations, then the shared client,
    // then its runtime. Cancellation cannot undo a remotely accepted request.
    drop(app);
    drop(client);
    drop(runtime);
    result?;
    println!("Vertex completion finished through ECS");
    Ok(())
}
