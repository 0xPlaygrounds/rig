//! Host construction, strict resume and effect replay without provider recipes.
//! Uses a real OpenAI codec with an explicitly offline recording transport; no
//! credentials are read and no network request is made. A live host supplies its
//! own client/credential/runtime policy at the same construction boundary.

use bevy_app::App;
use rig_cassette::{
    ecs::{EffectLogResource, Replay, ReplayPlugin},
    effect_log::EffectLog,
};
use rig_core::{
    completion::CompletionRequestBuilder,
    driver::Bind,
    effect::EffectKind,
    error::{ErrorKind, ErrorReport},
    providers::openai::wire::OpenAI,
    serve::{ErasedHandler, adapters::CompletionAdapter},
    test_utils::RecordingHttpClient,
};
use rig_ecs::{
    bus::{EffectOutcome, Handlers, PendingEffect},
    checkpoint::{RestoreMode, load_world, save_world},
};

const KEY: &str = "model";
const BODY: &str = r#"{"id":"demo","object":"chat.completion","created":0,"model":"demo","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((rig_ecs::RigPlugin::default(), ReplayPlugin));
    app.finish();
    app.cleanup();
    app
}

fn assemble() -> ErasedHandler {
    let model = OpenAI::new("demonstration-only")
        .with_base_url("http://offline.invalid/v1")
        .bind(RecordingHttpClient::new(BODY))
        .chat("demo");
    ErasedHandler::new(CompletionAdapter::new("demo", model))
}

fn finish(app: &mut App) -> Result<EffectLog, ErrorReport> {
    let started = std::time::Instant::now();
    loop {
        app.update();
        if let Some(outcome) = app
            .world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .next()
        {
            outcome.0.as_ref().map_err(Clone::clone)?;
            return Ok(app.world().resource::<EffectLogResource>().log());
        }
        if started.elapsed() > std::time::Duration::from_secs(5) {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                "offline example did not finish",
            ));
        }
        std::thread::yield_now();
    }
}

fn main() -> Result<(), ErrorReport> {
    let mut live = app();
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_erased(KEY, assemble())
    })??;
    live.world_mut().spawn(PendingEffect::new(
        KEY,
        EffectKind::Completion {
            request: CompletionRequestBuilder::unbound("hello").build(),
            stream: false,
        },
    ));
    let checkpoint = save_world(live.world_mut())?;
    EffectLogResource::install(live.world_mut(), Default::default());
    let log = finish(&mut live)?;
    drop(live);

    let mut resumed = app();
    checkpoint.validate(resumed.world())?;
    let handler = assemble(); // Reapply host settings, not saved connection data.
    load_world(
        &checkpoint,
        resumed.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), handler)],
    )?;
    EffectLogResource::install(resumed.world_mut(), Default::default());
    finish(&mut resumed)?;
    drop(resumed);

    // Replay branches before assemble: no live factory, credential lookup or
    // diagnostic-secret collection. The recorded implementation serves the key.
    let mut replay = app();
    Replay::default().register(replay.world_mut(), &log)?;
    load_world(&checkpoint, replay.world_mut(), RestoreMode::Strict, [])?;
    EffectLogResource::install(replay.world_mut(), Default::default());
    finish(&mut replay)?;
    println!("live, strict resume and recorded replay completed");
    Ok(())
}
