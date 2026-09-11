//! Construct a real Gemini provider with an in-memory HTTP transport.
//! This exercises provider request/response conversion without credentials or
//! network access. Replace the transport in an application to use live IO.

use std::time::Duration;

use bevy_app::{App, Update};
use rig_core::{
    client::CompletionClient, providers::gemini, serve::adapters::CompletionAdapter,
    test_utils::RecordingHttpClient,
};
use rig_ecs::{
    bus::{Handlers, run_to_quiescence},
    commands::{Agent, Prompt, install},
    inspect::inspect,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let http = RecordingHttpClient::new(
        r#"{"candidates":[{"content":{"parts":[{"text":"Hello from Gemini."}],"role":"model"},"finishReason":"STOP"}]}"#,
    );
    let client = gemini::Client::builder()
        .api_key("unused-by-in-memory-transport")
        .http_client(http)
        .build()?;
    let model = client.completion_model("gemini-2.5-flash");

    let mut app = App::new();
    install(app.world_mut(), Default::default())?;
    app.add_systems(Update, run_to_quiescence);
    let world = app.world_mut();
    let model = Handlers::register_in(world, "chat", CompletionAdapter::new("chat", model))?;
    let agent = Agent::new(model).preamble("Be concise.").spawn(world)?;
    let run = Prompt::new(agent, "Hello").spawn(world)?;

    for _ in 0..5_000 {
        app.update();
        let run = inspect(app.world(), run)?;
        if let Some(failure) = run.failure() {
            return Err(format!("provider example failed: {failure:?}").into());
        }
        if let Some(answer) = run.answer() {
            if answer != "Hello from Gemini." {
                return Err(format!("unexpected in-memory response: {answer}").into());
            }
            println!("{answer}");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Err("provider example did not complete within 5,000 host ticks".into())
}
