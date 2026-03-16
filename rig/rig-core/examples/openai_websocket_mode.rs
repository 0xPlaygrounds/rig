use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openai;
use rig::providers::openai::responses_api::streaming::{ItemChunkKind, ResponseChunkKind};
use rig::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let client = openai::Client::from_env();
    let model_name = openai::GPT_4O_MINI;
    let model = client.completion_model(model_name);
    let mut session = client.responses_websocket(model_name).await?;

    let warmup_request = model
        .completion_request("You will answer a follow-up question about websocket mode.")
        .preamble("Be precise and concise.".to_string())
        .build();

    let warmup_id = session.warmup(warmup_request).await?;
    println!("Warmup response id: {warmup_id}");

    let request = model
        .completion_request("Explain the benefit of websocket mode in one sentence.")
        .build();

    session.send(request).await?;

    loop {
        let event = session.next_event().await?;
        match event {
            ResponsesWebSocketEvent::Item(item) => {
                if let ItemChunkKind::OutputTextDelta(delta) = item.data {
                    print!("{}", delta.delta);
                }
            }
            ResponsesWebSocketEvent::Response(chunk) => {
                println!("\nresponse event: {:?}", chunk.kind);
                if matches!(
                    chunk.kind,
                    ResponseChunkKind::ResponseCompleted
                        | ResponseChunkKind::ResponseFailed
                        | ResponseChunkKind::ResponseIncomplete
                ) {
                    break;
                }
            }
            ResponsesWebSocketEvent::Done(done) => {
                println!("\nresponse.done id={:?}", done.response_id());
            }
            ResponsesWebSocketEvent::Error(error) => {
                return Err(anyhow::anyhow!(error.to_string()));
            }
        }
    }

    let chained_request = model
        .completion_request("Now restate that as three very short bullet points.")
        .build();
    let response = session.completion(chained_request).await?;

    println!("Chained response: {:?}", response.choice);
    session.close().await?;

    Ok(())
}
