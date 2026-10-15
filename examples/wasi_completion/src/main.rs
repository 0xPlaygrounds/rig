//! An agent prompt streamed inside a WASI 0.2 component, over `wasi:http`.
//!
//! Requires `OPENAI_API_KEY`. Build it for WASI and run it under wasmtime:
//!
//! ```text
//! cargo build -p wasi_completion --target wasm32-wasip2
//! wasmtime run -S http --env OPENAI_API_KEY target/wasm32-wasip2/debug/wasi_completion.wasm
//! ```

#[cfg(target_os = "wasi")]
#[wstd::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;
    use rig::http_client::WstdClient;
    use rig::prelude::*;
    use rig::providers::openai::{self, OpenAIConfig};
    use rig::streaming::{Item, StreamEvent};

    let openai = OpenAIConfig::from_env()?.connect(WstdClient::default());
    let agent = AgentBuilder::new(openai.completion(openai::GPT_4O))
        .preamble("Answer in two short sentences.")
        .build();

    let mut stream = agent.prompt("What is WebAssembly?").stream();
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            })) => print!("{text}"),
            MultiTurnStreamItem::FinalResponse(_) => println!(),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(not(target_os = "wasi"))]
fn main() {
    eprintln!("build this example for wasm32-wasip2 and run it under wasmtime; see its docs");
}
