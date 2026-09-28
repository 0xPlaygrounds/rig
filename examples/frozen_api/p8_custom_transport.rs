use bytes::Bytes;
use futures::stream;
use rig::driver::{Exchange, Model, Opened, Opening, Transport};
use rig::providers::openai::{self, OpenAIConfig};
use rig::wire::{Encoded, Wire, WireFrame};

/// Replays one recorded reply body for any HTTP wire, framed the way the
/// wire asked for it (SSE, NDJSON or one whole document).
#[derive(Clone)]
struct Replay {
    body: Bytes,
}

impl<W> Transport<W> for Replay
where
    W: Wire<Payload = Encoded, Frame = WireFrame>,
{
    fn send(&self, payload: Encoded, _exchange: Exchange) -> Opening<WireFrame> {
        let frames = payload.framing.split(&self.body);
        Opening::ready(Opened::new(stream::iter(frames.into_iter().map(Ok))))
    }
}

const RECORDED: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-5.2","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wire = OpenAIConfig::new("unused").chat(openai::GPT_5_2);
    let model = Model::new(
        wire,
        Replay {
            body: Bytes::from_static(RECORDED.as_bytes()),
        },
    );

    let response = model.call("hi").await?;
    assert_eq!(response.text(), "hello");
    assert_eq!(response.usage.output_tokens, Some(1));
    Ok(())
}
