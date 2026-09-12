//! Migrated completion adapters preserve recorded parser output under direct polling.
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
use bytes::Bytes;
use futures::{Stream, StreamExt};
use rig_core::http_client::{Request, Response, StatusCode};
use rig_core::{
    client::CompletionClient,
    completion::CompletionModel,
    http_client::{
        self, HttpClientExt, LazyBody, MultipartForm, StreamingResponse, sse::BoxedStream,
    },
    providers::{anthropic, deepseek, openai},
    streaming::{StreamEvent, StreamEvents},
    wasm_compat::WasmCompatSend,
};
use std::{
    pin::Pin,
    task::{Context, Poll, Waker},
};

#[derive(Clone)]
struct Replay {
    body: Bytes,
    chunk: usize,
    yielding: bool,
}
struct Chunks {
    replay: Replay,
    offset: usize,
    pending: bool,
}
impl Stream for Chunks {
    type Item = Result<Bytes, http_client::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.offset == self.replay.body.len() {
            return Poll::Ready(None);
        }
        if self.replay.yielding && !self.pending {
            self.pending = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        self.pending = false;
        let end = (self.offset + self.replay.chunk).min(self.replay.body.len());
        let bytes = self.replay.body.slice(self.offset..end);
        self.offset = end;
        Poll::Ready(Some(Ok(bytes)))
    }
}
impl HttpClientExt for Replay {
    fn send<T, U>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::non_success_with_details(
            StatusCode::NOT_IMPLEMENTED,
            rig::http_client::HeaderMap::new(),
            String::new(),
        )))
    }
    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::non_success_with_details(
            StatusCode::NOT_IMPLEMENTED,
            rig::http_client::HeaderMap::new(),
            String::new(),
        )))
    }
    fn send_streaming<T>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let body: BoxedStream = Box::pin(Chunks {
            replay: self.clone(),
            offset: 0,
            pending: false,
        });
        std::future::ready(
            Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(body)
                .map_err(http_client::Error::Protocol),
        )
    }
}
fn body(path: &str) -> Bytes {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cassettes")
        .join(path);
    let yaml = std::fs::read_to_string(path).unwrap();
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    Bytes::from(doc["then"]["body"].as_str().unwrap().to_owned())
}
async fn adapted<M: CompletionModel + Clone + 'static>(model: M, direct: bool) -> StreamEvents {
    let request = model
        .completion_request("response parsing probe")
        .max_tokens(1024)
        .build();
    if direct {
        Box::pin(model.stream(request).await.unwrap())
    } else {
        let handler = rig_core::serve::ErasedHandler::new(
            rig_core::serve::adapters::CompletionAdapter::new("probe", model),
        );
        handler
            .handle(
                rig_core::effect::EffectKind::Completion {
                    request,
                    stream: true,
                },
                rig_core::serve::Dispatch::new(rig_core::effect::EffectId::from_raw(0), true),
            )
            .await
            .into_stream()
    }
}

async fn stream(provider: &str, http: Replay, direct: bool) -> StreamEvents {
    match provider {
        "openai" => {
            adapted(
                openai::Client::new_with("test-not-a-key", http)
                    .unwrap()
                    .completions_api()
                    .completion_model("gpt-4o"),
                direct,
            )
            .await
        }
        "deepseek" => {
            adapted(
                deepseek::Client::new_with("test-not-a-key", http)
                    .unwrap()
                    .completion_model("deepseek-reasoner"),
                direct,
            )
            .await
        }
        "anthropic" => {
            adapted(
                anthropic::Client::new_with("test-not-a-key", http)
                    .unwrap()
                    .completion_model("claude-sonnet-4-5"),
                direct,
            )
            .await
        }
        _ => unreachable!(),
    }
}

#[test]
fn recorded_streams_equal_async_consumption_across_chunking_and_pending() {
    let cases = [
        (
            "openai",
            "openai/chat_tool_lifecycle_matrix/streaming_gpt4o_nested_model.yaml",
        ),
        (
            "openai",
            "openai/streaming_grammar_chat/long_text_stream.yaml",
        ),
        (
            "deepseek",
            "deepseek/raw_stream_capture_matrix/stream_reasoning_raw_round_trips_terminal_type.yaml",
        ),
        (
            "anthropic",
            "anthropic/raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream.yaml",
        ),
    ];
    for (provider, path) in cases {
        let bytes = body(path);
        let reference = futures::executor::block_on(async {
            stream(
                provider,
                Replay {
                    body: bytes.clone(),
                    chunk: usize::MAX,
                    yielding: false,
                },
                true,
            )
            .await
            .collect::<Vec<_>>()
            .await
        });
        assert!(reference.iter().all(Result::is_ok), "{path}: {reference:?}");
        assert!(
            reference
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Final(_))))
        );
        for chunk in [64, 4096, usize::MAX] {
            for yielding in [false, true] {
                let mut s = futures::executor::block_on(stream(
                    provider,
                    Replay {
                        body: bytes.clone(),
                        chunk,
                        yielding,
                    },
                    false,
                ));
                let mut actual = Vec::new();
                let mut pending = 0;
                let mut polls = 0;
                loop {
                    polls += 1;
                    assert!(polls < 1_000_000, "no progress: {path}");
                    let result =
                        Pin::new(&mut s).poll_next(&mut Context::from_waker(Waker::noop()));
                    match result {
                        Poll::Pending => pending += 1,
                        Poll::Ready(Some(item)) => actual.push(item),
                        Poll::Ready(None) => break,
                    }
                }
                assert_eq!(
                    serde_json::to_value(&actual).unwrap(),
                    serde_json::to_value(&reference).unwrap(),
                    "{path} chunk={chunk} yielding={yielding}"
                );
                if yielding {
                    assert!(pending > 0, "injected Pending must reach the consumer");
                }
                let mut expected_fold = rig_core::serve::StreamTap::new();
                let mut actual_fold = rig_core::serve::StreamTap::new();
                assert_eq!(
                    serde_json::to_value(
                        reference
                            .iter()
                            .find_map(|item| expected_fold.observe(item))
                    )
                    .unwrap(),
                    serde_json::to_value(actual.iter().find_map(|item| actual_fold.observe(item)))
                        .unwrap()
                );
            }
        }
    }
}
