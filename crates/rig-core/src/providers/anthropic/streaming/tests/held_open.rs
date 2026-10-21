use super::*;
use crate::http_client::{self, HttpClientExt, LazyBody, MultipartForm, StreamingResponse};
use crate::test_utils::MockStreamingClient;
use crate::wasm_compat::WasmCompatSend;
use bytes::Bytes;
use futures::StreamExt;
use http::{Request, Response};

#[derive(Clone)]
struct HeldOpenClient(Bytes);

impl HttpClientExt for HeldOpenClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let client = MockStreamingClient {
            sse_bytes: self.0.clone(),
        };
        let req = req.map(Into::into);
        async move { client.send(req).await }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let client = MockStreamingClient {
            sse_bytes: self.0.clone(),
        };
        async move { client.send_multipart(req).await }
    }

    async fn send_streaming<T>(&self, _req: Request<T>) -> http_client::Result<StreamingResponse>
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let stream = futures::stream::iter([Ok(self.0.clone())]).chain(futures::stream::pending());
        let body: http_client::BoxedStream = Box::pin(stream);
        Response::builder()
            .status(http::StatusCode::OK)
            .header(http::header::CONTENT_TYPE, "text/event-stream")
            .body(body)
            .map_err(http_client::Error::Protocol)
    }
}

#[tokio::test]
async fn terminal_delta_finishes_even_when_the_sse_connection_stays_open() {
    for message_stop in [false, true] {
        let mut frames = reply(
            vec![block(0, json!({"type": "text", "text": "hello"}), &[])],
            "end_turn",
        );
        if message_stop {
            frames.push(json!({"type": "message_stop"}));
        }
        let body = frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>();
        let model = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            HeldOpenClient(Bytes::from(body)),
        );
        let stream = match model.stream(CompletionRequest::new("hello")) {
            Ok(stream) => stream,
            Err(error) => panic!("stream failed to open: {error}"),
        };
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), stream.finish()).await;
        let response = match result {
            Ok(Ok(response)) => response,
            other => panic!("terminal stream did not finish: {other:?}"),
        };
        assert_eq!(response.stop(), StopReason::Stop);
        assert_eq!(response.usage.output_tokens, Some(5));
        assert!(
            matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello")
        );
    }
}
