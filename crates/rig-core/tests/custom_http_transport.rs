use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result as AnyResult};
use futures::stream;
use rig_core::completion::{AssistantContent, CompletionError, CompletionRequest};
use rig_core::http_client::{
    BoxedStream, Bytes, Error, HttpTransport, MultipartForm, Request, Response, Result,
    StreamingResponse, WasmBoxedFuture,
};
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::openai;
use rig_core::streaming::StreamedAssistantContent;
use rig_core::transcription::{TranscriptionError, TranscriptionRequest};

#[derive(Clone)]
struct ResponseSpec {
    status: http::StatusCode,
    content_type: Option<&'static str>,
    body: Bytes,
}

impl ResponseSpec {
    fn buffered(status: http::StatusCode, body: impl Into<Bytes>) -> Self {
        Self {
            status,
            content_type: Some("application/json"),
            body: body.into(),
        }
    }

    fn streaming(status: http::StatusCode, body: impl Into<Bytes>) -> Self {
        Self {
            status,
            content_type: Some("text/event-stream"),
            body: body.into(),
        }
    }

    fn response(&self) -> Result<Response<Bytes>> {
        let mut builder = Response::builder().status(self.status);
        if let Some(content_type) = self.content_type {
            builder = builder.header(http::header::CONTENT_TYPE, content_type);
        }
        builder.body(self.body.clone()).map_err(Error::Protocol)
    }

    fn streaming_response(&self) -> Result<StreamingResponse> {
        let mut builder = Response::builder().status(self.status);
        if let Some(content_type) = self.content_type {
            builder = builder.header(http::header::CONTENT_TYPE, content_type);
        }
        let chunks: BoxedStream = Box::pin(stream::iter([Ok(self.body.clone())]));
        builder.body(chunks).map_err(Error::Protocol)
    }
}

#[derive(Debug)]
struct RequestSnapshot {
    method: http::Method,
    uri: String,
    body: Vec<u8>,
}

#[derive(Default)]
struct TransportState {
    requests: Mutex<Vec<RequestSnapshot>>,
    buffered_calls: AtomicUsize,
    streaming_calls: AtomicUsize,
    multipart_calls: AtomicUsize,
}

#[derive(Clone)]
struct CannedTransport {
    state: Arc<TransportState>,
    buffered: ResponseSpec,
    streaming: ResponseSpec,
    _secret: Arc<String>,
}

impl CannedTransport {
    fn new(buffered: ResponseSpec, streaming: ResponseSpec) -> Self {
        Self {
            state: Arc::new(TransportState::default()),
            buffered,
            streaming,
            _secret: Arc::new("authorization: super-secret-token".to_string()),
        }
    }

    fn state(&self) -> Arc<TransportState> {
        Arc::clone(&self.state)
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn record_request(state: &TransportState, request: Request<Vec<u8>>) {
    let (parts, body) = request.into_parts();
    lock_unpoisoned(&state.requests).push(RequestSnapshot {
        method: parts.method,
        uri: parts.uri.to_string(),
        body,
    });
}

impl HttpTransport for CannedTransport {
    fn send(&self, request: Request<Vec<u8>>) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
        let state = Arc::clone(&self.state);
        let response = self.buffered.clone();
        Box::pin(async move {
            state.buffered_calls.fetch_add(1, Ordering::SeqCst);
            record_request(&state, request);
            response.response()
        })
    }

    fn send_streaming(
        &self,
        request: Request<Vec<u8>>,
    ) -> WasmBoxedFuture<'static, Result<StreamingResponse>> {
        let state = Arc::clone(&self.state);
        let response = self.streaming.clone();
        Box::pin(async move {
            state.streaming_calls.fetch_add(1, Ordering::SeqCst);
            record_request(&state, request);
            response.streaming_response()
        })
    }

    fn send_multipart(
        &self,
        _request: Request<MultipartForm>,
    ) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
        let state = Arc::clone(&self.state);
        let response = self.buffered.clone();
        Box::pin(async move {
            state.multipart_calls.fetch_add(1, Ordering::SeqCst);
            response.response()
        })
    }
}

#[derive(Clone)]
struct NoMultipartTransport;

impl HttpTransport for NoMultipartTransport {
    fn send(
        &self,
        _request: Request<Vec<u8>>,
    ) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
        Box::pin(async {
            Response::builder()
                .status(http::StatusCode::OK)
                .body(Bytes::new())
                .map_err(Error::Protocol)
        })
    }

    fn send_streaming(
        &self,
        _request: Request<Vec<u8>>,
    ) -> WasmBoxedFuture<'static, Result<StreamingResponse>> {
        Box::pin(async {
            let body: BoxedStream = Box::pin(stream::empty());
            Response::builder()
                .status(http::StatusCode::OK)
                .body(body)
                .map_err(Error::Protocol)
        })
    }
}

fn success_body() -> &'static str {
    r#"{
        "id":"chatcmpl-custom",
        "object":"chat.completion",
        "created":1,
        "model":"test-model",
        "system_fingerprint":null,
        "choices":[{
            "index":0,
            "message":{"role":"assistant","content":"transport works"},
            "logprobs":null,
            "finish_reason":"stop"
        }],
        "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
    }"#
}

fn sse_body() -> &'static str {
    concat!(
        "data: {\"id\":\"chatcmpl-stream\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"stream works\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
        "data: [DONE]\n\n",
    )
}

fn openai_config() -> openai::functions::Config {
    openai::functions::Config::new("test-model")
        .with_api_key("test-key")
        .with_base_url("https://custom.invalid/v1")
}

#[tokio::test]
async fn external_transport_drives_buffered_provider_call() -> AnyResult<()> {
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::OK, success_body()),
        ResponseSpec::streaming(http::StatusCode::OK, sse_body()),
    );
    let state = transport.state();
    let runtime = HttpRuntime::from_transport(transport);

    let response = openai::functions::complete(
        &openai_config(),
        &runtime,
        CompletionRequest::from_prompt("hello"),
    )
    .await?;

    let Some(AssistantContent::Text(text)) = response.choice.iter().next() else {
        anyhow::bail!("expected a normalized assistant text response");
    };
    anyhow::ensure!(text.text == "transport works");
    anyhow::ensure!(response.provider == "openai");
    anyhow::ensure!(state.buffered_calls.load(Ordering::SeqCst) == 1);

    let requests = lock_unpoisoned(&state.requests);
    let request = requests.first().context("request should be recorded")?;
    anyhow::ensure!(request.method == http::Method::POST);
    anyhow::ensure!(request.uri == "https://custom.invalid/v1/chat/completions");
    let body: serde_json::Value = serde_json::from_slice(&request.body)?;
    anyhow::ensure!(body.get("model").and_then(serde_json::Value::as_str) == Some("test-model"));
    Ok(())
}

#[tokio::test]
async fn buffered_non_success_is_normalized_before_provider_parsing() -> AnyResult<()> {
    let body = r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#;
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::TOO_MANY_REQUESTS, body),
        ResponseSpec::streaming(http::StatusCode::OK, sse_body()),
    );
    let runtime = HttpRuntime::from_transport(transport);

    let Err(error) = openai::functions::complete(
        &openai_config(),
        &runtime,
        CompletionRequest::from_prompt("hello"),
    )
    .await
    else {
        anyhow::bail!("a 429 response cannot be a successful completion");
    };

    anyhow::ensure!(matches!(error, CompletionError::HttpError(_)));
    anyhow::ensure!(error.provider_response_status() == Some(http::StatusCode::TOO_MANY_REQUESTS));
    anyhow::ensure!(error.provider_response_body() == Some(body));
    Ok(())
}

#[tokio::test]
async fn custom_stream_uses_the_existing_provider_sse_pipeline() -> AnyResult<()> {
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::OK, success_body()),
        ResponseSpec::streaming(http::StatusCode::OK, sse_body()),
    );
    let state = transport.state();
    let runtime = HttpRuntime::from_transport(transport);

    let mut completion = openai::functions::open_stream(
        &openai_config(),
        &runtime,
        CompletionRequest::from_prompt("hello"),
    )
    .await?;
    let mut text = String::new();
    while let Some(item) = completion.next().await {
        if let StreamedAssistantContent::Text(delta) = item? {
            text.push_str(&delta.text);
        }
    }

    anyhow::ensure!(text == "stream works");
    anyhow::ensure!(state.streaming_calls.load(Ordering::SeqCst) == 1);
    let requests = lock_unpoisoned(&state.requests);
    let request = requests
        .first()
        .context("stream request should be recorded")?;
    let body: serde_json::Value = serde_json::from_slice(&request.body)?;
    anyhow::ensure!(body.get("stream").and_then(serde_json::Value::as_bool) == Some(true));
    Ok(())
}

#[tokio::test]
async fn streaming_non_success_preserves_status_and_collected_body() -> AnyResult<()> {
    let body = r#"{"error":{"message":"stream rate limited"}}"#;
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::OK, success_body()),
        ResponseSpec::streaming(http::StatusCode::TOO_MANY_REQUESTS, body),
    );
    let runtime = HttpRuntime::from_transport(transport);
    let mut completion = openai::functions::open_stream(
        &openai_config(),
        &runtime,
        CompletionRequest::from_prompt("hello"),
    )
    .await?;

    let Some(item) = completion.next().await else {
        anyhow::bail!("stream should yield its transport error");
    };
    let Err(error) = item else {
        anyhow::bail!("429 cannot be a successful stream item");
    };

    anyhow::ensure!(matches!(error, CompletionError::HttpError(_)));
    anyhow::ensure!(error.provider_response_status() == Some(http::StatusCode::TOO_MANY_REQUESTS));
    anyhow::ensure!(error.provider_response_body() == Some(body));
    Ok(())
}

#[tokio::test]
async fn multipart_non_success_is_normalized_with_its_body() -> AnyResult<()> {
    let body = r#"{"error":{"message":"bad audio"}}"#;
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::TOO_MANY_REQUESTS, body),
        ResponseSpec::streaming(http::StatusCode::OK, sse_body()),
    );
    let state = transport.state();
    let runtime = HttpRuntime::from_transport(transport);

    let Err(error) = openai::functions::transcribe(
        &openai_config(),
        &runtime,
        TranscriptionRequest::new(vec![0, 1, 2]),
    )
    .await
    else {
        anyhow::bail!("a 429 response cannot be a successful transcription");
    };

    anyhow::ensure!(matches!(error, TranscriptionError::HttpError(_)));
    anyhow::ensure!(error.provider_response_status() == Some(http::StatusCode::TOO_MANY_REQUESTS));
    anyhow::ensure!(error.provider_response_body() == Some(body));
    anyhow::ensure!(state.multipart_calls.load(Ordering::SeqCst) == 1);
    Ok(())
}

#[tokio::test]
async fn omitted_multipart_method_returns_typed_capability_error() -> AnyResult<()> {
    let runtime = HttpRuntime::from_transport(NoMultipartTransport);
    let Err(error) = openai::functions::transcribe(
        &openai_config(),
        &runtime,
        TranscriptionRequest::new(vec![0, 1, 2]),
    )
    .await
    else {
        anyhow::bail!("default multipart implementation must reject the request");
    };

    anyhow::ensure!(matches!(
        error,
        TranscriptionError::HttpError(Error::UnsupportedMultipart)
    ));
    Ok(())
}

#[tokio::test]
async fn cloned_runtime_shares_transport_and_debug_is_redacted() -> AnyResult<()> {
    let transport = CannedTransport::new(
        ResponseSpec::buffered(http::StatusCode::OK, success_body()),
        ResponseSpec::streaming(http::StatusCode::OK, sse_body()),
    );
    let state = transport.state();
    let runtime = HttpRuntime::from_transport(transport);
    let cloned = runtime.clone();

    for active in [&runtime, &cloned] {
        let request = Request::get("https://custom.invalid/health").body(Vec::new())?;
        let (status, _) = active.send(request).await?;
        anyhow::ensure!(status == http::StatusCode::OK);
    }

    anyhow::ensure!(state.buffered_calls.load(Ordering::SeqCst) == 2);
    let debug = format!("{runtime:?}");
    anyhow::ensure!(debug.contains("custom"));
    anyhow::ensure!(!debug.contains("super-secret-token"));
    Ok(())
}

#[cfg(feature = "reqwest-middleware")]
struct ObservingMiddleware {
    observed: Arc<AtomicUsize>,
}

#[cfg(feature = "reqwest-middleware")]
#[async_trait::async_trait]
impl reqwest_middleware::Middleware for ObservingMiddleware {
    async fn handle(
        &self,
        request: reqwest::Request,
        extensions: &mut http::Extensions,
        next: reqwest_middleware::Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        self.observed.fetch_add(1, Ordering::SeqCst);
        next.run(request, extensions).await
    }
}

#[cfg(feature = "reqwest-middleware")]
#[tokio::test]
async fn reqwest_middleware_transport_runs_attached_middleware() -> AnyResult<()> {
    use std::io;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await?;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
            .await?;
        Ok::<(), io::Error>(())
    });

    let observed = Arc::new(AtomicUsize::new(0));
    let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
        .with(ObservingMiddleware {
            observed: Arc::clone(&observed),
        })
        .build();
    let runtime = HttpRuntime::from_transport(client);
    let request = Request::get(format!("http://{address}/observed")).body(Vec::new())?;

    let (status, body) = runtime.send(request).await?;
    anyhow::ensure!(status == http::StatusCode::OK);
    anyhow::ensure!(body == "ok");
    anyhow::ensure!(observed.load(Ordering::SeqCst) == 1);
    server.await.context("loopback server task should join")??;
    Ok(())
}
