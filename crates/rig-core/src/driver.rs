//! Unified driver for provider wires.
//!
//! There is exactly one way to call a provider in rig-core:
//! - [`call`] for unary and paged operations.
//! - [`stream`] for streaming operations.
//! - [`Bound`] pairs a [`Wire`] with an HTTP transport.

use bytes::Bytes;
use futures::{Stream, StreamExt};

use crate::http_client::framing::{Framing, NdjsonFramer, SseFramer};
use crate::http_client::{BoxedHttpClient, HttpClientExt};
use crate::observe::{AdapterAttempt, AdapterContext, AdapterEnding};
use crate::operation::{Completion, Fold, Operation};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Body, Decoder, Encoded, Output, Wire, WireEvent, WireFrame};

/// Wire-specific configuration for embedding models.
pub trait EmbeddingWire: Wire<Op = crate::operation::Embedding> {
    fn max_documents(&self) -> usize;
    fn ndims(&self) -> usize;
}

/// Wire-specific configuration for image embedding models.
pub trait ImageEmbeddingWire: Wire<Op = crate::operation::ImageEmbedding> {
    fn max_documents(&self) -> usize;
    fn ndims(&self) -> usize;
}

/// Wire-specific configuration for reranking models.
pub trait RerankWire: Wire<Op = crate::operation::Rerank> {
    fn max_documents(&self) -> usize;
}

/// Provider configurations that can construct a completion wire.
pub trait HasCompletion {
    type Wire: Wire<Op = Completion>;
    fn completion(&self, model: &str) -> Self::Wire;
}

/// A pure fold state machine over wire frames.
pub struct WireDriver<Op: Operation, D: Decoder<Op>> {
    decoder: D,
    out: Output<Op>,
    frames: usize,
    finished: bool,
    request_id: Option<String>,
}

impl<Op: Operation, D: Decoder<Op>> WireDriver<Op, D>
where
    Op::Error: From<serde_json::Error>,
{
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            out: Output::new(),
            frames: 0,
            finished: false,
            request_id: None,
        }
    }

    pub fn set_request_id(&mut self, request_id: Option<String>) {
        self.request_id = request_id;
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn continuation(&self) -> Option<http::Request<Body>> {
        self.decoder.continuation()
    }

    pub fn project(&self, payload: &[u8], sink: &mut dyn crate::wire::ObservationSink) {
        self.decoder.project(payload, sink);
    }

    pub fn push(
        &mut self,
        frame: Result<WireFrame, Op::Error>,
    ) -> Vec<Result<Op::Event, Op::Error>> {
        if self.finished {
            return Vec::new();
        }
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                self.decoder.flush_before_terminal_error(&mut self.out);
                let mut drained: Vec<_> = self.out.drain().collect();
                drained.push(Err(error));
                self.finished = true;
                return drained;
            }
        };

        self.frames += 1;
        match self.decoder.classify(frame) {
            WireEvent::Known(event) => {
                self.decoder.interpret(event, &mut self.out);
            }
            WireEvent::Unknown { event_type, value } => {
                crate::providers::internal::adapter::warn_unmodeled(&event_type, &value);
            }
            WireEvent::Corrupt(error) => {
                let mut drained: Vec<_> = self.out.drain().collect();
                drained.push(Err(<Op::Error>::from(error)));
                return drained;
            }
        }
        self.out.drain().collect()
    }

    pub fn finish(&mut self) -> Vec<Result<Op::Event, Op::Error>> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        self.decoder.finish(&mut self.out);
        self.out.drain().collect()
    }
}

/// Execute a unary or paged operation.
pub async fn call<W: Wire, H: HttpClientExt>(
    wire: &W,
    http: &H,
    request: <W::Op as Operation>::Request,
    observe: Option<AdapterContext>,
) -> Result<<W::Op as Operation>::Response, <W::Op as Operation>::Error>
where
    <W::Op as Operation>::Error: From<crate::http_client::Error> + From<serde_json::Error>,
    W::Decoder: WasmCompatSend,
{
    let mut fold = <W::Op as Operation>::Fold::default();
    let mut decoder = wire.decoder();
    let Encoded {
        request,
        framing: _,
        request_id_header,
        route,
    } = wire.encode(request)?;

    let mut next_request = Some(request);
    let mut last_body_bytes: Option<Bytes> = None;
    let mut last_request_id: Option<String> = None;

    while let Some(current_req) = next_request.take() {
        let mut attempt: Option<AdapterAttempt> = observe
            .as_ref()
            .and_then(|ctx| ctx.begin(current_req.method(), route));

        let (parts, body_bytes) = match current_req.into_parts() {
            (parts, Body::Bytes(bytes)) => {
                let req = http::Request::from_parts(parts, bytes);
                let resp = http.send::<_, Bytes>(req).await.inspect_err(|err| {
                    if let Some(attempt) = &mut attempt {
                        if let Some(status) = err.non_success_status() {
                            attempt.response_with_headers(status, err.non_success_headers());
                        }
                        if let Some(body) = err.non_success_body() {
                            attempt.payload(body.as_bytes());
                        }
                    }
                });
                let resp = match resp {
                    Ok(r) => r,
                    Err(e) => {
                        if let Some(attempt) = &mut attempt {
                            if let Some(status) = e.non_success_status() {
                                attempt.response_with_headers(status, e.non_success_headers());
                                if let Some(body) = e.non_success_body() {
                                    decoder.project(body.as_bytes(), attempt);
                                }
                                attempt.finish(AdapterEnding::Error {
                                    boundary:
                                        crate::observe::AdapterErrorBoundary::ProviderResponse,
                                    kind: "provider_response".to_string(),
                                    status: Some(status.as_u16()),
                                    retryable: status == http::StatusCode::TOO_MANY_REQUESTS
                                        || status == http::StatusCode::SERVICE_UNAVAILABLE,
                                });
                            } else {
                                attempt.finish(AdapterEnding::Error {
                                    boundary: crate::observe::AdapterErrorBoundary::Request,
                                    kind: "transport_error".to_string(),
                                    status: None,
                                    retryable: false,
                                });
                            }
                        }
                        let req_id = request_id_header.and_then(|h| {
                            e.non_success_headers().and_then(|headers| {
                                headers
                                    .get(h)
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_string)
                            })
                        });
                        let mut err = <W::Op as Operation>::Error::from(e);
                        if let Some(comp_err) = (&mut err as &mut dyn std::any::Any)
                            .downcast_mut::<crate::completion::CompletionError>(
                        ) {
                            let old = std::mem::replace(
                                comp_err,
                                crate::completion::CompletionError::RequestError(
                                    String::new().into(),
                                ),
                            );
                            *comp_err = old.with_provider_request_id(req_id);
                        }
                        return Err(err);
                    }
                };
                let (parts, body) = resp.into_parts();
                let body_bytes = body.await.map_err(<W::Op as Operation>::Error::from)?;
                (parts, body_bytes)
            }
            (parts, Body::Multipart(form)) => {
                let req = http::Request::from_parts(parts, form);
                let resp = http.send_multipart::<Bytes>(req).await.inspect_err(|err| {
                    if let Some(attempt) = &mut attempt {
                        if let Some(status) = err.non_success_status() {
                            attempt.response_with_headers(status, err.non_success_headers());
                        }
                        if let Some(body) = err.non_success_body() {
                            attempt.payload(body.as_bytes());
                        }
                    }
                });
                let resp = match resp {
                    Ok(r) => r,
                    Err(e) => {
                        if let Some(attempt) = &mut attempt {
                            if let Some(status) = e.non_success_status() {
                                attempt.response_with_headers(status, e.non_success_headers());
                                if let Some(body) = e.non_success_body() {
                                    decoder.project(body.as_bytes(), attempt);
                                }
                                attempt.finish(AdapterEnding::Error {
                                    boundary:
                                        crate::observe::AdapterErrorBoundary::ProviderResponse,
                                    kind: "provider_response".to_string(),
                                    status: Some(status.as_u16()),
                                    retryable: status == http::StatusCode::TOO_MANY_REQUESTS
                                        || status == http::StatusCode::SERVICE_UNAVAILABLE,
                                });
                            } else {
                                attempt.finish(AdapterEnding::Error {
                                    boundary: crate::observe::AdapterErrorBoundary::Request,
                                    kind: "transport_error".to_string(),
                                    status: None,
                                    retryable: false,
                                });
                            }
                        }
                        let req_id = request_id_header.and_then(|h| {
                            e.non_success_headers().and_then(|headers| {
                                headers
                                    .get(h)
                                    .and_then(|v| v.to_str().ok())
                                    .map(str::to_string)
                            })
                        });
                        let mut err = <W::Op as Operation>::Error::from(e);
                        if let Some(comp_err) = (&mut err as &mut dyn std::any::Any)
                            .downcast_mut::<crate::completion::CompletionError>(
                        ) {
                            let old = std::mem::replace(
                                comp_err,
                                crate::completion::CompletionError::RequestError(
                                    String::new().into(),
                                ),
                            );
                            *comp_err = old.with_provider_request_id(req_id);
                        }
                        return Err(err);
                    }
                };
                let (parts, body) = resp.into_parts();
                let body_bytes = body.await.map_err(<W::Op as Operation>::Error::from)?;
                (parts, body_bytes)
            }
        };

        if let Some(attempt) = &mut attempt {
            attempt.response_with_headers(parts.status, Some(&parts.headers));
        }

        let provider_request_id = request_id_header.and_then(|header| {
            parts
                .headers
                .get(header)
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        });
        last_request_id = provider_request_id.clone();

        last_body_bytes = Some(body_bytes.clone());
        if let Some(attempt) = &mut attempt {
            decoder.project(&body_bytes, attempt);
        }

        if !parts.status.is_success() {
            let err = crate::http_client::Error::InvalidStatusCodeWithDetails {
                status: parts.status,
                body: String::from_utf8_lossy(&body_bytes).into_owned(),
                headers: parts.headers,
            };
            if let Some(attempt) = &mut attempt {
                attempt.finish(AdapterEnding::Error {
                    boundary: crate::observe::AdapterErrorBoundary::ProviderResponse,
                    kind: "provider_response".to_string(),
                    status: Some(parts.status.as_u16()),
                    retryable: parts.status == http::StatusCode::TOO_MANY_REQUESTS
                        || parts.status == http::StatusCode::SERVICE_UNAVAILABLE,
                });
            }
            return Err(<W::Op as Operation>::Error::from(err));
        }

        let frame = WireFrame::Bytes(body_bytes.to_vec());
        let mut out = Output::new();
        match decoder.classify(frame) {
            WireEvent::Known(event) => {
                decoder.interpret(event, &mut out);
            }
            WireEvent::Unknown { event_type, value } => {
                crate::providers::internal::adapter::warn_unmodeled(&event_type, &value);
            }
            WireEvent::Corrupt(error) => {
                if let Some(attempt) = &mut attempt {
                    attempt.finish(AdapterEnding::Error {
                        boundary: crate::observe::AdapterErrorBoundary::Decode,
                        kind: "corrupt".to_string(),
                        status: None,
                        retryable: false,
                    });
                }
                return Err(<W::Op as Operation>::Error::from(error));
            }
        }
        decoder.finish(&mut out);

        for item in out.drain() {
            match item {
                Ok(event) => fold.fold(event),
                Err(mut error) => {
                    if let Some(crate::completion::CompletionError::ProviderResponse(stored)) =
                        (&mut error as &mut dyn std::any::Any)
                            .downcast_mut::<crate::completion::CompletionError>()
                    {
                        if stored.status.is_none() {
                            stored.status = Some(parts.status);
                        }
                        if stored.headers.is_none() {
                            stored.headers = Some(parts.headers.clone());
                        }
                    }
                    if let Some(attempt) = &mut attempt {
                        attempt.finish(AdapterEnding::Error {
                            boundary: crate::observe::AdapterErrorBoundary::ProviderResponse,
                            kind: "provider_response".to_string(),
                            status: Some(parts.status.as_u16()),
                            retryable: parts.status == http::StatusCode::TOO_MANY_REQUESTS
                                || parts.status == http::StatusCode::SERVICE_UNAVAILABLE,
                        });
                    }
                    return Err(error);
                }
            }
        }

        if let Some(attempt) = &mut attempt {
            attempt.finish(AdapterEnding::Decoded);
        }

        next_request = decoder.continuation();
    }

    let mut response = fold.finish()?;
    let raw_value: Option<serde_json::Value> = last_body_bytes
        .as_ref()
        .and_then(|b| serde_json::from_slice(b).ok());
    apply_response_metadata(&mut response, wire.name(), raw_value, last_request_id);
    Ok(response)
}

fn apply_response_metadata<R: 'static>(
    response: &mut R,
    provider_name: &str,
    raw_value: Option<serde_json::Value>,
    provider_request_id: Option<String>,
) {
    if let Some(resp) =
        (response as &mut dyn std::any::Any).downcast_mut::<crate::completion::CompletionResponse>()
    {
        resp.provider = provider_name.to_owned();
        if resp.raw.is_null() {
            if let Some(mut raw) = raw_value {
                if let Some(id) = &provider_request_id
                    && let Some(obj) = raw.as_object_mut()
                {
                    obj.insert(
                        "provider_request_id".to_string(),
                        serde_json::Value::String(id.clone()),
                    );
                }
                resp.raw = raw;
            }
        } else if let Some(id) = &provider_request_id
            && let Some(obj) = resp.raw.as_object_mut()
        {
            obj.insert(
                "provider_request_id".to_string(),
                serde_json::Value::String(id.clone()),
            );
        }
        if resp.provider_request_id.is_none() {
            resp.provider_request_id = provider_request_id;
        }
    }
}

/// Execute a streaming operation.
pub fn stream<W: Wire, H: HttpClientExt + Clone + 'static>(
    wire: &W,
    http: &H,
    request: <W::Op as Operation>::Request,
    observe: Option<AdapterContext>,
) -> Result<
    impl Stream<Item = Result<<W::Op as Operation>::Event, <W::Op as Operation>::Error>>
    + WasmCompatSend
    + 'static,
    <W::Op as Operation>::Error,
>
where
    <W::Op as Operation>::Error:
        From<crate::http_client::Error> + From<serde_json::Error> + WasmCompatSend,
    <W::Op as Operation>::Event: WasmCompatSend,
    W::Decoder: WasmCompatSend,
{
    let Encoded {
        mut request,
        framing,
        request_id_header,
        route,
    } = wire.encode(request)?;

    if framing == Framing::Sse {
        request.headers_mut().insert(
            http::header::ACCEPT,
            http::HeaderValue::from_static("text/event-stream"),
        );
        if let Body::Bytes(bytes) = request.body_mut()
            && let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(bytes)
            && let Some(map) = val.as_object_mut()
        {
            map.insert("stream".to_string(), serde_json::Value::Bool(true));
            if wire.name() == "anthropic" {
                if map.contains_key("tools") {
                    map.entry("tool_choice")
                        .or_insert_with(|| serde_json::json!({ "type": "auto" }));
                } else {
                    map.remove("tool_choice");
                }
            }
            if let Ok(new_bytes) = serde_json::to_vec(&val) {
                *bytes = new_bytes;
            }
        }
    }

    let req_bytes = match request.into_parts() {
        (parts, Body::Bytes(bytes)) => http::Request::from_parts(parts, bytes),
        _ => {
            return Err(<W::Op as Operation>::Error::from(
                crate::http_client::Error::NoHeaders,
            ));
        }
    };

    let decoder = wire.decoder();
    let http = http.clone();

    let mut attempt: Option<AdapterAttempt> = observe
        .as_ref()
        .and_then(|ctx| ctx.begin(&http::Method::POST, route));

    let stream = async_stream::stream! {
        let response: crate::http_client::StreamingResponse = match http.send_streaming(req_bytes).await {
            Ok(r) => {
                if let Some(attempt) = &mut attempt {
                    attempt.response_with_headers(r.status(), Some(r.headers()));
                }
                r
            }
            Err(e) => {
                if let Some(attempt) = &mut attempt {
                    if let Some(status) = e.non_success_status() {
                        attempt.response_with_headers(status, e.non_success_headers());
                        if let Some(body) = e.non_success_body() {
                            decoder.project(body.as_bytes(), attempt);
                        }
                        attempt.finish(AdapterEnding::Error {
                            boundary: crate::observe::AdapterErrorBoundary::ProviderResponse,
                            kind: "provider_response".to_string(),
                            status: Some(status.as_u16()),
                            retryable: status == http::StatusCode::TOO_MANY_REQUESTS
                                || status == http::StatusCode::SERVICE_UNAVAILABLE,
                        });
                    } else {
                        attempt.finish(AdapterEnding::Error {
                            boundary: crate::observe::AdapterErrorBoundary::Transport,
                            kind: "transport".to_string(),
                            status: None,
                            retryable: false,
                        });
                    }
                }
                    let req_id = request_id_header.and_then(|h| {
                        e.non_success_headers().and_then(|headers| {
                            headers.get(h).and_then(|v| v.to_str().ok()).map(str::to_string)
                        })
                    });
                    let mut err = <W::Op as Operation>::Error::from(e);
                    if let Some(comp_err) = (&mut err as &mut dyn std::any::Any).downcast_mut::<crate::completion::CompletionError>() {
                        let old = std::mem::replace(comp_err, crate::completion::CompletionError::RequestError(String::new().into()));
                        *comp_err = old.with_provider_request_id(req_id);
                    }
                    yield Err(err);
                return;
            }
        };
        let (parts, byte_stream) = response.into_parts();
        let stream_request_id = request_id_header.and_then(|h| {
            parts.headers.get(h).and_then(|v| v.to_str().ok()).map(str::to_string)
        });
        let mut driver = WireDriver::new(decoder);
        let mut byte_stream = Box::pin(byte_stream);

        let stamp_item = |mut item: Result<<W::Op as Operation>::Event, <W::Op as Operation>::Error>| {
            if let Ok(event) = &mut item
                && let Some(crate::streaming::StreamEvent::Final(terminal)) =
                    (event as &mut dyn std::any::Any).downcast_mut::<crate::streaming::StreamEvent>()
                && terminal.provider_request_id.is_none()
            {
                terminal.provider_request_id = stream_request_id.clone();
            }
            item
        };
        match framing {
            Framing::Sse => {
                let mut framer = SseFramer::new();
                while let Some(chunk_res) = byte_stream.next().await {
                    let chunk: Bytes = match chunk_res {
                        Ok(c) => c,
                        Err(e) => {
                            yield Err(<W::Op as Operation>::Error::from(e));
                            return;
                        }
                    };
                    for sse in framer.push(&chunk) {
                        if let Some(attempt) = &mut attempt {
                            driver.decoder.project(sse.data.as_bytes(), attempt);
                        }
                        let frame = WireFrame::Text(sse.data);
                        for item in driver.push(Ok(frame)) {
                            yield stamp_item(item);
                        }
                        if driver.is_finished() {
                            return;
                        }
                    }
                }
            }
            Framing::Ndjson => {
                let mut framer = NdjsonFramer::new();
                while let Some(chunk_res) = byte_stream.next().await {
                    let chunk: Bytes = match chunk_res {
                        Ok(c) => c,
                        Err(e) => {
                            yield Err(<W::Op as Operation>::Error::from(e));
                            return;
                        }
                    };
                    for line in framer.push(&chunk) {
                        let frame = WireFrame::Bytes(line.to_vec());
                        for item in driver.push(Ok(frame)) {
                            yield stamp_item(item);
                        }
                        if driver.is_finished() {
                            return;
                        }
                    }
                }
                if let Some(line) = framer.finish() {
                    let frame = WireFrame::Bytes(line.to_vec());
                    for item in driver.push(Ok(frame)) {
                        yield stamp_item(item);
                    }
                }
            }
            Framing::Whole => {
                let mut body = Vec::new();
                while let Some(chunk_res) = byte_stream.next().await {
                    let chunk: Bytes = match chunk_res {
                        Ok(c) => c,
                        Err(e) => {
                            yield Err(<W::Op as Operation>::Error::from(e));
                            return;
                        }
                    };
                    body.extend_from_slice(&chunk);
                }
                let frame = WireFrame::Bytes(body);
                for item in driver.push(Ok(frame)) {
                    yield stamp_item(item);
                }
            }
        }

        for item in driver.finish() {
            yield stamp_item(item);
        }
        if let Some(attempt) = &mut attempt {
            attempt.finish(AdapterEnding::Terminal);
        }
    };

    Ok(stream)
}

/// A wire bound to an HTTP transport.
#[derive(Debug, Clone)]
pub struct Bound<W, H = BoxedHttpClient> {
    pub wire: W,
    pub http: H,
}

impl<W, H> Bound<W, H> {
    pub fn new(wire: W, http: H) -> Self {
        Self { wire, http }
    }

    pub fn map_wire<U>(self, f: impl FnOnce(W) -> U) -> Bound<U, H> {
        Bound {
            wire: f(self.wire),
            http: self.http,
        }
    }
}
impl<W, H> std::ops::Deref for Bound<W, H> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.wire
    }
}

impl<W, H> std::ops::DerefMut for Bound<W, H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wire
    }
}

impl<W: HasCompletion, H: HttpClientExt + Clone + 'static> crate::client::CompletionClient
    for Bound<W, H>
where
    W::Wire: 'static,
    <W::Wire as Wire>::Decoder: WasmCompatSend,
{
    type CompletionModel = Bound<W::Wire, H>;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        Bound::new(self.wire.completion(&model.into()), self.http.clone())
    }
}

// Consumer trait implementations for Bound

impl<W, H> crate::completion::CompletionModel for Bound<W, H>
where
    W: Wire<Op = Completion>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone + 'static,
{
    async fn completion(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<crate::completion::CompletionResponse, crate::completion::CompletionError> {
        call(&self.wire, &self.http, request, None).await
    }

    async fn stream(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, crate::completion::CompletionError>
    {
        let events = stream(&self.wire, &self.http, request, None)?;
        Ok(crate::streaming::StreamingCompletionResponse::stream(
            self.wire.name().to_string(),
            Box::pin(events),
        ))
    }
    async fn completion_with_context(
        &self,
        request: crate::completion::CompletionRequest,
        context: Option<crate::observe::AdapterContext>,
    ) -> Result<crate::completion::CompletionResponse, crate::completion::CompletionError> {
        call(&self.wire, &self.http, request, context).await
    }

    async fn stream_with_context(
        &self,
        request: crate::completion::CompletionRequest,
        context: Option<crate::observe::AdapterContext>,
    ) -> Result<crate::streaming::StreamingCompletionResponse, crate::completion::CompletionError>
    {
        let events = stream(&self.wire, &self.http, request, context)?;
        Ok(crate::streaming::StreamingCompletionResponse::stream(
            self.wire.name().to_string(),
            Box::pin(events),
        ))
    }

    fn capabilities(&self) -> crate::completion::ProviderCapabilities {
        self.wire.capabilities()
    }
}

impl<W, H> crate::embeddings::EmbeddingModel for Bound<W, H>
where
    W: EmbeddingWire,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    fn max_documents(&self) -> usize {
        self.wire.max_documents()
    }

    fn ndims(&self) -> usize {
        self.wire.ndims()
    }

    async fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        let texts_vec: Vec<String> = texts.into_iter().collect();
        call(&self.wire, &self.http, texts_vec, None).await
    }
}

impl<W, H> crate::embeddings::ImageEmbeddingModel for Bound<W, H>
where
    W: ImageEmbeddingWire,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    fn max_documents(&self) -> usize {
        self.wire.max_documents()
    }

    fn ndims(&self) -> usize {
        self.wire.ndims()
    }

    async fn embed_images_response(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> Result<crate::embeddings::ImageEmbeddingResponse, crate::embeddings::EmbeddingError> {
        let images_vec: Vec<Vec<u8>> = images.into_iter().collect();
        call(&self.wire, &self.http, images_vec, None).await
    }
}

impl<W, H> crate::transcription::TranscriptionModel for Bound<W, H>
where
    W: Wire<Op = crate::operation::Transcription>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    async fn transcription(
        &self,
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<crate::transcription::TranscriptionResponse, crate::transcription::TranscriptionError>
    {
        call(&self.wire, &self.http, request, None).await
    }
}

#[cfg(feature = "image")]
impl<W, H> crate::image_generation::ImageGenerationModel for Bound<W, H>
where
    W: Wire<Op = crate::operation::ImageGeneration>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    async fn image_generation(
        &self,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<
        crate::image_generation::ImageGenerationResponse,
        crate::image_generation::ImageGenerationError,
    > {
        call(&self.wire, &self.http, request, None).await
    }
}

#[cfg(feature = "audio")]
impl<W, H> crate::audio_generation::AudioGenerationModel for Bound<W, H>
where
    W: Wire<Op = crate::operation::AudioGeneration>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    async fn audio_generation(
        &self,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse,
        crate::audio_generation::AudioGenerationError,
    > {
        call(&self.wire, &self.http, request, None).await
    }
}

impl<W, H> crate::rerank::RerankModel for Bound<W, H>
where
    W: RerankWire,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    fn max_documents(&self) -> usize {
        self.wire.max_documents()
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<crate::rerank::RerankResponse, crate::rerank::RerankError> {
        let req = crate::operation::rerank::RerankRequest {
            query: query.to_owned(),
            documents,
            top_n: None,
        };
        call(&self.wire, &self.http, req, None).await
    }
}

impl<W, H> crate::client::ModelLister<H> for Bound<W, H>
where
    W: Wire<Op = crate::operation::ModelListing>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    async fn list_all(
        &self,
    ) -> Result<crate::model::ModelList, crate::model::listing::ModelListingError> {
        let models = call(&self.wire, &self.http, (), None).await?;
        Ok(crate::model::ModelList { data: models })
    }
}

impl<W, H> crate::client::VerifyClient for Bound<W, H>
where
    W: Wire<Op = crate::operation::Verify>,
    W::Decoder: WasmCompatSend,
    H: HttpClientExt + Clone,
{
    async fn verify(&self) -> Result<(), crate::client::verify::VerifyError> {
        call(&self.wire, &self.http, (), None).await
    }
}

#[cfg(test)]
mod tests;
