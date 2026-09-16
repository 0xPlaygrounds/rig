use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, Request, Response, StatusCode};

use crate::http_client::framing::Framing;
use crate::http_client::{HttpClientExt, LazyBody, StreamingResponse};
use crate::model::Model;
use crate::model::listing::ModelListingError;
use crate::operation::{Completion, ModelListing};
use crate::streaming::{StreamEvent, StreamFinal};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::ObservationSink;
use crate::wire::{Body, Decoder, Encoded, Output, Wire, WireEvent, WireFrame};

use super::{call, stream};

// Mock HTTP client for driver tests
#[derive(Clone, Default)]
struct MockHttpClient {
    responses: Arc<Mutex<VecDeque<Result<Response<Vec<u8>>, crate::http_client::Error>>>>,
    streaming_responses: Arc<Mutex<VecDeque<Result<StreamingResponse, crate::http_client::Error>>>>,
    sent_requests: Arc<Mutex<Vec<Request<Vec<u8>>>>>,
}

impl MockHttpClient {
    fn push_response(&self, res: Result<Response<Vec<u8>>, crate::http_client::Error>) {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(res);
    }

    fn push_streaming(&self, res: Result<StreamingResponse, crate::http_client::Error>) {
        self.streaming_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(res);
    }
}

impl HttpClientExt for MockHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>, crate::http_client::Error>>
    + WasmCompatSend
    + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let (parts, body) = req.into_parts();
        let bytes: Bytes = body.into();
        self.sent_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Request::from_parts(parts, bytes.to_vec()));

        let res = self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| {
                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .body(Vec::new())
                    .unwrap_or_else(|_| Response::new(Vec::new())))
            });

        async move {
            let res = res?;
            let (parts, body) = res.into_parts();
            let lazy: LazyBody<U> = Box::pin(async move { Ok(U::from(Bytes::from(body))) });
            Ok(Response::from_parts(parts, lazy))
        }
    }

    fn send_multipart<U>(
        &self,
        _req: Request<crate::http_client::MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>, crate::http_client::Error>>
    + WasmCompatSend
    + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(crate::http_client::Error::NoHeaders))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse, crate::http_client::Error>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let (parts, body) = req.into_parts();
        let bytes: Bytes = body.into();
        self.sent_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Request::from_parts(parts, bytes.to_vec()));

        let res = self
            .streaming_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| Err(crate::http_client::Error::NoHeaders));

        async move { res }
    }
}

// A simple test completion decoder
#[derive(Default)]
struct TestCompletionDecoder;

impl Decoder<Completion> for TestCompletionDecoder {
    type Event = String;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match frame {
            WireFrame::Text(s) => WireEvent::Known(s),
            WireFrame::Bytes(b) => WireEvent::Known(String::from_utf8_lossy(&b).into_owned()),
        }
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Completion>) {
        if event == "[DONE]" {
            out.emit(StreamEvent::Final(StreamFinal::new(
                "test_provider",
                crate::completion::Usage::default(),
            )));
        } else {
            out.emit(StreamEvent::text(
                crate::streaming::BlockId::wire("0"),
                &event,
            ));
        }
    }

    fn finish(&mut self, _out: &mut Output<Completion>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut Output<Completion>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn ObservationSink) {}
}

// A test completion wire
struct TestCompletionWire;

impl Wire for TestCompletionWire {
    type Op = Completion;
    type Decoder = TestCompletionDecoder;

    fn name(&self) -> &str {
        "test_provider"
    }

    fn encode(
        &self,
        _request: crate::completion::CompletionRequest,
    ) -> Result<Encoded, crate::completion::CompletionError> {
        let req = Request::builder()
            .method(Method::POST)
            .uri("https://api.test.com/v1/chat")
            .body(Body::Bytes(b"{}".to_vec()))
            .unwrap_or_else(|_| Request::new(Body::Bytes(Vec::new())));

        Ok(Encoded::new(req, Framing::Sse, "/v1/chat"))
    }

    fn decoder(&self) -> Self::Decoder {
        TestCompletionDecoder
    }

    fn capabilities(&self) -> crate::completion::ProviderCapabilities {
        crate::completion::ProviderCapabilities::default()
    }
}

// A paged model listing decoder
struct PagedModelDecoder {
    current_page: Arc<Mutex<usize>>,
}

impl Decoder<ModelListing> for PagedModelDecoder {
    type Event = Vec<Model>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let text = frame.as_str();
        let models = vec![Model::new(text.trim(), text.trim())];
        WireEvent::Known(models)
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<ModelListing>) {
        let mut page = self
            .current_page
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *page += 1;
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut Output<ModelListing>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut Output<ModelListing>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn ObservationSink) {}

    fn continuation(&self) -> Option<Request<Body>> {
        let page = *self
            .current_page
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if page < 2 {
            let req = Request::builder()
                .method(Method::GET)
                .uri(format!("https://api.test.com/v1/models?page={}", page + 1))
                .body(Body::Bytes(Vec::new()))
                .unwrap_or_else(|_| Request::new(Body::Bytes(Vec::new())));
            Some(req)
        } else {
            None
        }
    }
}

struct PagedModelWire {
    page_counter: Arc<Mutex<usize>>,
}

impl Wire for PagedModelWire {
    type Op = ModelListing;
    type Decoder = PagedModelDecoder;

    fn name(&self) -> &str {
        "test_models"
    }

    fn encode(&self, _request: ()) -> Result<Encoded, ModelListingError> {
        let req = Request::builder()
            .method(Method::GET)
            .uri("https://api.test.com/v1/models")
            .body(Body::Bytes(Vec::new()))
            .unwrap_or_else(|_| Request::new(Body::Bytes(Vec::new())));
        Ok(Encoded::new(req, Framing::Whole, "/v1/models"))
    }

    fn decoder(&self) -> Self::Decoder {
        PagedModelDecoder {
            current_page: self.page_counter.clone(),
        }
    }

    fn capabilities(&self) {}
}

fn test_completion_request() -> crate::completion::CompletionRequest {
    crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[tokio::test]
async fn unary_call_folds_response() {
    let wire = TestCompletionWire;
    let client = MockHttpClient::default();
    client.push_response(Ok(Response::builder()
        .status(StatusCode::OK)
        .body(b"hello world".to_vec())
        .unwrap_or_else(|_| Response::new(Vec::new()))));

    let req = test_completion_request();
    let resp = call(&wire, &client, req, None)
        .await
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(resp.provider, "test_provider");
}

#[tokio::test]
async fn unary_call_preserves_non_success_headers() {
    let wire = TestCompletionWire;
    let client = MockHttpClient::default();

    let mut headers = HeaderMap::new();
    headers.insert("retry-after", HeaderValue::from_static("120"));

    let mut builder = Response::builder().status(StatusCode::TOO_MANY_REQUESTS);
    for (k, v) in headers.iter() {
        builder = builder.header(k, v);
    }
    client.push_response(Ok(builder
        .body(b"rate limit".to_vec())
        .unwrap_or_else(|_| Response::new(Vec::new()))));

    let req = test_completion_request();
    let err = call(&wire, &client, req, None).await.unwrap_err();
    assert!(err.to_string().contains("429"));
}

#[tokio::test]
async fn streaming_connect_error_is_only_item() {
    use futures::StreamExt;

    let wire = TestCompletionWire;
    let client = MockHttpClient::default();
    client.push_streaming(Err(
        crate::http_client::Error::InvalidStatusCodeWithDetails {
            status: StatusCode::UNAUTHORIZED,
            body: "bad key".to_string(),
            headers: HeaderMap::new(),
        },
    ));

    let req = test_completion_request();
    let stream = stream(&wire, &client, req, None).unwrap_or_else(|e| panic!("{e:?}"));
    let items: Vec<_> = stream.collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_err());
}

#[tokio::test]
async fn paged_operation_loops_until_continuation_none() {
    let counter = Arc::new(Mutex::new(0));
    let wire = PagedModelWire {
        page_counter: counter.clone(),
    };
    let client = MockHttpClient::default();
    client.push_response(Ok(Response::new(b"model-1".to_vec())));
    client.push_response(Ok(Response::new(b"model-2".to_vec())));

    let models = call(&wire, &client, (), None)
        .await
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "model-1");
    assert_eq!(models[1].id, "model-2");
}
