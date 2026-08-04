use futures::stream;
use rig::http_client::{
    BoxedStream, Bytes, CustomTransport, Error, HttpTransport, MultipartForm, Request, Response,
    Result, StreamingResponse, WasmBoxedFuture,
};
use rig::http_runtime::HttpRuntime;
use rig::providers::openai;

#[derive(Clone)]
struct FacadeTransport;

impl HttpTransport for FacadeTransport {
    fn send(
        &self,
        _request: Request<Vec<u8>>,
    ) -> WasmBoxedFuture<'static, Result<Response<Bytes>>> {
        Box::pin(async {
            Response::builder()
                .status(200)
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
                .status(200)
                .body(body)
                .map_err(Error::Protocol)
        })
    }
}

#[test]
fn root_facade_exposes_custom_transport_and_client_injection() -> anyhow::Result<()> {
    let _multipart = MultipartForm::new();
    let erased = CustomTransport::new(FacadeTransport);
    anyhow::ensure!(format!("{erased:?}") == "CustomTransport { .. }");

    let runtime = HttpRuntime::from_transport(FacadeTransport);
    let client = openai::Client::builder()
        .api_key("test-key")
        .http_runtime(runtime)
        .build()?;

    anyhow::ensure!(format!("{:?}", client.http_runtime()).contains("custom"));
    Ok(())
}
