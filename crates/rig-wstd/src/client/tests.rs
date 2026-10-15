use super::*;
use futures::TryStreamExt;
use http::StatusCode;

fn response(status: StatusCode, body: &'static str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("retry-after", "20")
        .body(Body::from(body))
        .unwrap()
}

#[test]
fn non_success_keeps_status_headers_and_body() {
    let error = wstd::runtime::block_on(into_response::<Bytes>(response(
        StatusCode::TOO_MANY_REQUESTS,
        "slow down",
    )))
    .err()
    .unwrap();

    assert_eq!(
        error.non_success_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.non_success_body(), Some("slow down"));
    assert_eq!(
        error
            .non_success_headers()
            .and_then(|headers| headers.get("retry-after")),
        Some(&HeaderValue::from_static("20"))
    );
}

#[test]
fn success_bodies_are_read_lazily_and_streamed() {
    wstd::runtime::block_on(async {
        let unary = into_response::<Vec<u8>>(response(StatusCode::OK, "whole"))
            .await
            .unwrap();
        assert_eq!(unary.headers().get("retry-after").unwrap(), "20");
        assert_eq!(unary.into_body().await.unwrap(), b"whole");

        let streamed = into_streaming_response(response(StatusCode::OK, "chunks"))
            .await
            .unwrap();
        let chunks: Vec<Bytes> = streamed.into_body().try_collect().await.unwrap();
        assert_eq!(chunks.concat(), b"chunks");
    });
}

#[test]
fn multipart_requests_carry_their_boundary() {
    let form = MultipartForm::new()
        .text("model", "whisper-1")
        .boundary("b0");
    let req = Request::builder()
        .uri("https://example.com/upload")
        .header(CONTENT_TYPE, "text/plain")
        .body(form)
        .unwrap();

    let req = multipart(req).unwrap();

    assert_eq!(
        req.headers().get(CONTENT_TYPE).unwrap(),
        "multipart/form-data; boundary=b0"
    );
    assert!(req.body().starts_with(b"--b0\r\n"));
}

/// rig-http's timers run on wstd's reactor here, the runtime this
/// transport needs.
#[test]
fn wasm_compat_timers_fire_on_wasi() {
    use rig_http::wasm_compat::{Elapsed, timeout};
    use std::time::Duration;

    let elapsed = wstd::runtime::block_on(timeout(
        Duration::from_millis(10),
        futures::future::pending::<()>(),
    ));
    assert_eq!(elapsed, Err(Elapsed));

    let ready = wstd::runtime::block_on(timeout(Duration::from_secs(5), async { 7 }));
    assert_eq!(ready, Ok(7));
}
