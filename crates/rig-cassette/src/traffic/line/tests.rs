use super::*;
use crate::traffic::End;
use rig_core::http_client::{StatusCode, multipart::Part};

fn exchange() -> Exchange {
    Exchange {
        number: 1,
        started_ms: 0,
        serving: None,
        method: "POST".into(),
        uri: "https://example.test/".into(),
        request_headers: HeaderMap::new(),
        request_body: RequestBody::Bytes(Bytes::new()),
        status: Some(StatusCode::OK),
        response_headers: Some(HeaderMap::new()),
        response_body: None,
        chunks: None,
        headers_at: None,
        first_byte_at: None,
        last_byte_at: None,
        ended_at: None,
        error: None,
        end: End::Complete,
    }
}

#[test]
fn compressed_streams_render_decoded_scrubbed_events() {
    use std::io::Write;
    let secret = "opaque-compressed-stream-credential";
    let source = format!("\u{feff}id: 7\rdata: {{\"echo\":\"{secret}\"}}\r\rdata: [DONE]\r\r");
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(source.as_bytes()).expect("compress");
    let compressed = encoder.finish().expect("gzip");
    let (first, last) = compressed.split_at(compressed.len() / 2);
    let mut exchange = exchange();
    exchange.request_body =
        RequestBody::Bytes(Bytes::from(json!({"client_secret":secret}).to_string()));
    let headers = exchange.response_headers.as_mut().expect("headers");
    headers.insert(
        "content-type",
        "TeXT/Event-Stream; charset=utf-8".parse().expect("type"),
    );
    headers.insert("content-encoding", "gzip".parse().expect("encoding"));
    exchange.chunks = Some(vec![
        (Duration::from_millis(10), Bytes::copy_from_slice(first)),
        (Duration::from_millis(20), Bytes::copy_from_slice(last)),
    ]);
    let line = line(&exchange, 1, &mut Bases::default());
    assert_eq!(
        line.pointer("/response/events/0/data/echo"),
        Some(&json!("[REDACTED]"))
    );
    assert_eq!(line.pointer("/response/events/0/ms"), Some(&json!(20)));
    assert_eq!(
        line.pointer("/response/events/1/data"),
        Some(&json!("[DONE]"))
    );
    assert_eq!(line.pointer("/response/events/1/id"), Some(&json!("7")));
    assert!(line.pointer("/response/body").is_none());
    assert!(!line.to_string().contains(secret));
}

#[test]
fn byte_multipart_credentials_are_learned_before_error_echoes() {
    let secret = "opaque-multipart-credential-97531";
    let mut exchange = exchange();
    exchange.request_body = RequestBody::Multipart(MultipartForm::new().part(Part::bytes(
        "client_secret",
        Bytes::from_static(secret.as_bytes()),
    )));
    exchange.status = Some(StatusCode::BAD_REQUEST);
    exchange.response_body = Some(Bytes::from(
        json!({"error":format!("invalid {secret}")}).to_string(),
    ));
    let line = line(&exchange, 1, &mut Bases::default());
    assert_eq!(
        line.pointer("/response/body/error"),
        Some(&json!("invalid [REDACTED]"))
    );
    assert!(!line.to_string().contains(secret));
}
