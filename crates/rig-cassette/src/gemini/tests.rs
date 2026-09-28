use serde_json::json;

use super::*;

const CASSETTE: &str = r#"when:
  path: /v1beta/models/gemini-3.8-flash:generateContent
  method: POST
  body: '{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}'
then:
  status: 200
  body: '{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"x=1"},"thoughtSignature":"Y29kZQ=="},{"functionCall":{"id":"c1","name":"f","args":{}},"thoughtSignature":"Y2FsbA=="}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":3,"totalTokenCount":15}}'
---
when:
  path: /v1beta/models/gemini-3.8-flash:generateContent
  method: POST
  body: '{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"x=1"},"thoughtSignature":"Y29kZQ=="},{"functionCall":{"id":"c1","name":"f","args":{}},"thoughtSignature":"Y2FsbA=="}]},{"role":"user","parts":[{"functionResponse":{"id":"c1","name":"f","response":{"result":1}}}]}]}'
then:
  status: 200
  body: '{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}'
"#;

fn cassette(text: &str) -> tempfile_path::Path {
    tempfile_path::Path::new(text)
}

/// A cassette written to a unique temporary file, removed on drop.
mod tempfile_path {
    pub(super) struct Path(std::path::PathBuf);

    impl Path {
        pub(super) fn new(text: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "rig-cassette-gemini-{}-{unique}.yaml",
                std::process::id()
            ));
            std::fs::write(&path, text).expect("write the cassette");
            Self(path)
        }

        pub(super) fn get(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for Path {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

#[test]
fn a_replayed_cassette_reads_back_through_the_mirror() {
    let file = cassette(CASSETTE);
    let exchanges = Exchanges::from_cassette(file.get()).expect("parses");
    assert_eq!(exchanges.len(), 2);
    assert!(exchanges.unmodeled().is_empty());
    exchanges.assert_replayed_verbatim().expect("verbatim");
    let usage = exchanges
        .turns()
        .first()
        .and_then(Exchange::usage)
        .expect("usage");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(15));
}

#[test]
fn a_dropped_native_part_is_reported() {
    // Drop the native part from the second request only.
    let native =
        r#"{"executableCode":{"language":"PYTHON","code":"x=1"},"thoughtSignature":"Y29kZQ=="},"#;
    let at = CASSETTE.rfind(native).expect("the replayed part");
    let dropped = format!("{}{}", &CASSETTE[..at], &CASSETTE[at + native.len()..]);
    let file = cassette(&dropped);
    let exchanges = Exchanges::from_cassette(file.get()).expect("parses");
    let error = exchanges
        .assert_replayed_verbatim()
        .expect_err("the native part is missing");
    assert!(
        matches!(
            &error,
            ExchangesError::NotReplayed { request: 1, missing }
                if missing == r#""thoughtSignature":"Y29kZQ==""#
        ),
        "{error}"
    );
}

#[test]
fn scripted_replies_are_googles_documents() {
    let call = reply::function_call("lookup_order", json!({"order_id": "A-17"}), Some("c2ln"));
    let value = serde_json::to_value(&call).expect("JSON");
    assert_eq!(
        value["candidates"][0]["content"]["parts"][0],
        json!({
            "functionCall": {"id": "call-lookup_order", "name": "lookup_order", "args": {"order_id": "A-17"}},
            "thoughtSignature": "c2ln"
        })
    );
    assert_eq!(as_events("{}".to_owned()), "data: {}\r\n\r\n");
    assert_eq!(as_events("data: {}\n\n".to_owned()), "data: {}\n\n");
}

#[tokio::test]
async fn scripted_answers_in_order_and_keeps_requests() {
    let scripted = Scripted::new([reply::text("one")]);
    let request = Request::post("https://example.test/v1beta/models/x:generateContent")
        .body(Bytes::from_static(
            br#"{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}"#,
        ))
        .expect("request");
    let response = scripted
        .send::<Bytes, Bytes>(request)
        .await
        .expect("a reply");
    let body = response.into_body().await.expect("body");
    assert!(String::from_utf8_lossy(&body).contains("one"));
    let requests = scripted.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests
            .first()
            .and_then(|request| request.contents.first())
            .and_then(|content| content.parts.first())
            .and_then(|part| part.text.as_deref()),
        Some("hi")
    );
    let empty = Request::post("https://example.test")
        .body(Bytes::new())
        .expect("request");
    assert!(scripted.send::<Bytes, Bytes>(empty).await.is_err());
}
