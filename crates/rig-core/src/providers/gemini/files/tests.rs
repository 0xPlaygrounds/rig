use super::*;
use crate::providers::gemini::GeminiConfig;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

fn wire() -> Files {
    GeminiConfig::new("test-key").files()
}

#[test]
fn an_upload_is_one_multipart_request() {
    let encoded = wire()
        .encode(
            FileRequest::Upload {
                bytes: b"%PDF-1.7".to_vec(),
                mime_type: "application/pdf".into(),
                display_name: Some("report".into()),
            },
            Mode::Unary,
        )
        .expect("encodes");
    let request = &encoded.request;
    assert_eq!(request.uri().path(), "/upload/v1beta/files");
    assert_eq!(
        request.uri().query(),
        Some("uploadType=multipart&key=test-key")
    );
    assert_eq!(
        request
            .headers()
            .get("Content-Type")
            .and_then(|value| value.to_str().ok()),
        Some("multipart/related; boundary=rig-gemini-upload-4f2c8a")
    );
    let Body::Bytes(body) = request.body() else {
        panic!("bytes");
    };
    let body = String::from_utf8_lossy(body);
    assert!(body.contains(r#"{"file":{"displayName":"report","mimeType":"application/pdf"}}"#));
    assert!(body.contains("Content-Type: application/pdf\r\n\r\n%PDF-1.7\r\n"));
    assert!(body.ends_with("--rig-gemini-upload-4f2c8a--\r\n"));
}

#[test]
fn a_name_that_would_retarget_the_path_is_refused() {
    for name in ["abc?x", "abc/def", "", "files/../x"] {
        assert!(file_path(name).is_err(), "{name}");
    }
    assert_eq!(
        file_path("files/abc-1").expect("a name"),
        "/v1beta/files/abc-1"
    );
}

#[tokio::test]
async fn the_verbs_decode_what_the_api_returns() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(
            r#"{"file":{"name":"files/abc","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc","state":"ACTIVE"}}"#,
        ),
        MockHttpResponse::success(r#"{"files":[{"name":"files/abc"}],"nextPageToken":"two"}"#),
        MockHttpResponse::success(r#"{"files":[{"name":"files/def"}]}"#),
        MockHttpResponse::success("{}"),
    ]);
    let files = crate::driver::Model::new(wire(), http.clone());
    let file = files
        .upload(b"bytes".to_vec(), "text/plain", None)
        .await
        .expect("uploaded");
    assert_eq!(file.name.as_deref(), Some("files/abc"));
    assert_eq!(file.state, Some(api::FileState::Active));
    let listed = files.list().await.expect("listed");
    assert_eq!(listed.len(), 2);
    files.delete("files/abc").await.expect("deleted");
}
