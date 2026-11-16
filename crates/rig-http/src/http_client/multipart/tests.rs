use super::*;

#[test]
fn test_multipart_encoding() {
    let form = MultipartForm::new()
        .text("field1", "value1")
        .text("field2", "value2");

    let (boundary, body) = form.encode();
    let body_str = String::from_utf8_lossy(&body);

    assert!(body_str.contains("field1"));
    assert!(body_str.contains("value1"));
    assert!(body_str.contains(&boundary));
}

#[test]
fn test_file_part() {
    let form = MultipartForm::new().file(
        "upload",
        "test.txt",
        "text/plain".parse().unwrap(),
        Bytes::from("file contents"),
    );

    let (_, body) = form.encode();
    let body_str = String::from_utf8_lossy(&body);

    assert!(body_str.contains("filename=\"test.txt\""));
    assert!(body_str.contains("Content-Type: text/plain"));
    assert!(body_str.contains("file contents"));
}

#[test]
fn filename_with_quote_and_crlf_cannot_add_a_header_line() {
    let form = MultipartForm::new().boundary("b").file(
        "file",
        "a\".wav\r\nX-Injected: yes",
        mime::APPLICATION_OCTET_STREAM,
        &b"data"[..],
    );
    let (_, body) = form.encode();
    let text = String::from_utf8_lossy(&body).into_owned();
    assert!(
        !text.contains("\r\nX-Injected: yes"),
        "filename broke out of its header:\n{text}"
    );
}

#[test]
fn names_and_filenames_escape_quotes_and_line_breaks() {
    let form = MultipartForm::new().boundary("b").file(
        "fi\"le\r\n",
        "a\".wav\r\nX",
        mime::APPLICATION_OCTET_STREAM,
        &b"data"[..],
    );
    let (_, body) = form.encode();
    let text = String::from_utf8_lossy(&body).into_owned();
    assert!(
        text.contains("name=\"fi%22le%0D%0A\"; filename=\"a%22.wav%0D%0AX\"\r\n"),
        "{text}"
    );
}
