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
