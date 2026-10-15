use a2a::{Message, Part, Role};
use rig_core::message::{DocumentMediaType, ImageMediaType, UserContent};

use crate::parts::{UnsupportedPart, user_message};

fn content(part: Part) -> Result<Vec<UserContent>, UnsupportedPart> {
    match user_message(&Message::new(Role::User, vec![part]))? {
        rig_core::message::Message::User { content } => Ok(content),
        _ => Err(UnsupportedPart::Empty),
    }
}

#[test]
fn parts_become_user_content_by_media_type() {
    let png = b"\x89PNG\r\n\x1a\nrest".to_vec();
    let cases = [
        (Part::text("hi"), Ok(vec![UserContent::text("hi")])),
        (
            Part::data(serde_json::json!({"a": 1})),
            Ok(vec![UserContent::text(r#"{"a":1}"#)]),
        ),
        (
            Part::raw(png.clone()),
            Ok(vec![UserContent::image_raw(
                png,
                Some(ImageMediaType::PNG),
                None,
            )]),
        ),
        (
            Part::raw(b"a,b".to_vec()).with_media_type("text/csv"),
            Ok(vec![UserContent::document_text(
                "a,b",
                Some(DocumentMediaType::CSV),
            )]),
        ),
        (
            Part::raw(b"%PDF-".to_vec()).with_media_type("application/pdf"),
            Ok(vec![UserContent::document_raw(
                b"%PDF-".to_vec(),
                Some(DocumentMediaType::PDF),
            )]),
        ),
        (
            Part::url("https://example.com/a.pdf").with_media_type("application/pdf"),
            Ok(vec![UserContent::document_url(
                "https://example.com/a.pdf",
                Some(DocumentMediaType::PDF),
            )]),
        ),
        (
            Part::url("https://example.com/file"),
            Err(UnsupportedPart::UntypedUrl),
        ),
        (
            Part::raw(vec![0xff, 0xfe]),
            Err(UnsupportedPart::UntypedBytes),
        ),
        (
            Part::raw(vec![1]).with_media_type("application/zip"),
            Err(UnsupportedPart::MediaType("application/zip".to_owned())),
        ),
    ];
    for (part, expected) in cases {
        assert_eq!(content(part.clone()), expected, "{part:?}");
    }
    assert_eq!(
        user_message(&Message::new(Role::User, Vec::new())).err(),
        Some(UnsupportedPart::Empty)
    );
}
