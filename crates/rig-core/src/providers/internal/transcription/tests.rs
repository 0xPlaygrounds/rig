use super::*;

fn request() -> TranscriptionRequest {
    TranscriptionRequest {
        data: vec![1, 2, 3],
        filename: "audio.mp3".to_owned(),
        language: Some("en".to_owned()),
        prompt: Some("a prompt".to_owned()),
        temperature: Some(0.5),
        additional_params: None,
    }
}

/// Form field names, in the order they are written to the wire.
fn field_names(form: &MultipartForm) -> Vec<&str> {
    form.parts().iter().map(Part::name).collect()
}

/// The encoded body, so assertions cover what is actually sent rather
/// than the builder's internal state.
fn encoded(form: MultipartForm) -> String {
    let (_, body) = form.boundary("BOUNDARY").encode();
    String::from_utf8_lossy(&body).into_owned()
}

/// The wire representation of a text field.
fn text_field(name: &str, value: &str) -> String {
    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
}

/// Each provider's form shape, as a table so a new provider is one row.
#[test]
fn form_field_shape_per_provider() {
    let cases = [
        (
            "openai/groq: model in body",
            TranscriptionFields {
                model: Some("whisper-1"),
            },
            &["model", "file", "language", "prompt", "temperature"][..],
        ),
        (
            "azure: model addressed through the URL",
            TranscriptionFields { model: None },
            &["file", "language", "prompt", "temperature"][..],
        ),
    ];

    for (case, fields, expected) in cases {
        let form = transcription_form(request(), fields).expect(case);
        assert_eq!(field_names(&form), expected, "{case}");
    }
}

#[test]
fn sends_field_values_on_the_wire() {
    let form = transcription_form(
        request(),
        TranscriptionFields {
            model: Some("whisper-1"),
        },
    )
    .expect("form should build");

    let body = encoded(form);
    for (name, value) in [
        ("model", "whisper-1"),
        ("language", "en"),
        ("prompt", "a prompt"),
        ("temperature", "0.5"),
    ] {
        assert!(body.contains(&text_field(name, value)), "{name}: {body}");
    }
    assert!(
        body.contains("name=\"file\"; filename=\"audio.mp3\""),
        "{body}"
    );
}

#[test]
fn omits_unset_optional_fields() {
    let request = TranscriptionRequest {
        data: vec![1, 2, 3],
        filename: "audio.mp3".to_owned(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    };

    let form = transcription_form(
        request,
        TranscriptionFields {
            model: Some("whisper-1"),
        },
    )
    .expect("form should build");

    assert_eq!(field_names(&form), ["model", "file"]);
}

#[test]
fn flattens_additional_params_onto_the_form() {
    let mut request = request();
    request.additional_params = Some(serde_json::json!({
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
    }));

    let form = transcription_form(
        request,
        TranscriptionFields {
            model: Some("whisper-1"),
        },
    )
    .expect("form should build");

    // String values go on the form verbatim (a JSON-quoted
    // `"verbose_json"` would be rejected or ignored by the provider);
    // non-string values stay JSON-encoded.
    let body = encoded(form);
    assert!(
        body.contains(&text_field("response_format", "verbose_json")),
        "{body}"
    );
    assert!(
        body.contains(&text_field("timestamp_granularities", "[\"word\"]")),
        "{body}"
    );
}

#[test]
fn rejects_non_object_additional_params() {
    let mut request = request();
    request.additional_params = Some(serde_json::json!("not an object"));

    let error = transcription_form(
        request,
        TranscriptionFields {
            model: Some("whisper-1"),
        },
    )
    .expect_err("non-object additional params should be rejected");

    assert!(matches!(error, TranscriptionError::RequestError(_)));
}
