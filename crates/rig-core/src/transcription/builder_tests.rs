use serde_json::{Value, json};

use super::*;

fn fields(request: TranscriptionRequest) -> Value {
    json!({
        "data": request.data,
        "filename": request.filename,
        "language": request.language,
        "prompt": request.prompt,
        "temperature": request.temperature,
        "additional_params": request.additional_params,
    })
}

/// Requests match what the replaced typestate builder produced for the same
/// inputs, captured before it was removed.
#[test]
fn builds_the_requests_the_typestate_builder_built() {
    let dir = assert_fs::TempDir::new().expect("temp dir");
    let path = dir.path().join("clip.wav");
    std::fs::write(&path, [9u8, 8, 7]).expect("fixture");

    let cases = [
        (
            TranscriptionRequestBuilder::new((), vec![1, 2, 3]).build(),
            json!({"data": [1, 2, 3], "filename": "file", "language": null, "prompt": null,
                   "temperature": null, "additional_params": null}),
        ),
        (
            TranscriptionRequestBuilder::new((), vec![1])
                .filename(Some("a.mp3".to_owned()))
                .language("en".to_owned())
                .prompt("ctx".to_owned())
                .temperature(0.5)
                .build(),
            json!({"data": [1], "filename": "a.mp3", "language": "en", "prompt": "ctx",
                   "temperature": 0.5, "additional_params": null}),
        ),
        (
            TranscriptionRequestBuilder::new((), vec![1])
                .additional_params(json!({"a": 1, "nested": {"x": 1}}))
                .additional_params(json!({"b": 2, "nested": {"y": 2}}))
                .build(),
            json!({"data": [1], "filename": "file", "language": null, "prompt": null,
                   "temperature": null, "additional_params": {"a": 1, "b": 2, "nested": {"y": 2}}}),
        ),
        (
            TranscriptionRequestBuilder::from_file((), &path)
                .expect("reads")
                .build(),
            json!({"data": [9, 8, 7], "filename": "clip.wav", "language": null, "prompt": null,
                   "temperature": null, "additional_params": null}),
        ),
        (
            TranscriptionRequestBuilder::from_file((), &path)
                .expect("reads")
                .filename(None)
                .build(),
            json!({"data": [9, 8, 7], "filename": "file", "language": null, "prompt": null,
                   "temperature": null, "additional_params": null}),
        ),
    ];
    for (request, expected) in cases {
        assert_eq!(fields(request), expected);
    }
}

#[test]
fn a_missing_file_reports_the_read_error() {
    let error = TranscriptionRequestBuilder::from_file((), "/nonexistent/rig/clip.wav")
        .err()
        .expect("missing file");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

/// `None` clears parameters set by earlier calls, as on every request builder.
#[test]
fn additional_params_none_clears() {
    let request = TranscriptionRequestBuilder::new((), vec![1])
        .additional_params(json!({"a": 1}))
        .additional_params(None)
        .build();
    assert_eq!(request.additional_params, None);
}
