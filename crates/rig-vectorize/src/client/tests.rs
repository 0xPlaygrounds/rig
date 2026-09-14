use super::{VectorizeError, parse_api};

#[test]
fn parse_api_unwraps_successful_envelope() {
    let body = r#"{"success": true, "result": 42, "errors": [], "messages": []}"#;
    let n: u32 = parse_api(body, "query").expect("successful envelope");
    assert_eq!(n, 42);
}

#[test]
fn parse_api_surfaces_envelope_errors() {
    let body = r#"{
            "success": false,
            "result": null,
            "errors": [{"code": 7, "message": "index not found"}],
            "messages": []
        }"#;
    match parse_api::<u32>(body, "query") {
        Err(VectorizeError::ApiError { code: 7, message }) => {
            assert_eq!(message, "index not found");
        }
        other => panic!("expected ApiError, got {other:?}"),
    }
}

#[test]
fn parse_api_errors_on_missing_result() {
    let body = r#"{"success": true, "result": null, "errors": [], "messages": []}"#;
    match parse_api::<u32>(body, "upsert") {
        Err(VectorizeError::ApiError { code: 0, message }) => {
            assert_eq!(message, "No result in successful upsert response");
        }
        other => panic!("expected ApiError, got {other:?}"),
    }
}
