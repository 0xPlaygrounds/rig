use super::*;

#[test]
fn an_empty_id_is_refused_by_construction_and_deserialization() {
    let error = ResponseId::new("").unwrap_err();
    assert_eq!(error.to_string(), "a response id must not be empty");
    let error = serde_json::from_str::<MessageId>("\"\"").unwrap_err();
    assert!(
        error.to_string().contains("message id must not be empty"),
        "{error}"
    );
}

#[test]
fn an_id_serializes_as_a_plain_string() {
    let id = RequestId::new("req_1").unwrap();
    assert_eq!(
        serde_json::to_value(&id).unwrap(),
        serde_json::json!("req_1")
    );
    let back: RequestId = serde_json::from_str("\"req_1\"").unwrap();
    assert_eq!(back, id);
    assert_eq!(back, "req_1");
    assert_eq!(format!("{back}"), "req_1");
    assert_eq!(format!("{back:?}"), "request id(\"req_1\")");
}

#[test]
fn an_optional_id_reads_absent_and_present() {
    #[derive(serde::Deserialize)]
    struct Holder {
        model: Option<ModelName>,
    }
    let absent: Holder = serde_json::from_str("{}").unwrap();
    assert!(absent.model.is_none());
    let present: Holder = serde_json::from_str(r#"{"model":"m"}"#).unwrap();
    assert_eq!(present.model.as_deref(), Some("m"));
    assert!(serde_json::from_str::<Holder>(r#"{"model":""}"#).is_err());
}
