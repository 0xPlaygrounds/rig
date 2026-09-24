use super::*;

#[test]
fn a_response_serializes_its_output_beside_its_metadata() {
    let response = Response {
        output: vec![1u8, 2],
        meta: ResponseMeta {
            model: Some(ModelName::new("m").unwrap()),
            usage: Usage {
                input_tokens: Some(3),
                ..Usage::default()
            },
            ..ResponseMeta::new(ProviderName::new("openai").unwrap())
        },
    };
    let encoded = serde_json::to_value(&response).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({
            "output": [1, 2],
            "meta": {"provider": "openai", "model": "m", "usage": {"input_tokens": 3}}
        })
    );
    assert_eq!(
        serde_json::from_value::<Response<Vec<u8>>>(encoded).unwrap(),
        response
    );
}
