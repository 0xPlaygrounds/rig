#[path = "tests/http.rs"]
mod http;

use rig_core::embeddings::Embedding;
use rig_core::vector_store::{InsertDocuments, VectorStoreError};
use serde_json::json;

#[tokio::test]
async fn insert_checks_the_application_response_and_keeps_error_details() {
    let cases = [
        (
            200,
            r#"{"code":0,"data":{"insertCount":1,"insertIds":[7]}}"#,
        ),
        (200, r#"{"code":100,"message":"collection missing"}"#),
        (
            200,
            r#"{"code":1804,"message":"wrong dimensions","data":{}}"#,
        ),
        (200, r#"{"code":-1,"message":"rejected"}"#),
        (401, r#"{"code":1800,"message":"unauthorized"}"#),
        (503, "temporarily unavailable"),
        (200, "not JSON"),
        (200, r#"{"data":{"insertCount":1}}"#),
    ];
    for (index, (status, body)) in cases.into_iter().enumerate() {
        let server = http::Server::start(vec![(status, body.to_owned())]);
        assert!(server.is_ok());
        let Ok(server) = server else { return };
        let result = server
            .store()
            .insert_documents(vec![(
                "alpha".to_owned(),
                vec![Embedding {
                    document: "alpha".into(),
                    vec: vec![1.0, 0.0],
                }],
            )])
            .await;
        match index {
            0 => assert!(result.is_ok(), "{result:?}"),
            1..=5 => assert!(
                matches!(result, Err(VectorStoreError::ExternalAPIError(code, ref text)) if code.as_u16() == status && text == body),
                "{result:?}"
            ),
            _ => assert!(
                matches!(result, Err(VectorStoreError::JsonError(_))),
                "{result:?}"
            ),
        }
        let requests = server.finish();
        assert!(requests.is_ok(), "{requests:?}");
        let Ok(requests) = requests else { return };
        assert_eq!(
            requests.len(),
            1,
            "inserts must not call the embedding model"
        );
        assert_eq!(
            requests.first().map(|it| it.path.as_str()),
            Some("/v2/vectordb/entities/insert")
        );
        assert_eq!(
            requests.first().map(|it| &it.body),
            Some(&json!({
                "collectionName":"documents", "dbName":"default",
                "data":[{"document":"\"alpha\"","embedded_text":"alpha","embedding":[1.0,0.0]}]
            }))
        );
    }
}
