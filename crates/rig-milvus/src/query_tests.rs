#[path = "tests/http.rs"]
mod http;

use rig_core::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
use serde_json::json;

#[tokio::test]
async fn searches_send_a_batch_containing_the_query_vector() {
    for id_only in [false, true] {
        let embedding = json!({"data":[{"index":0,"embedding":[1.0,0.0]}],"usage":{"prompt_tokens":1,"total_tokens":1}});
        let search = json!({"code":0,"data":[{"id":7,"distance":1.0,"document":"alpha"}]});
        let server = http::Server::start(vec![
            (200, embedding.to_string()),
            (200, search.to_string()),
        ]);
        assert!(server.is_ok());
        let Ok(server) = server else { return };
        let store = server.store();
        let req = VectorSearchRequest::builder()
            .query("alpha")
            .samples(1)
            .build();
        if id_only {
            let found = store.top_n_ids(req).await;
            assert!(found.is_ok(), "{found:?}");
            assert_eq!(
                found
                    .ok()
                    .and_then(|hits| hits.into_iter().next())
                    .map(|it| (it.id, it.score)),
                Some(("7".into(), 1.0))
            );
        } else {
            let found = store.top_n::<String>(req).await;
            assert!(found.is_ok(), "{found:?}");
            assert_eq!(
                found
                    .ok()
                    .and_then(|hits| hits.into_iter().next())
                    .map(|it| (it.id, it.score, it.document)),
                Some(("7".into(), 1.0, "alpha".into()))
            );
        }
        let requests = server.finish();
        assert!(requests.is_ok(), "{requests:?}");
        let Ok(requests) = requests else { return };
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests.first().map(|it| it.path.as_str()),
            Some("/embeddings")
        );
        assert_eq!(
            requests.first().and_then(|it| it.body.get("input")),
            Some(&json!(["alpha"]))
        );
        assert_eq!(
            requests.get(1).map(|it| it.path.as_str()),
            Some("/v2/vectordb/entities/search")
        );
        let fields = if id_only {
            json!(["id", "distance"])
        } else {
            json!(["id", "distance", "document", "embeddedText"])
        };
        assert_eq!(
            requests.get(1).map(|it| &it.body),
            Some(&json!({
                "collectionName":"documents", "dbName":"default", "annsField":"embedding",
                "data":[[1.0,0.0]],"limit":1,"outputFields":fields
            }))
        );
    }
}
