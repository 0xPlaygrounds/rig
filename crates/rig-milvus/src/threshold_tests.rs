#[path = "tests/http.rs"]
mod http;

use rig_core::vector_store::{
    VectorStoreIndex,
    request::{SearchFilter, VectorSearchRequest},
};
use serde_json::json;

#[tokio::test]
async fn thresholds_compare_returned_scores_and_preserve_metadata_filters() {
    for id_only in [false, true] {
        for (minimum, expected) in [
            (None, 4),
            (Some(0.5), 2),
            (Some(1.0), 1),
            (Some(2.0), 0),
            (Some(0.0), 3),
            (Some(-0.5), 4),
        ] {
            for with_filter in [false, true] {
                let embedding = json!({"data":[{"index":0,"embedding":[1.0,0.0]}],"usage":{"prompt_tokens":1,"total_tokens":1}});
                let rows: Vec<_> = [(1, 1.0), (2, 0.5), (3, 0.0), (4, -0.5)]
                    .into_iter()
                    .map(|(id, distance)| json!({"id":id,"distance":distance,"document":"alpha"}))
                    .collect();
                let search = json!({"code":0,"data":rows});
                let server = http::Server::start(vec![
                    (200, embedding.to_string()),
                    (200, search.to_string()),
                ]);
                assert!(server.is_ok());
                let Ok(server) = server else { return };
                let store = server.store();
                let filter = crate::Filter::eq("category", "science".to_owned().into());
                let mut req = VectorSearchRequest::builder().query("alpha").samples(4);
                if let Some(minimum) = minimum {
                    req = req.threshold(minimum);
                }
                if with_filter {
                    req = req.filter(filter.clone());
                }
                let req = req.build();
                let found = if id_only {
                    store
                        .top_n_ids(req)
                        .await
                        .map(|hits| hits.into_iter().map(|it| it.id).collect::<Vec<_>>())
                } else {
                    store
                        .top_n::<String>(req)
                        .await
                        .map(|hits| hits.into_iter().map(|it| it.id).collect::<Vec<_>>())
                };
                assert!(found.is_ok(), "{found:?}");
                assert_eq!(
                    found.ok(),
                    Some((1..=expected).map(|id| id.to_string()).collect())
                );
                let requests = server.finish();
                assert!(requests.is_ok(), "{requests:?}");
                let Ok(requests) = requests else { return };
                assert_eq!(
                    requests.get(1).map(|it| it.path.as_str()),
                    Some("/v2/vectordb/entities/search")
                );
                let actual = requests.get(1).and_then(|it| it.body.get("filter"));
                let expected = with_filter.then(|| json!(filter.into_inner()));
                assert_eq!(
                    actual,
                    expected.as_ref(),
                    "the score threshold must not enter the metadata expression"
                );
            }
        }
    }
}
