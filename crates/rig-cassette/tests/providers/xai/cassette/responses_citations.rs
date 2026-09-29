use super::support::with_xai_cassette;
use crate::responses_citations::{
    assert_has_citations, assert_recorded_parity, assert_stream_snapshots, assert_unary_snapshots,
    search_request,
};

#[tokio::test]
async fn hosted_search_stream_and_unary_citations() {
    with_xai_cassette("responses_citations/hosted_search", |client| async move {
        let model = client.completion("grok-4-1-fast-non-reasoning");
        let (streamed, _) =
            assert_stream_snapshots(model.stream(search_request()).expect("start search")).await;
        assert_has_citations(&streamed.raw);
        let unary = model
            .call(search_request())
            .await
            .expect("unary search counterpart");
        assert_has_citations(&unary.raw);
        assert_unary_snapshots(&unary);
    })
    .await;
    assert_recorded_parity("xai", "responses_citations/hosted_search").await;
}
