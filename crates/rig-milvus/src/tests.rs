use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::ensure;
use rig_core::embeddings::Embedding;
use rig_core::test_utils::MockEmbeddings;
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_core::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{Filter, MilvusVectorStore};

const HIT: &str = r#"{"code":0,"cost":0,"data":[{"distance":1,"document":"{\"word\":\"alpha\"}","id":7}],"topks":[1]}"#;
const REJECTED: &str = r#"{"code":100,"message":"can't find collection"}"#;

#[derive(Debug, Deserialize, PartialEq)]
struct Word {
    word: String,
}

/// A Milvus stand-in that answers one request and returns the body it received.
struct Stub {
    store: MilvusVectorStore,
    request: JoinHandle<io::Result<Value>>,
}

impl Stub {
    fn start(status: u16, reply: &'static str) -> anyhow::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}", listener.local_addr()?);
        let request = std::thread::spawn(move || answer(&listener, status, reply));
        let mut store = MilvusVectorStore::new(
            MockEmbeddings::model(),
            url,
            "default".into(),
            "documents".into(),
        );
        store.client = reqwest::Client::builder().no_proxy().build()?;
        Ok(Self { store, request })
    }

    fn request(self) -> anyhow::Result<Value> {
        self.request
            .join()
            .map_err(|_| anyhow::anyhow!("stub panicked"))?
            .map_err(Into::into)
    }
}

fn answer(listener: &TcpListener, status: u16, reply: &str) -> io::Result<Value> {
    let (mut stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(&mut stream);
    let mut length = 0;
    let mut line = String::new();
    while reader.read_line(&mut line)? > 2 {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().map_err(io::Error::other)?;
        }
        line.clear();
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    write!(
        stream,
        "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    )?;
    serde_json::from_slice(&body).map_err(io::Error::other)
}

fn request() -> VectorSearchRequest<Filter> {
    VectorSearchRequest::builder()
        .query("alpha")
        .samples(1)
        .build()
}

async fn insert(stub: &Stub) -> Result<(), VectorStoreError> {
    let embedding = Embedding {
        document: "alpha".into(),
        vec: vec![1.0, 0.0],
    };
    stub.store
        .insert_documents(vec![("alpha".to_owned(), vec![embedding])])
        .await
}

#[tokio::test]
async fn insert_succeeds_only_on_application_code_zero() -> anyhow::Result<()> {
    for accepted in [
        r#"{"code":0,"cost":0,"data":{"insertCount":1,"insertIds":[7]}}"#,
        r#"{"code":0}"#,
    ] {
        let stub = Stub::start(200, accepted)?;
        let result = insert(&stub).await;
        ensure!(result.is_ok(), "accepted insert failed: {result:?}");
    }

    let stub = Stub::start(200, REJECTED)?;
    let result = insert(&stub).await;
    ensure!(
        matches!(&result, Err(VectorStoreError::ExternalAPIError(_, text)) if text == REJECTED),
        "rejected insert returned {result:?}"
    );

    let stub = Stub::start(200, r#"{"data":{"insertCount":1}}"#)?;
    let result = insert(&stub).await;
    ensure!(
        matches!(result, Err(VectorStoreError::JsonError(_))),
        "reply without a code returned {result:?}"
    );
    Ok(())
}

#[tokio::test]
async fn insert_without_rows_sends_no_request() -> anyhow::Result<()> {
    let stub = Stub::start(200, REJECTED)?;
    let none: Vec<(String, Vec<Embedding>)> = vec![];
    let result = stub.store.insert_documents(none).await;
    ensure!(result.is_ok(), "no documents returned {result:?}");

    let result = stub
        .store
        .insert_documents(vec![("alpha".to_owned(), vec![])])
        .await;
    ensure!(result.is_ok(), "no embeddings returned {result:?}");

    // The stub keeps the first request it gets, so this probe shows none came before.
    let probe = json!({ "probe": true });
    let url = stub.store.base_url.clone();
    stub.store.client.post(url).json(&probe).send().await?;
    let first = stub.request()?;
    ensure!(first == probe, "insert sent {first}");
    Ok(())
}

#[tokio::test]
async fn searches_send_the_query_as_a_batch_of_one() -> anyhow::Result<()> {
    let stub = Stub::start(200, HIT)?;
    stub.store.top_n::<Word>(request()).await?;
    let documents = stub.request()?;

    let stub = Stub::start(200, HIT)?;
    stub.store.top_n_ids(request()).await?;
    let ids = stub.request()?;

    for body in [documents, ids] {
        let data = body.get("data").and_then(Value::as_array);
        let first = data.and_then(|data| data.first());
        ensure!(
            data.is_some_and(|data| data.len() == 1) && first.is_some_and(Value::is_array),
            "search vector is not a batch of one: {body}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn search_surfaces_application_errors() -> anyhow::Result<()> {
    let stub = Stub::start(200, REJECTED)?;
    let documents = stub.store.top_n::<Word>(request()).await;
    let stub = Stub::start(200, REJECTED)?;
    let ids = stub.store.top_n_ids(request()).await;

    ensure!(
        matches!(&documents, Err(VectorStoreError::ExternalAPIError(_, text)) if text == REJECTED),
        "top_n returned {documents:?}"
    );
    ensure!(
        matches!(&ids, Err(VectorStoreError::ExternalAPIError(_, text)) if text == REJECTED),
        "top_n_ids returned {ids:?}"
    );
    Ok(())
}

#[tokio::test]
async fn thresholds_become_a_range_search_beside_the_caller_filter() -> anyhow::Result<()> {
    let filter = Filter::eq("kind", "noun".to_owned().into());
    let req = VectorSearchRequest::builder()
        .query("alpha")
        .samples(4)
        .threshold(0.5)
        .filter(filter.clone())
        .build();
    let stub = Stub::start(200, HIT)?;
    stub.store.top_n_ids(req).await?;
    let body = stub.request()?;

    ensure!(
        body.get("filter") == Some(&json!(filter.into_inner())),
        "the score reached the metadata filter: {body}"
    );
    let radius = body
        .pointer("/searchParams/params/radius")
        .and_then(Value::as_f64);
    // Milvus excludes the radius, so it sits just below the inclusive minimum.
    ensure!(
        radius.is_some_and(|radius| radius < 0.5 && radius as f32 == 0.5_f32.next_down()),
        "no radius just below the threshold: {body}"
    );
    Ok(())
}

#[test]
fn radius_is_the_largest_f32_below_the_rounded_threshold() {
    // 0.7 rounds down as f32, 0.5 is exact and 0.6 rounds up: each keeps its own score.
    for (threshold, score) in [(0.7, 0.7_f32), (0.5, 0.5_f32), (0.6, 0.6_f32)] {
        assert_eq!(crate::radius(threshold).ok(), Some(score.next_down()));
    }
}

#[tokio::test]
async fn thresholds_f32_cannot_hold_are_refused() -> anyhow::Result<()> {
    // f32::MIN fits, but the radius one step below it does not.
    for threshold in [
        f64::NAN,
        f64::INFINITY,
        1e300,
        -1e300,
        f64::from(f32::MIN),
        f64::from(f32::MAX).next_up(),
    ] {
        let req = VectorSearchRequest::builder()
            .query("alpha")
            .samples(1)
            .threshold(threshold)
            .build();
        let result = Stub::start(200, HIT)?.store.top_n_ids(req).await;
        ensure!(
            matches!(result, Err(VectorStoreError::FilterError(_))),
            "threshold {threshold} returned {result:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn searches_request_only_stored_fields() -> anyhow::Result<()> {
    let stub = Stub::start(200, HIT)?;
    stub.store.top_n::<Word>(request()).await?;
    let documents = stub.request()?;
    ensure!(
        documents.get("outputFields") == Some(&json!(["document"])),
        "{documents}"
    );

    let stub = Stub::start(200, HIT)?;
    stub.store.top_n_ids(request()).await?;
    let ids = stub.request()?;
    ensure!(ids.get("outputFields") == Some(&json!([])), "{ids}");
    Ok(())
}

#[tokio::test]
async fn top_n_decodes_the_stored_document() -> anyhow::Result<()> {
    let stub = Stub::start(200, HIT)?;
    let hits = stub.store.top_n::<Word>(request()).await?;
    let hit = hits.first();
    ensure!(
        hit.map(|hit| (hit.id.as_str(), &hit.document.word[..])) == Some(("7", "alpha")),
        "{} hits",
        hits.len()
    );
    Ok(())
}

#[tokio::test]
async fn top_n_decodes_json_values_and_refuses_plain_text() -> anyhow::Result<()> {
    let quoted = r#"{"code":0,"data":[{"distance":1,"document":"\"x\"","id":7}]}"#;
    let hits = Stub::start(200, quoted)?
        .store
        .top_n::<String>(request())
        .await?;
    let document = hits.first().map(|hit| hit.document.as_str());
    ensure!(document == Some("x"), "{document:?}");

    let object = r#"{"code":0,"data":[{"distance":1,"document":{"word":"alpha"},"id":7}]}"#;
    let hits = Stub::start(200, object)?
        .store
        .top_n::<Word>(request())
        .await?;
    let document = hits.first().map(|hit| hit.document.word.as_str());
    ensure!(document == Some("alpha"), "{document:?}");

    let text = r#"{"code":0,"data":[{"distance":1,"document":"plain text","id":7}]}"#;
    let result = Stub::start(200, text)?
        .store
        .top_n::<String>(request())
        .await;
    ensure!(
        matches!(result, Err(VectorStoreError::JsonError(_))),
        "plain text returned {result:?}"
    );
    Ok(())
}
