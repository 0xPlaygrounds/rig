//! The in-memory vector store's search and the vector distances, on
//! generated 384-dimension unit embeddings in 200 clusters (the shape of a
//! small embedding model's vectors over a corpus of related documents).
//! Approximate search pays off from about ten thousand documents: there it
//! scores a few percent of them and still returns about nine in ten of the
//! exact top results. On a thousand, exact search is fast and LSH misses
//! too many.
//!
//! `cargo bench -p rig-core --bench vector_search`.

use std::sync::Arc;

use divan::{Bencher, black_box};
use fastrand::Rng;
use rig_core::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use rig_core::embeddings::distance::VectorDistance;
use rig_core::embeddings::{Embedding, EmbeddingResponse};
use rig_core::error::ProviderError;
use rig_core::operation::Embedding as EmbeddingOp;
use rig_core::vector_store::in_memory_store::{InMemoryVectorIndex, InMemoryVectorStore};
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::vector_store::{IndexStrategy, VectorStoreIndex};
use rig_core::wire::Capabilities;

fn main() {
    divan::main();
}

/// Embedding width.
const DIM: usize = 384;
/// Clusters the documents and queries are drawn around.
const CLUSTERS: usize = 200;
/// Distinct queries a bench cycles through.
const QUERIES: usize = 64;
/// Per-coordinate noise around a cluster's center: a document's nearest
/// neighbors are at a cosine similarity of about 0.8.
const SPREAD: f64 = 0.04;
/// Results asked of each search.
const SAMPLES: u64 = 10;

/// `vec` scaled to unit length.
fn unit(mut vec: Vec<f64>) -> Vec<f64> {
    let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    vec.iter_mut().for_each(|x| *x /= norm);
    vec
}

fn random(rng: &mut Rng) -> Vec<f64> {
    (0..DIM).map(|_| rng.f64() * 2.0 - 1.0).collect()
}

/// A unit vector near `center`: its direction plus `spread` of noise.
fn near(rng: &mut Rng, center: &[f64], spread: f64) -> Vec<f64> {
    unit(
        center
            .iter()
            .map(|x| x + (rng.f64() * 2.0 - 1.0) * spread)
            .collect(),
    )
}

/// `n` documents and [`QUERIES`] queries around the same clusters.
fn corpus(n: usize) -> (Vec<(String, Vec<Embedding>)>, Vec<Vec<f64>>) {
    let mut rng = Rng::with_seed(0x5eed);
    let centers: Vec<Vec<f64>> = (0..CLUSTERS).map(|_| unit(random(&mut rng))).collect();
    let center = |i: usize| centers.get(i % CLUSTERS).cloned().unwrap_or_default();
    let documents = (0..n)
        .map(|i| {
            let vec = near(&mut rng, &center(i), SPREAD);
            (
                format!("doc {i}"),
                vec![Embedding {
                    document: format!("doc {i}"),
                    vec,
                }],
            )
        })
        .collect();
    let queries = (0..QUERIES)
        .map(|i| near(&mut rng, &center(i * 7), SPREAD))
        .collect();
    (documents, queries)
}

/// Embeds the query `"{i}"` to the `i`th generated query vector.
#[derive(Clone)]
struct Queries(Arc<Vec<Vec<f64>>>);

impl Transport<Local<EmbeddingOp>> for Queries {
    fn send(&self, texts: Vec<String>, _exchange: Exchange) -> Opening<Step<EmbeddingOp>> {
        let embeddings = texts
            .into_iter()
            .map(|document| {
                let vec = document
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| self.0.get(i).cloned())
                    .unwrap_or_default();
                Embedding { document, vec }
            })
            .collect();
        Opening::ready(Opened::new(futures::stream::iter([
            Ok::<_, ProviderError>(Step::End(EmbeddingResponse::new(embeddings))),
        ])))
    }
}

fn index(
    store: InMemoryVectorStore<String>,
    queries: Vec<Vec<f64>>,
) -> InMemoryVectorIndex<String> {
    let wire = Local::new("bench").with_capabilities(Capabilities::embedding(1, DIM));
    store.index(Model::new(wire, Queries(Arc::new(queries))))
}

/// The ids `index` returns for the `i`th query.
fn top(index: &InMemoryVectorIndex<String>, i: usize) -> Vec<String> {
    let request = VectorSearchRequest::builder()
        .query(i.to_string())
        .samples(SAMPLES)
        .build();
    futures::executor::block_on(index.top_n_ids(request))
        .unwrap_or_default()
        .into_iter()
        .map(|result| result.id)
        .collect()
}

/// Times `top_n_ids` over the queries in turn.
fn search(bencher: Bencher, index: &InMemoryVectorIndex<String>) {
    let mut next = 0;
    bencher.bench_local(|| {
        next = (next + 1) % QUERIES;
        let request = VectorSearchRequest::builder()
            .query(next.to_string())
            .samples(SAMPLES)
            .build();
        black_box(futures::executor::block_on(index.top_n_ids(request)))
    });
}

/// Exact search: every stored embedding is scored.
#[divan::bench(args = [1_000, 10_000, 50_000])]
fn exact(bencher: Bencher, n: usize) {
    let (documents, queries) = corpus(n);
    let store = InMemoryVectorStore::from_documents(documents);
    search(bencher, &index(store, queries));
}

/// The LSH strategy the approximate bench uses.
const LSH: IndexStrategy = IndexStrategy::LSH {
    num_tables: 16,
    num_hyperplanes: 10,
};

/// Approximate search: only the documents sharing an LSH bucket with the
/// query are scored. Prints its recall: the share of the exact top results
/// it returns.
#[divan::bench(args = [1_000, 10_000, 50_000])]
fn lsh(bencher: Bencher, n: usize) {
    let (documents, queries) = corpus(n);
    let exact = index(
        InMemoryVectorStore::from_documents(documents.clone()),
        queries.clone(),
    );
    let store = InMemoryVectorStore::builder()
        .index_strategy(LSH)
        .documents(documents)
        .build();
    let approximate = index(store, queries);
    let found: usize = (0..QUERIES)
        .map(|i| {
            let want = top(&exact, i);
            top(&approximate, i)
                .iter()
                .filter(|id| want.contains(id))
                .count()
        })
        .sum();
    eprintln!(
        "lsh {n}: recall@{SAMPLES} {:.3}",
        found as f64 / (QUERIES as f64 * SAMPLES as f64)
    );
    search(bencher, &approximate);
}

/// One pair of embeddings at each width.
fn pair(dim: usize) -> (Embedding, Embedding) {
    let mut rng = Rng::with_seed(dim as u64);
    let mut vector = || Embedding {
        document: String::new(),
        vec: (0..dim).map(|_| rng.f64() * 2.0 - 1.0).collect(),
    };
    (vector(), vector())
}

#[divan::bench(args = [384, 1536, 3072])]
fn cosine(bencher: Bencher, dim: usize) {
    let (a, b) = pair(dim);
    bencher.bench_local(|| black_box(&a).cosine_similarity(black_box(&b), false));
}
