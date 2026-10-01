//! Scaffolding shared by the vector-store suites: an httpmock stand-in for the
//! OpenAI embeddings endpoint, the client pointed at it, and the Docker guard.

use httpmock::{Method::POST, MockServer};
use rig::providers::openai;
use serde_json::{Value, json};

/// The three definitions most suites index, in `doc0`..`doc2` order.
pub const WORD_DEFINITIONS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];

/// Returns true, after logging why, when no Docker daemon is reachable.
pub fn skip_if_docker_unavailable(test_name: &str) -> bool {
    let docker_socket = std::path::Path::new("/var/run/docker.sock");
    if std::env::var_os("DOCKER_HOST").is_some() || docker_socket.exists() {
        return false;
    }

    eprintln!("skipping {test_name}: Docker is unavailable");
    true
}

/// A 1536-dimension unit vector along `axis`.
pub fn axis_embedding(axis: usize) -> Vec<f64> {
    let mut embedding = vec![0.0; 1536];
    embedding[axis] = 1.0;
    embedding
}

/// An OpenAI client with key `TEST` that sends to `server`.
pub fn openai_client(server: &MockServer) -> openai::OpenAI {
    openai::OpenAIConfig::new("TEST")
        .with_base_url(server.base_url())
        .client()
}

/// Mocks one OpenAI `POST /embeddings` exchange on `server`: a request with
/// key `TEST` whose JSON body equals `request` is answered with `embeddings`,
/// in order, under the model named in `request`.
pub fn mock_embeddings(
    server: &MockServer,
    request: Value,
    embeddings: impl IntoIterator<Item = Vec<f64>>,
) {
    let data: Vec<Value> = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({ "object": "embedding", "embedding": embedding, "index": index })
        })
        .collect();
    let response = json!({
        "object": "list",
        "data": data,
        "model": request["model"],
        "usage": { "prompt_tokens": 8, "total_tokens": 8 },
    });
    server.mock(|when, then| {
        when.method(POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .header("Content-Type", "application/json")
            .json_body(request);
        then.status(200)
            .header("content-type", "application/json")
            .json_body(response);
    });
}
