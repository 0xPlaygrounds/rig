use std::sync::{Arc, Mutex};

use super::*;
use crate::vector_store::request::{DynamicSearchFilter, Filter};

struct TestIndex {
    queries: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct NativeFilter;

impl SearchFilter for NativeFilter {
    type Value = String;

    fn eq(_key: impl AsRef<str>, _value: Self::Value) -> Self {
        Self
    }

    fn gt(_key: impl AsRef<str>, _value: Self::Value) -> Self {
        Self
    }

    fn lt(_key: impl AsRef<str>, _value: Self::Value) -> Self {
        Self
    }

    fn and(self, _rhs: Self) -> Self {
        self
    }

    fn or(self, _rhs: Self) -> Self {
        self
    }
}

impl DynamicSearchFilter for NativeFilter {
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        filter.try_interpret(|value| match value {
            Value::String(value) => Ok(value),
            other => Err(FilterError::Expected {
                expected: "string".to_owned(),
                got: other.to_string(),
            }),
        })
    }
}

struct NativeIndex;

impl VectorStoreIndex for NativeIndex {
    type Filter = NativeFilter;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let document = serde_json::from_value(Value::Array(vec![Value::Null; 401]))?;
        Ok(vec![(0.9, "doc-1".to_owned(), document)])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![(0.9, "doc-1".to_owned())])
    }
}

impl VectorStoreIndex for TestIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.queries
            .lock()
            .expect("query recorder lock")
            .push(req.query().to_string());
        let document = serde_json::from_value(json!({ "answer": 42 }))?;
        Ok(vec![(0.9, "doc-1".to_string(), document)])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![(0.9, "doc-1".to_string())])
    }
}

#[tokio::test]
async fn vector_store_index_remains_a_tool() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let index = TestIndex {
        queries: queries.clone(),
    };
    let request = VectorSearchRequest::builder()
        .query("answer")
        .samples(1)
        .build();
    let output = <TestIndex as PortableTool>::call(&index, request)
        .await
        .expect("vector tool call should succeed");

    assert_eq!(<TestIndex as PortableTool>::NAME, "search_vector_store");
    assert_eq!(
        *queries.lock().expect("query recorder lock"),
        vec!["answer"]
    );
    assert_eq!(output.len(), 1);
    let result = output.first().expect("one vector result");
    assert_eq!(result.score, 0.9);
    assert_eq!(result.id, "doc-1");
    assert_eq!(result.document, json!({ "answer": 42 }));
}

#[tokio::test]
async fn dynamic_native_filter_preserves_backend_documents() {
    let request = VectorSearchRequest::builder()
        .query("answer")
        .samples(1)
        .filter(Filter::eq("tag", json!("example")))
        .build();

    let outcome = crate::serve::serve_inline(
        &crate::serve::ErasedHandler::new(crate::serve::adapters::RetrieveAdapter::new(
            NativeIndex,
        )),
        crate::effect::EffectKind::Retrieve {
            query: crate::effect::RetrieveQuery::TopN { req: request },
        },
    )
    .await
    .expect("dynamic vector search should succeed");
    let crate::effect::Outcome::Documents(crate::effect::RetrievedDocuments::Scored(results)) =
        outcome
    else {
        panic!("expected scored documents");
    };

    assert_eq!(results[0].2.as_array().map(Vec::len), Some(401));
}

#[test]
fn datastore_wraps_backend_errors() {
    let err = VectorStoreError::datastore(std::io::Error::other("db down"));
    assert!(matches!(err, VectorStoreError::DatastoreError(_)));
    assert_eq!(err.to_string(), "Datastore error: db down");
}
