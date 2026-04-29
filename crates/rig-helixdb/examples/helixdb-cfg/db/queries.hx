QUERY InsertVector (vector: [F64], doc: String, json_payload: String) =>
    AddV<Document>(vector, { doc: doc, json_payload: json_payload })
    RETURN doc

QUERY VectorSearch(vector: [F64], limit: U64, threshold: F64) =>
    vec_docs <- SearchV<Document>(vector, limit)
    RETURN vec_docs
