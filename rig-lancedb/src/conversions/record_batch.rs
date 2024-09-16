use std::sync::Arc;

use arrow_array::{
    builder::{Float64Builder, ListBuilder, StringBuilder, StructBuilder},
    RecordBatch, StringArray,
};
use lancedb::arrow::arrow_schema::{ArrowError, Schema};
use rig::vector_store::VectorStoreError;

pub fn arrow_to_rig_error(e: lancedb::arrow::arrow_schema::ArrowError) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

impl TryFrom<super::DocumentEmbeddings> for RecordBatch {
    type Error = ArrowError;

    fn try_from(documents: super::DocumentEmbeddings) -> Result<Self, Self::Error> {
        let id = StringArray::from_iter_values(documents.as_iter().map(|doc| doc.id.clone()));
        let document = StringArray::from_iter_values(
            documents
                .as_iter()
                .map(|doc| serde_json::to_string(&doc.document).unwrap()),
        );

        let mut list_builder = ListBuilder::new(StructBuilder::from_fields(documents.schema(), 0));
        documents.as_iter().map(|doc| {
            let struct_builder = list_builder.values();

            doc.embeddings.clone().into_iter().for_each(|embedding| {
                struct_builder
                    .field_builder::<StringBuilder>(0)
                    .unwrap()
                    .append_value(embedding.document);
                struct_builder
                    .field_builder::<ListBuilder<Float64Builder>>(1)
                    .unwrap()
                    .append_value(embedding.vec.into_iter().map(Some).collect::<Vec<_>>());
                struct_builder.append(true); // Append the first struct
            });

            list_builder.append(true)
        });
        let embeddings = list_builder.finish();

        RecordBatch::try_new(
            Arc::new(Schema::new(documents.schema())),
            vec![Arc::new(id), Arc::new(document), Arc::new(embeddings)],
        )
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn record_batch_deserialization() {}
}
