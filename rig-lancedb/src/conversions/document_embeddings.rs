use arrow_array::cast::AsArray;
use lancedb::arrow::arrow_schema::DataType;

impl From<arrow_array::RecordBatch> for super::DocumentEmbeddings {
    fn from(record_batch: arrow_array::RecordBatch) -> Self {
        let columns = record_batch.columns().into_iter();

        let ids = match columns.next() {
            Some(column) => match column.data_type() {
                DataType::Utf8 => column.as_string::<i32>().into_iter().collect(),
                _ => vec![],
            },
            None => vec![],
        };
        let documents = match columns.next() {
            Some(column) => match column.data_type() {
                DataType::Utf8 => column.as_string::<i32>().into_iter().collect(),
                _ => vec![],
            },
            None => vec![],
        };

        let embeddings = match columns.next() {
            Some(column) => match column.data_type() {
                DataType::List(embeddings_list) => match embeddings_list.data_type() {
                    DataType::Struct(embedding_fields) => match embedding_fields.into_iter().next()
                    {
                        Some(field) => match field.data_type() {
                            DataType::Utf8 => {}
                            _ => vec![],
                        },
                        None => vec![],
                    },
                    _ => vec![],
                },
                _ => vec![],
            },
            None => vec![],
        };

        todo!()
    }
}
