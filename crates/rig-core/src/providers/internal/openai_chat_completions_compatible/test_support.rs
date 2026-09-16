use bytes::Bytes;

pub(crate) fn sse_bytes_from_data_lines<T>(events: impl IntoIterator<Item = T>) -> Bytes
where
    T: AsRef<str>,
{
    Bytes::from(
        events
            .into_iter()
            .map(|event| format!("data: {}\n\n", event.as_ref()))
            .collect::<String>(),
    )
}

pub(crate) fn sse_bytes_from_json_events(events: &[serde_json::Value]) -> Bytes {
    Bytes::from(
        events
            .iter()
            .map(|event| {
                format!(
                    "data: {}\n\n",
                    serde_json::to_string(event).expect("event should serialize")
                )
            })
            .collect::<String>(),
    )
}
