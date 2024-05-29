pub fn merge(a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (a.clone(), b) {
        (serde_json::Value::Object(mut a), serde_json::Value::Object(b)) => {
            b.into_iter().for_each(|(key, value)| {
                a.insert(key.clone(), value.clone());
            });
            serde_json::Value::Object(a)
        }
        _ => a,
    }
}
