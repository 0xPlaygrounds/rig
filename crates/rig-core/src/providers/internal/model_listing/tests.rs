use super::ListModelEntry;
use crate::model::Model;

/// `id` is the one required field: an entry that carries nothing else
/// (some OpenAI-compatible gateways omit `created`/`owned_by`) still
/// decodes, mapping the absent fields to `None` on the `Model`.
#[test]
fn minimal_entry_decodes_with_id_alone() {
    let entry: ListModelEntry =
        serde_json::from_str(r#"{"id":"gpt-test"}"#).expect("minimal entry should decode");
    let model = Model::from(entry);
    assert_eq!(model.id, "gpt-test");
    assert_eq!(model.name, None);
    assert_eq!(model.created_at, None);
    assert_eq!(model.owned_by, None);
}
