use super::{EditArgs, batch_failure};

/// A misspelled key inside an edit, such as `replaceAll`, is refused
/// rather than ignored, which would replace one match instead of all.
#[test]
fn a_misspelled_key_inside_an_edit_is_refused() {
    let edit = serde_json::json!({"old_text": "a", "new_text": "b", "replaceAll": true});
    let args = serde_json::json!({"path": "a.rs", "edits": [edit]});
    let parsed = serde_json::from_value::<EditArgs>(args).map(|_| ());
    assert!(parsed.is_err_and(|error| error.to_string().contains("replaceAll")));
}

/// A failed edit is sent again; in a batch, every failed edit is named and
/// the whole batch is sent again.
#[test]
fn a_failure_names_every_failed_edit() {
    let single = batch_failure("a.rs", 1, &[(0, "`old_text` was not found".to_owned())]);
    assert_eq!(
        single,
        "`old_text` was not found. a.rs was not changed; fix the edit and send it again."
    );
    let one = batch_failure(
        "a.rs",
        3,
        &[(1, "edits[1]: `old_text` was not found".to_owned())],
    );
    assert_eq!(
        one,
        "edits[1]: `old_text` was not found. No edit was applied, so a.rs was not changed. \
         Send all 3 edits again, with this one fixed; edits[0], edits[2] matched and can be \
         sent as they were."
    );
    let two = batch_failure(
        "a.rs",
        3,
        &[
            (0, "edits[0]: one".to_owned()),
            (2, "edits[2]: two".to_owned()),
        ],
    );
    assert_eq!(
        two,
        "2 of the 3 edits failed:\nedits[0]: one.\nedits[2]: two.\nNo edit was applied, so \
         a.rs was not changed. Send all 3 edits again, with these fixed; edits[1] matched and \
         can be sent as it was."
    );
}
