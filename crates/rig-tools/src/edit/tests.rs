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

#[test]
fn a_single_edit_is_sent_again() {
    let text = batch_failure("a.rs", 1, &[(0, "`old_text` was not found".to_owned())]);
    assert_eq!(
        text,
        "`old_text` was not found. a.rs was not changed; fix the edit and send it again."
    );
}

#[test]
fn a_batch_names_the_failed_edit_and_the_ones_that_matched() {
    let text = batch_failure(
        "a.rs",
        3,
        &[(1, "edits[1]: `old_text` was not found".to_owned())],
    );
    assert_eq!(
        text,
        "edits[1]: `old_text` was not found. No edit was applied, so a.rs was not changed. \
         Send all 3 edits again, with this one fixed; edits[0], edits[2] matched and can be \
         sent as they were."
    );
}

#[test]
fn a_batch_lists_every_failed_edit() {
    let text = batch_failure(
        "a.rs",
        3,
        &[
            (0, "edits[0]: one".to_owned()),
            (2, "edits[2]: two".to_owned()),
        ],
    );
    assert_eq!(
        text,
        "2 of the 3 edits failed:\nedits[0]: one.\nedits[2]: two.\nNo edit was applied, so \
         a.rs was not changed. Send all 3 edits again, with these fixed; edits[1] matched and \
         can be sent as it was."
    );
}
