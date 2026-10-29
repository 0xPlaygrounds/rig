use std::collections::BTreeMap;

use super::*;

/// A form accepted with `content`.
fn accepted(content: &[(&str, acp::ElicitationContentValue)]) -> acp::ElicitationAction {
    let content: BTreeMap<_, _> = content
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect();
    acp::ElicitationAction::Accept(acp::ElicitationAcceptAction::new().content(content))
}

#[test]
fn a_form_answers_with_what_its_question_offers_and_several_may_be_none() {
    let several = Ask::new("Which files?")
        .choice("a", "A")
        .choice("b", "B")
        .multiple();
    let none = acp::ElicitationContentValue::StringArray(Vec::new());
    let both = acp::ElicitationContentValue::StringArray(vec!["a".to_owned(), "b".to_owned()]);
    let chosen = |ask: &Ask, action| answer(ask, action).map(|answer| answer.choices);
    assert_eq!(
        chosen(&several, accepted(&[(CHOICE, none)])),
        Some(Vec::new())
    );
    assert_eq!(
        chosen(&several, accepted(&[(CHOICE, both)])),
        Some(vec!["a".to_owned(), "b".to_owned()])
    );
    assert_eq!(chosen(&several, acp::ElicitationAction::Decline), None);
    let one = Ask::new("Which one?").choice("a", "A");
    let other = acp::ElicitationContentValue::String("z".to_owned());
    assert_eq!(chosen(&one, accepted(&[(CHOICE, other)])), None);
    assert_eq!(chosen(&one, accepted(&[])), None);
}

#[test]
fn a_picker_the_client_cannot_show_says_what_to_type_instead() {
    let models = (0..12).fold(Picker::new("Model"), |picker, n| {
        picker.item(format!("model {n}"), format!("model m{n}"))
    });
    let note = unshown(&models);
    assert!(
        note.contains("/model m0, ") && note.contains("/model m9, …"),
        "{note}"
    );
    assert!(!note.contains("/model m10"), "{note}");
    let mut key = Picker::new("OpenAI API key");
    key.secret = true;
    assert!(unshown(&key).contains("in rig in a terminal, where it is hidden"));
}
