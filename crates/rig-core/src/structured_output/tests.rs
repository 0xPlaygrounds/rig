use super::json_in_text;
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct Answer {
    value: String,
}

fn answer(value: &str) -> Option<Answer> {
    Some(Answer {
        value: value.to_owned(),
    })
}

#[test]
fn json_in_text_tolerates_fences_and_prose() {
    assert_eq!(json_in_text(r#"{"value":"x"}"#).ok(), answer("x"));
    assert_eq!(
        json_in_text("```json\n{\"value\":\"y\"}\n```").ok(),
        answer("y")
    );
    assert_eq!(
        json_in_text("Here you go: {\"value\":\"z\"} voilà, that is all.").ok(),
        answer("z")
    );
    assert!(json_in_text::<Answer>("no json here").is_err());
}
