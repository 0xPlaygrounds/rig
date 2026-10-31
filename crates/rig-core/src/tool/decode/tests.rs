use serde::Deserialize;
use serde_json::json;

use super::*;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Args {
    offset: Option<usize>,
    timeout_secs: Option<u64>,
    #[serde(default)]
    delta: i32,
    #[serde(default)]
    scale: f64,
    #[serde(default)]
    raw: Value,
    #[serde(default)]
    edits: Vec<Edit>,
    mode: Option<Mode>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Edit {
    line: u32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Fast,
    Lines(u16),
    Window { first: u8 },
}

fn decoded(args: Value) -> Option<Args> {
    decode_args(&args).ok()
}

#[test]
fn an_integral_float_is_the_integer_its_field_asks_for() {
    let args = decoded(json!({
        "offset": 1.0,
        "timeout_secs": 30.0,
        "delta": -2.0,
        "edits": [{"line": 7.0}],
        "mode": {"window": {"first": 3.0}},
    }));
    let edits = vec![Edit { line: 7 }];
    let mode = Some(Mode::Window { first: 3 });
    assert_eq!(args.as_ref().map(|args| args.offset), Some(Some(1)));
    assert_eq!(args.as_ref().map(|args| args.timeout_secs), Some(Some(30)));
    assert_eq!(args.as_ref().map(|args| args.delta), Some(-2));
    assert_eq!(args.as_ref().map(|args| &args.edits), Some(&edits));
    assert_eq!(args.as_ref().map(|args| &args.mode), Some(&mode));
    let lines = decoded(json!({"mode": {"lines": 4.0}})).and_then(|args| args.mode);
    assert_eq!(lines, Some(Mode::Lines(4)));
    let fast = decoded(json!({"mode": "fast"})).and_then(|args| args.mode);
    assert_eq!(fast, Some(Mode::Fast));
}

#[test]
fn every_other_value_reaches_its_field_as_sent() {
    let args = decoded(json!({"scale": 2.0, "raw": {"at": 30.0, "n": 3}, "offset": 4}));
    assert_eq!(args.as_ref().map(|args| args.scale), Some(2.0));
    let raw = json!({"at": 30.0, "n": 3});
    assert_eq!(args.as_ref().map(|args| &args.raw), Some(&raw));
    assert!(raw.pointer("/at").is_some_and(Value::is_f64));
    assert_eq!(args.map(|args| args.offset), Some(Some(4)));
}

#[test]
fn a_fraction_a_negative_unsigned_and_an_overflow_are_refused_not_rounded() {
    for args in [
        json!({"offset": 1.5}),
        json!({"offset": -1.0}),
        json!({"offset": -1}),
        json!({"offset": 1e30}),
        json!({"delta": 3e9}),
        json!({"edits": [{"line": 0.5}]}),
        json!({"mode": {"window": {"first": 300.0}}}),
    ] {
        assert_eq!(decoded(args.clone()), None, "{args}");
    }
    assert!(decoded(json!({"unknown": 1})).is_none());
}

#[test]
fn a_whole_number_is_one_a_64_bit_integer_holds() {
    let number = |value: Value| value.as_number().and_then(whole);
    assert_eq!(number(json!(30.0)), Some(Whole::Unsigned(30)));
    assert_eq!(number(json!(-0.0)), Some(Whole::Unsigned(0)));
    assert_eq!(number(json!(-2.0)), Some(Whole::Signed(-2)));
    assert_eq!(number(json!(u64::MAX)), Some(Whole::Unsigned(u64::MAX)));
    assert_eq!(number(json!(i64::MIN)), Some(Whole::Signed(i64::MIN)));
    assert_eq!(number(json!(18_446_744_073_709_551_616.0)), None);
    assert_eq!(number(json!(-9_223_372_036_854_777_856.0)), None);
    assert_eq!(number(json!(1e30)), None);
    assert_eq!(number(json!(1.5)), None);
}
