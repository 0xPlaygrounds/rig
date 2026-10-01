use super::*;
use crate::message::{AssistantContent, Issuer, Message, Sealed};
use serde_json::json;

struct Alpha;
impl NativeDialect for Alpha {
    const FORMAT: WireFormat = WireFormat::from_static("alpha.v1");
    type Item = serde_json::Value;
}

struct Beta;
impl NativeDialect for Beta {
    const FORMAT: WireFormat = WireFormat::from_static("beta.v1");
    type Item = serde_json::Value;
}

#[test]
fn decodes_only_as_its_own_format() -> Result<(), serde_json::Error> {
    let item = Native::new::<Alpha>(&json!({"type": "novel", "x": 1}))?;
    assert_eq!(
        item.decode::<Alpha>().transpose()?,
        Some(json!({"type": "novel", "x": 1}))
    );
    assert!(item.decode::<Beta>().is_none());
    assert_eq!(item.kind(), Some("novel"));
    Ok(())
}

#[test]
fn serializes_sealed_with_issuer_and_format() -> Result<(), serde_json::Error> {
    let part = AssistantContent::Native(Sealed::new(
        Issuer::from("alpha"),
        Native::new::<Alpha>(&json!({"type": "novel"}))?,
    ));
    let value = serde_json::to_value(&part)?;
    assert_eq!(
        value,
        json!({"type": "native", "issuer": "alpha", "format": "alpha.v1", "item": {"type": "novel"}})
    );
    let back: AssistantContent = serde_json::from_value(value)?;
    assert_eq!(back, part);
    Ok(())
}

#[test]
fn foreign_native_only_turn_does_not_replay() -> Result<(), serde_json::Error> {
    let message = Message::Assistant {
        id: None,
        content: vec![AssistantContent::Native(Sealed::new(
            Issuer::from("alpha"),
            Native::new::<Alpha>(&json!({"type": "novel"}))?,
        ))],
    };
    let alpha = Issuer::from("alpha");
    assert!(message.replays_to(std::slice::from_ref(&alpha), Some(&Alpha::FORMAT)));
    // Another issuer, another format, or a wire with no native format.
    assert!(!message.replays_to(&[Issuer::from("beta")], Some(&Alpha::FORMAT)));
    assert!(!message.replays_to(std::slice::from_ref(&alpha), Some(&Beta::FORMAT)));
    assert!(!message.replays_to(std::slice::from_ref(&alpha), None));
    Ok(())
}
