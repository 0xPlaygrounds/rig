use serde::{Deserialize, Serialize};

use super::{Fingerprint, Format, Native, NativeData, TextBound};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Alpha {
    n: u32,
}

impl NativeData for Alpha {
    const FORMAT: Format = Format::from_static("alpha-wire");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Beta {
    n: u32,
}

impl NativeData for Beta {
    const FORMAT: Format = Format::from_static("beta-wire");
}

#[test]
fn data_reads_back_only_in_its_own_format() -> anyhow::Result<()> {
    let native = Native::new(&Alpha { n: 7 })?;
    anyhow::ensure!(native.decode::<Alpha>().transpose()? == Some(Alpha { n: 7 }));
    anyhow::ensure!(native.decode::<Beta>().is_none());
    Ok(())
}

#[test]
fn a_verbatim_value_round_trips_through_serde_unchanged() -> anyhow::Result<()> {
    let value = serde_json::json!({"type": "brand_new_item", "nested": {"x": [1, 2.5, null]}});
    let native = Native::verbatim(Format::from_static("alpha-wire"), value.clone());
    let restored: Native = serde_json::from_str(&serde_json::to_string(&native)?)?;
    anyhow::ensure!(restored == native, "the native changed");
    anyhow::ensure!(restored.value() == &value, "the value changed");
    Ok(())
}

#[test]
fn the_fingerprint_is_fnv1a_and_serializes_as_hex() -> anyhow::Result<()> {
    // Published FNV-1a 64-bit test vectors.
    anyhow::ensure!(String::from(Fingerprint::of("")) == "cbf29ce484222325");
    anyhow::ensure!(String::from(Fingerprint::of("a")) == "af63dc4c8601ec8c");
    let encoded = serde_json::to_string(&Fingerprint::of("a"))?;
    anyhow::ensure!(encoded == "\"af63dc4c8601ec8c\"");
    anyhow::ensure!(serde_json::from_str::<Fingerprint>(&encoded)? == Fingerprint::of("a"));
    anyhow::ensure!(serde_json::from_str::<Fingerprint>("\"xyz\"").is_err());
    Ok(())
}

#[test]
fn bound_data_is_read_only_for_the_text_it_was_bound_to() {
    let bound = TextBound::new("the sky is blue", vec![0_u32, 3]);
    assert_eq!(bound.for_text("the sky is blue"), Some(&vec![0, 3]));
    assert_eq!(bound.for_text("the sky is green"), None);
    assert_eq!(bound.into_for_text("edited"), None);
}
