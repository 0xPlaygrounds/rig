use super::*;

#[test]
fn discovers_bins_and_libraries_only_for_standalone_examples() {
    let package = serde_json::json!({"manifest_path":"/repo/examples/demo/Cargo.toml", "targets":[
        {"kind":["bin"],"name":"demo"},
        {"kind":["cdylib","rlib"],"name":"library"},
        {"kind":["example"],"name":"image", "required-features":["image","gemini"]},
        {"kind":["test"],"name":"integration"}
    ]});
    let mut ordinary = BTreeSet::new();
    collect(&package, false, &mut ordinary).unwrap();
    assert_eq!(ordinary.len(), 1);
    let example = ordinary.first().unwrap();
    assert_eq!(example.features, ["image", "gemini"]);
    let mut standalone = BTreeSet::new();
    collect(&package, true, &mut standalone).unwrap();
    assert_eq!(standalone.len(), 3);
}
