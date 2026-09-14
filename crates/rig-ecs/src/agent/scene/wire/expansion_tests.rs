use super::*;

#[test]
fn preflight_and_expansion_charge_repeated_raw_arrays_identically() {
    let mut assets = BinaryAssets::default();
    let id = assets.insert(vec![1, 2]).expect("small asset");
    let reference = marker(PartSource::Binary {
        id,
        encoding: BinaryEncoding::Raw,
    })
    .expect("marker");
    let value = Value::Array(vec![reference.clone(), reference]);
    for nodes in [8, 9] {
        let mut before = Budget { bytes: 1000, nodes };
        let mut after = Budget { bytes: 1000, nodes };
        let preflight = check_expansion(&value, &assets, &mut before, 0);
        let actual = expand(value.clone(), &assets, &mut after, 0);
        assert_eq!(preflight.is_ok(), nodes == 9);
        assert_eq!(preflight.is_ok(), actual.is_ok());
        assert_eq!((before.bytes, before.nodes), (after.bytes, after.nodes));
    }
}
