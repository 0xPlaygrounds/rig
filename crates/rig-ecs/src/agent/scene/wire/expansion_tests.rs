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

#[test]
fn malformed_references_and_binaries_are_request_errors() {
    let assets = BinaryAssets::default();
    let reference = marker(PartSource::Binary {
        id: BinaryId::of(&[1]),
        encoding: BinaryEncoding::Raw,
    })
    .unwrap();
    let error = expand(reference, &assets, &mut Budget::new(), 0).unwrap_err();
    assert!(matches!(
        error,
        SceneWireError::Binary(BinaryError::Missing)
    ));
    assert_eq!(ErrorReport::from(error).kind, ErrorKind::Request);

    let error = expand(
        serde_json::json!({BINARY: PartSource::Url("https://example.invalid".into())}),
        &assets,
        &mut Budget::new(),
        0,
    )
    .unwrap_err();
    assert!(matches!(error, SceneWireError::Reference(_)));
    assert_eq!(ErrorReport::from(error).kind, ErrorKind::Request);
}

#[test]
fn expansion_limits_and_internal_invariants_have_distinct_origins() {
    let error = Budget { bytes: 0, nodes: 1 }.take(1, 0).unwrap_err();
    assert!(matches!(error, SceneWireError::Limit(_)));
    assert_eq!(ErrorReport::from(error).kind, ErrorKind::Request);
    // The encoder's internal graph accessor is only given a serialized DTO.
    // A missing graph there means the implementation broke its own contract.
    let error = table(&mut Value::Null).unwrap_err();
    assert!(matches!(error, SceneWireError::Internal(_)));
    assert_eq!(ErrorReport::from(error).kind, ErrorKind::Internal);
}

#[test]
fn external_graph_and_binary_corruption_remain_request_errors() {
    let missing_graph = serde_json::json!({"format": FORMAT});
    let error = WorldScene::from_json(&serde_json::to_vec(&missing_graph).unwrap()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);

    let corrupt_binary = serde_json::json!({
        "format": FORMAT,
        "graph": {"binaries": [{
            "id": BinaryId::of(&[1]),
            "data": BASE64_STANDARD.encode([2]),
        }]},
    });
    let error = decode(corrupt_binary.clone()).unwrap_err();
    assert!(matches!(
        error,
        SceneWireError::Binary(BinaryError::Corrupt)
    ));
    let error = WorldScene::from_json(&serde_json::to_vec(&corrupt_binary).unwrap()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);

    // A graph that turns into a binary during expansion is malformed input,
    // not the encoder's missing-graph implementation invariant.
    let binary_graph = serde_json::json!({
        "format": FORMAT,
        "graph": {
            "binaries": [{"id": BinaryId::of(&[1]), "data": BASE64_STANDARD.encode([1])}],
            BINARY: PartSource::Binary {id: BinaryId::of(&[1]), encoding: BinaryEncoding::Raw},
        },
    });
    let error = WorldScene::from_json(&serde_json::to_vec(&binary_graph).unwrap()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);
}
