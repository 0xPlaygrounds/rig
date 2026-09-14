use super::*;

#[test]
fn bounds_generic_deserialization_before_building_the_tree() {
    let input = format!("{}0{}", "[".repeat(66), "]".repeat(66));
    let error = deserialize(&mut serde_json::Deserializer::from_str(&input)).unwrap_err();
    assert!(error.to_string().contains("scene depth limit"));
    let input = r#"{"same":0,"same":1}"#;
    let error = deserialize(&mut serde_json::Deserializer::from_str(input)).unwrap_err();
    assert!(error.to_string().contains("duplicate scene object key"));
}

#[test]
fn enforces_node_and_byte_limits_during_visitation() {
    let mut budget = Budget {
        bytes: 100,
        nodes: 2,
    };
    assert!(
        Seed {
            budget: &mut budget,
            depth: 0
        }
        .deserialize(&mut serde_json::Deserializer::from_str("[0,1]"))
        .unwrap_err()
        .to_string()
        .contains("node limit")
    );
    let mut budget = Budget {
        bytes: 9,
        nodes: 10,
    };
    assert!(
        Seed {
            budget: &mut budget,
            depth: 0
        }
        .deserialize(&mut serde_json::Deserializer::from_str(r#""ab""#))
        .unwrap_err()
        .to_string()
        .contains("byte limit")
    );
}
