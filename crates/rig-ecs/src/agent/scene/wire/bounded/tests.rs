use super::*;

#[test]
fn bounds_generic_deserialization_before_building_the_tree() {
    let input = format!("{}0{}", "[".repeat(64), "]".repeat(64));
    assert!(deserialize(&mut serde_json::Deserializer::from_str(&input)).is_ok());
    let input = format!("{}0{}", "[".repeat(65), "]".repeat(65));
    assert!(deserialize(&mut serde_json::Deserializer::from_str(&input)).is_err());
    let input = r#"{"same":0,"same":1}"#;
    assert!(deserialize(&mut serde_json::Deserializer::from_str(input)).is_err());
    let input = r#"{"same":0,"different":1}"#;
    assert!(deserialize(&mut serde_json::Deserializer::from_str(input)).is_ok());
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
        .is_err()
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
        .is_err()
    );
}
