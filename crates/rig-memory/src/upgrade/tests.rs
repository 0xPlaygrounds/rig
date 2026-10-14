use rig_core::conversation::{Aside, Author};
use serde_json::{Value, json};

use super::{Unread, Upgrade, Upgrades, Versioned};

fn label(old: Value) -> Result<Value, String> {
    Ok(json!([old, ""]))
}

fn refuse(_: Value) -> Result<Value, String> {
    Err("no".to_owned())
}

/// A value is read through every step from its own version; a newer one, a
/// failing step and version 0 are set aside with the reason; a plugin
/// aside's envelope reads back at its version.
#[test]
fn saved_values_and_asides_read_through_their_steps() {
    const STEPS: &[Upgrade] = &[label, label];
    let mut upgrades = Upgrades::default();
    upgrades.add("k", STEPS);
    upgrades.add("broken", &[refuse]);
    assert_eq!(upgrades.version("k"), 3);
    assert_eq!(upgrades.version("unknown"), 1);

    let saved = |v, value| Versioned { v, value };
    assert_eq!(
        upgrades.upgrade("k", saved(1, json!(7))),
        Ok(json!([[7, ""], ""]))
    );
    assert_eq!(upgrades.upgrade("k", saved(3, json!(7))), Ok(json!(7)));
    assert_eq!(
        upgrades.upgrade("k", saved(4, json!(7))),
        Err(Unread::Newer { v: 4, reads: 3 })
    );
    assert_eq!(
        upgrades.upgrade("broken", saved(1, json!(7))),
        Err(Unread::Step {
            from: 1,
            why: "no".to_owned()
        })
    );
    assert!(upgrades.upgrade("k", saved(0, json!(7))).is_err());

    let at_two = saved(2, json!({"v": 9, "value": 1})).to_aside_value();
    let aside = Aside::new("k", Author::rig(), at_two);
    assert_eq!(
        upgrades.read_aside(&aside),
        Ok(json!([{"v": 9, "value": 1}, ""]))
    );
    assert_eq!(saved(1, json!(5)).to_aside_value(), json!(5));
}
