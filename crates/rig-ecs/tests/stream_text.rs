//! Cursor regressions independent of provider timing.

#![allow(clippy::expect_used, reason = "test assertions")]

use bevy_ecs::prelude::*;
use rig_ecs::{bus::Streamed, stream::StreamText};

#[test]
fn interleaved_unequal_utf8_streams_have_independent_positions() {
    let mut world = World::new();
    let first = world.spawn_empty().id();
    let second = world.spawn_empty().id();
    let mut cursor = StreamText::default();
    let mut a = Streamed {
        text: "こんにちは".into(),
        ..Default::default()
    };
    let mut b = Streamed {
        text: "Hi".into(),
        ..Default::default()
    };
    assert_eq!(cursor.read(first, &a).expect("first"), "こんにちは");
    assert_eq!(cursor.read(second, &b).expect("second"), "Hi");
    b.text.push_str(" 🦀");
    assert_eq!(cursor.read(second, &b).expect("second delta"), " 🦀");
    assert_eq!(cursor.read(first, &a).expect("unchanged"), "");
    a.text.push_str("世界");
    assert_eq!(cursor.read(first, &a).expect("first delta"), "世界");
}

#[test]
fn replacement_is_explicit_and_invalid_utf8_offsets_never_panic() {
    let mut world = World::new();
    let entity = world.spawn_empty().id();
    let mut cursor = StreamText::default();
    let mut stream = Streamed {
        text: "a".into(),
        ..Default::default()
    };
    cursor.read(entity, &stream).expect("initial");
    stream.text = "🦀".into();
    assert!(cursor.read(entity, &stream).is_err());
    cursor.forget(entity);
    assert_eq!(cursor.read(entity, &stream).expect("reset"), "🦀");
    stream.text.clear();
    assert!(cursor.read(entity, &stream).is_err());
    cursor.forget(entity);
    assert_eq!(cursor.read(entity, &stream).expect("empty"), "");
}
