//! Values that follow the line of a conversation, such as an agent's state
//! kept per line: asides of kind [`FOLLOWS`] by rig, `{"key", "value"}`, following the line's last entry, at most one per key and
//! entry. While the line has no entry of its own since it parted from a
//! kept branch (or has no entry at all), the value is one of the line's own
//! values (`Conversation::set_line_value`), which travel with
//! the line as its title does, and it is anchored after the line's first
//! entry of its own once it has one ([`anchor_line_values`]). A removed
//! value is kept as `null`, so it stays removed on that line.

use rig_core::conversation::{Aside, AsideId, Author, Conversation, MessageId};
use serde_json::{Map, Value};

/// The kind of the asides holding followed values:
/// `{"key": <key>, "value": …}`, by rig.
pub(super) const FOLLOWS: &str = "rig.follows";

/// The value of `key` the active line of `conversation` has: its line
/// value, else the newest aside of `key` (the latest in the line); `None` when
/// it has none, or one that says it was removed.
pub(super) fn followed(conversation: &Conversation, key: &str) -> Option<Value> {
    unwrapped(&current(conversation, key)?)
}

/// Sets the value of `key` on the active line of `conversation` to `value`,
/// `None` for a removal, unless it has it: as a line value while the line
/// has no entry of its own, else after its last entry.
pub(super) fn follow(conversation: &mut Conversation, key: &str, value: Option<&Value>) {
    set(conversation, key, envelope(value));
}

/// A followed value as kept: `{"value": …}`, or `null` when the component
/// was removed.
fn envelope(saved: Option<&Value>) -> Value {
    let Some(saved) = saved else {
        return Value::Null;
    };
    let mut envelope = Map::new();
    envelope.insert("value".to_owned(), saved.clone());
    Value::Object(envelope)
}

/// A kept value read back: `None` for `null` (removed) or no value.
fn unwrapped(envelope: &Value) -> Option<Value> {
    envelope.get("value").filter(|value| !value.is_null()).cloned()
}

/// Whether `aside` holds a followed value of `key`.
fn holds(aside: &Aside, key: &str) -> bool {
    aside.kind == FOLLOWS && aside.value.get("key").and_then(Value::as_str) == Some(key)
}

/// The followed value of `key` an aside holds, as an envelope.
fn of_aside(aside: &Aside, key: &str) -> Option<Value> {
    if !holds(aside, key) {
        return None;
    }
    let mut envelope = aside.value.as_object()?.clone();
    envelope.shift_remove("key");
    let removed = envelope.get("value").is_none_or(Value::is_null);
    Some(if removed {
        Value::Null
    } else {
        Value::Object(envelope)
    })
}

/// An aside holding `envelope` as the value of `key`, after `after`.
fn follows_aside(key: &str, envelope: &Value, after: Option<MessageId>) -> Aside {
    let mut value = match envelope {
        Value::Object(envelope) => envelope.clone(),
        _ => Map::new(),
    };
    value.insert("key".to_owned(), Value::String(key.to_owned()));
    value.entry("value").or_insert(Value::Null);
    Aside::new(FOLLOWS, Author::rig(), Value::Object(value)).after(after)
}

/// The value of `key` the active line of `conversation` has: its line value,
/// else the newest of its asides (the latest in the line); `None` when it has
/// none.
fn current(conversation: &Conversation, key: &str) -> Option<Value> {
    if let Some(value) = conversation
        .line_values()
        .and_then(|values| values.get(key))
    {
        return Some(value.clone());
    }
    let held = conversation.asides().filter(|(_, aside)| holds(aside, key));
    let newest = held
        .enumerate()
        .max_by_key(|(order, (_, aside))| (conversation.aside_position(aside), *order));
    newest.and_then(|(_, (_, aside))| of_aside(aside, key))
}

/// Whether a value set now belongs to the line itself: it has no entry, or
/// a kept branch parts at its end.
fn at_shared_end(conversation: &Conversation) -> bool {
    conversation.messages().is_empty() || conversation.parts_at_end()
}

/// Puts `envelope` as the value of `key` after `after`: in place of the
/// aside of `key` that follows exactly that entry, else a new one. Unless
/// `over`, an aside there is kept as it is.
fn anchor(
    conversation: &mut Conversation,
    key: &str,
    envelope: &Value,
    after: Option<MessageId>,
    over: bool,
) {
    let there = conversation
        .asides()
        .find(|(_, aside)| aside.after == after && of_aside(aside, key).is_some())
        .map(|(id, aside)| (id, of_aside(aside, key)));
    let aside = follows_aside(key, envelope, after);
    match there {
        Some((_, Some(value))) if value == *envelope => {}
        Some((id, _)) if over => drop(conversation.replace_aside(id, aside)),
        Some(_) => {}
        None => drop(conversation.add_aside(aside)),
    }
}

/// Sets the value of `key` on the active line to `envelope`, unless it has
/// it: a line value at a shared end, else after the last entry.
fn set(conversation: &mut Conversation, key: &str, envelope: Value) {
    if current(conversation, key).as_ref() == Some(&envelope) {
        return;
    }
    if at_shared_end(conversation) {
        conversation.set_line_value(key, Some(envelope), Author::rig());
        return;
    }
    if conversation
        .line_values()
        .is_some_and(|values| values.contains_key(key))
    {
        conversation.set_line_value(key, None, Author::rig());
    }
    let last = conversation.entries().next_back().map(|entry| entry.id);
    anchor(conversation, key, &envelope, last, true);
}

/// Anchors each of `keys` the line values hold after the last entry, once
/// the line has an entry of its own; whether it changed anything.
pub(super) fn anchor_line_values(conversation: &mut Conversation, keys: &[String]) -> bool {
    if at_shared_end(conversation) {
        return false;
    }
    let Some(values) = conversation.line_values().cloned() else {
        return false;
    };
    let last = conversation.entries().next_back().map(|entry| entry.id);
    let mut changed = false;
    for key in keys {
        if let Some(envelope) = values.get(key) {
            conversation.set_line_value(key, None, Author::rig());
            anchor(conversation, key, envelope, last, true);
            changed = true;
        }
    }
    changed
}

/// Moves the followed values a splice orphaned onto the entry before them.
/// After a splice took out `removed` and put in `with`, each of `keys`
/// whose asides followed a removed entry gets one, the value of the one
/// that followed the latest removed entry, after the entry now before them
/// (the line value when that entry is the line's shared end). One already
/// there that follows an entry the splice put in is newer and kept.
/// Whether it changed anything.
pub(super) fn refollow_after_splice(
    conversation: &mut Conversation,
    keys: &[String],
    removed: &[MessageId],
    with: &[MessageId],
) -> bool {
    let mut changed = false;
    for key in keys {
        let gone: Vec<(AsideId, usize, Value, usize)> = conversation
            .asides()
            .filter_map(|(id, aside)| {
                let order = removed.iter().position(|id| Some(*id) == aside.after)?;
                let value = of_aside(aside, key)?;
                Some((id, conversation.aside_position(aside), value, order))
            })
            .collect();
        let Some((_, position, value, _)) = gone.iter().max_by_key(|gone| gone.3).cloned() else {
            continue;
        };
        for (id, ..) in &gone {
            conversation.remove_aside(*id);
            changed = true;
        }
        let target = position.checked_sub(1).and_then(|at| conversation.id(at));
        let last = conversation.entries().next_back().map(|entry| entry.id);
        if target.is_some() && target == last && conversation.parts_at_end() {
            if current(conversation, key).as_ref() != Some(&value) {
                conversation.set_line_value(key, Some(value), Author::rig());
            }
            continue;
        }
        let newer = target.is_some_and(|target| with.contains(&target));
        anchor(conversation, key, &value, target, !newer);
    }
    changed
}
