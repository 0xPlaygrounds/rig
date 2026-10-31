//! Rewriting a session's saved history in place ([`rewrite`]) through a
//! [`Rewrite`]: [`Strings`] maps every string, [`Messages`] only messages.

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Change, Conversation, Lineage, ResultFacts};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::blobs::{BLOBS, blob_names, named_blobs, put_blob, read_blob, store_change};
use crate::format::{Body, Header, LogLines, LogRewrite, Meta, parse, write_change, write_meta};
use crate::session::{Fold, Folded, fold_lines, read_header, read_logs};
use crate::writer::{refused, set_aside_tail};
use crate::{Files, MemoryStore, SessionError, Stamp, Store};
use rig_core::now_ms;

/// Where a value being rewritten is kept.
#[derive(Clone, Copy, Debug)]
pub enum Place<'a> {
    /// In a record of the log `log` (an agent's id or
    /// [`SESSION`](crate::SESSION)), stamped `stamp`, made by `by`.
    Record {
        /// The log.
        log: &'a str,
        /// The record's stamp.
        stamp: Stamp,
        /// Who made the record.
        by: Option<&'a Author>,
    },
    /// In the file `name`: one of the store's, one a host keeps outside it,
    /// or a copy of either held in memory.
    File {
        /// The file.
        name: &'a str,
    },
}

/// A change applied to every value a session keeps. Each hook edits its
/// argument in place and returns whether it changed anything. Every hook
/// defaults to no change, except [`file`](Self::file), which sends JSON
/// files to [`value`](Self::value), JSON-lines files line by line, and
/// other UTF-8 to [`text`](Self::text). A hook cannot add, drop or re-kind
/// a record. Shared between owners, so it takes `&self`.
pub trait Rewrite: Send + Sync {
    /// A message of a push, a splice or a reset. Images are as stored:
    /// `blob:<sha256>.<ext>` in a push or a splice, inline in a reset;
    /// inline data put in a push or a splice is stored as a blob.
    fn message(&self, at: &Place<'_>, message: &mut Message) -> bool {
        let _ = (at, message);
        false
    }

    /// An aside put or replaced.
    fn aside(&self, at: &Place<'_>, aside: &mut Aside) -> bool {
        let _ = (at, aside);
        false
    }

    /// What the tool calls of an entry of a push, a splice or a reset
    /// published, kept beside its message.
    fn facts(&self, at: &Place<'_>, facts: &mut ResultFacts) -> bool {
        let _ = (at, facts);
        false
    }

    /// A saved value under `key`, or a JSON value of a file (`key` is its
    /// name). A value made `null` is removed.
    fn value(&self, at: &Place<'_>, key: &str, value: &mut Value) -> bool {
        let _ = (at, key, value);
        false
    }

    /// Text of a file that is not JSON.
    fn text(&self, at: &Place<'_>, text: &mut String) -> bool {
        let _ = (at, text);
        false
    }

    /// The file `name`, whole.
    fn file(&self, name: &str, bytes: &mut Vec<u8>) -> bool {
        map_file(self, name, bytes)
    }
}

/// Every string, keys included, of every value: messages, their tool
/// calls' facts and asides through their serde form, saved values, text and
/// every file. Redaction and path remapping are this. A string that is part
/// of a message's or an aside's form, such as a field name or a tag, is
/// left as it is where changing it would make the value unreadable.
///
/// ```
/// use rig_core::conversation::Author;
/// use rig_core::message::Message;
/// use rig_memory::{MemoryStore, Session, Strings, load, rewrite, save};
///
/// let store = MemoryStore::default();
/// let mut conversation = load(&store, "main")?;
/// conversation.push(Message::user("my key is sk-123"));
/// save(&store, "main", &mut conversation)?;
/// let redact = Strings(|text: &mut String| {
///     let hit = text.contains("sk-123");
///     if hit {
///         *text = text.replace("sk-123", "[REDACTED]");
///     }
///     hit
/// });
/// let rewritten = rewrite(&store, Author::plugin("redact"), &redact)?;
/// assert_eq!(rewritten.logs.len(), 1);
/// let main = Session::read(&store)?.agents.remove(0);
/// assert_eq!(main.conversation.messages(), [Message::user("my key is [REDACTED]").into()]);
/// # Ok::<(), rig_memory::SessionError>(())
/// ```
pub struct Strings<F: Fn(&mut String) -> bool + Send + Sync>(pub F);

impl<F: Fn(&mut String) -> bool + Send + Sync> Rewrite for Strings<F> {
    fn message(&self, _: &Place<'_>, message: &mut Message) -> bool {
        map_serde(message, &self.0)
    }

    fn aside(&self, _: &Place<'_>, aside: &mut Aside) -> bool {
        map_serde(aside, &self.0)
    }

    fn facts(&self, _: &Place<'_>, facts: &mut ResultFacts) -> bool {
        map_serde(facts, &self.0)
    }

    fn value(&self, _: &Place<'_>, _: &str, value: &mut Value) -> bool {
        map_strings(value, &self.0)
    }

    fn text(&self, _: &Place<'_>, text: &mut String) -> bool {
        (self.0)(text)
    }
}

/// Only messages; files are left.
pub struct Messages<F: Fn(&mut Message) -> bool + Send + Sync>(pub F);

impl<F: Fn(&mut Message) -> bool + Send + Sync> Rewrite for Messages<F> {
    fn message(&self, _: &Place<'_>, message: &mut Message) -> bool {
        (self.0)(message)
    }

    fn file(&self, _: &str, _: &mut Vec<u8>) -> bool {
        false
    }
}

/// Maps every string (keys included) of `value` through `f`; whether any
/// changed.
pub fn map_strings(value: &mut Value, f: &dyn Fn(&mut String) -> bool) -> bool {
    match value {
        Value::String(text) => f(text),
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| map_strings(item, f) | changed),
        Value::Object(map) => {
            let mut changed = map
                .values_mut()
                .fold(false, |changed, item| map_strings(item, f) | changed);
            let renamed: Vec<(String, String)> = map
                .keys()
                .filter_map(|key| {
                    let mut new = key.clone();
                    f(&mut new).then(|| (key.clone(), new))
                })
                .collect();
            if !renamed.is_empty() {
                let renamed: HashMap<String, String> = renamed.into_iter().collect();
                *map = std::mem::take(map)
                    .into_iter()
                    .map(|(key, item)| match renamed.get(&key) {
                        Some(new) => (new.clone(), item),
                        None => (key, item),
                    })
                    .collect();
                changed = true;
            }
            changed
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// Maps every string of `item`'s serde form through `f`, keys included,
/// and reads it back; whether it changed. When the whole map does not read
/// back (it changed a field name or a tag), each string is mapped on its
/// own and kept where it reads back.
fn map_serde<T: Serialize + DeserializeOwned>(
    item: &mut T,
    f: &dyn Fn(&mut String) -> bool,
) -> bool {
    let Ok(mut value) = serde_json::to_value(&*item) else {
        return false;
    };
    if !map_strings(&mut value, f) {
        return false;
    }
    if let Ok(read) = T::deserialize(&value) {
        *item = read;
        return true;
    }
    let Ok(mut kept) = serde_json::to_value(&*item) else {
        return false;
    };
    let mut leaves = Vec::new();
    strings_of(&kept, &mut Vec::new(), f, &mut leaves);
    let mut changed = false;
    for leaf in leaves {
        let mut trial = kept.clone();
        if map_leaf(&mut trial, &leaf, f)
            && let Ok(read) = T::deserialize(&trial)
        {
            *item = read;
            kept = trial;
            changed = true;
        }
    }
    changed
}

/// A step from a JSON value to one inside it.
#[derive(Clone)]
enum Step {
    Index(usize),
    Key(String),
}

/// A string of a JSON value: the value at `path`, or with `key`, the name
/// of the last step of `path`.
struct Leaf {
    path: Vec<Step>,
    key: bool,
}

/// Every string of `value` that `f` changes, each after the strings inside
/// it, so changing one leaves the paths of the rest.
fn strings_of(
    value: &Value,
    path: &mut Vec<Step>,
    f: &dyn Fn(&mut String) -> bool,
    leaves: &mut Vec<Leaf>,
) {
    let changes = |text: &str| f(&mut text.to_owned());
    match value {
        Value::String(text) if changes(text) => leaves.push(Leaf {
            path: path.clone(),
            key: false,
        }),
        Value::Array(items) => {
            for (at, item) in items.iter().enumerate() {
                path.push(Step::Index(at));
                strings_of(item, path, f, leaves);
                path.pop();
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                path.push(Step::Key(key.clone()));
                strings_of(item, path, f, leaves);
                if changes(key) {
                    leaves.push(Leaf {
                        path: path.clone(),
                        key: true,
                    });
                }
                path.pop();
            }
        }
        _ => {}
    }
}

/// Maps the string `leaf` names in `value` through `f`; whether it changed.
fn map_leaf(value: &mut Value, leaf: &Leaf, f: &dyn Fn(&mut String) -> bool) -> bool {
    let (last, parent) = match leaf.path.split_last() {
        Some((last, parent)) if leaf.key => (Some(last), parent),
        _ => (None, leaf.path.as_slice()),
    };
    let mut at = value;
    for step in parent {
        let next = match step {
            Step::Index(index) => at.get_mut(*index),
            Step::Key(key) => at.get_mut(key.as_str()),
        };
        let Some(next) = next else {
            return false;
        };
        at = next;
    }
    match (last, at) {
        (None, Value::String(text)) => f(text),
        (Some(Step::Key(key)), Value::Object(map)) => {
            let mut new = key.clone();
            if !f(&mut new) || map.contains_key(&new) || !map.contains_key(key) {
                return false;
            }
            *map = std::mem::take(map)
                .into_iter()
                .map(|(old, item)| match old == *key {
                    true => (new.clone(), item),
                    false => (old, item),
                })
                .collect();
            true
        }
        _ => false,
    }
}

/// `item` mapped through `with` by its serde form, as a JSON value of a
/// file is ([`Rewrite::value`] at `at`, under `key`): whether it changed.
/// How a host or a plugin maps what it keeps of a rewritten session, such
/// as a copy of a tool call held in memory.
///
/// # Errors
///
/// When the mapped form no longer reads as a `T`; `item` is left as it was,
/// and the owner drops it or keeps it unreadable to no one.
pub fn rewrite_value<T: Serialize + DeserializeOwned>(
    with: &dyn Rewrite,
    at: &Place<'_>,
    key: &str,
    item: &mut T,
) -> Result<bool, serde_json::Error> {
    let mut value = serde_json::to_value(&*item)?;
    if !with.value(at, key, &mut value) {
        return Ok(false);
    }
    *item = T::deserialize(value)?;
    Ok(true)
}

/// The file `name` of `files` through [`Rewrite::file`], written back
/// whole ([`Files::write_file`], atomic) when it changed: whether it did.
/// A file that is not there is left. How a host or a plugin maps a file it
/// keeps from the session outside the session's store, such as a history
/// of prompts.
///
/// # Errors
///
/// When the file cannot be read or written.
pub fn rewrite_file(files: &dyn Files, name: &str, with: &dyn Rewrite) -> io::Result<bool> {
    let mut bytes = match files.read_file(name) {
        Ok(bytes) => bytes,
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(failure) => return Err(failure),
    };
    if !with.file(name, &mut bytes) {
        return Ok(false);
    }
    files.write_file(name, &bytes)?;
    Ok(true)
}

/// The generic file pass of [`Rewrite::file`].
fn map_file<R: Rewrite + ?Sized>(with: &R, name: &str, bytes: &mut Vec<u8>) -> bool {
    let at = Place::File { name };
    if let Ok(mut value) = serde_json::from_slice::<Value>(bytes) {
        if !with.value(&at, name, &mut value) {
            return false;
        }
        let pretty = bytes.trim_ascii_end().contains(&b'\n');
        let written = match pretty {
            true => serde_json::to_vec_pretty(&value),
            false => serde_json::to_vec(&value),
        };
        let Ok(mut written) = written else {
            return false;
        };
        if bytes.ends_with(b"\n") {
            written.push(b'\n');
        }
        *bytes = written;
        return true;
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let lines: Vec<(&str, &str, Option<Value>)> = text
        .split_inclusive('\n')
        .map(|line| {
            let (body, end) = line
                .strip_suffix('\n')
                .map_or((line, ""), |body| (body, "\n"));
            (body, end, serde_json::from_str(body).ok())
        })
        .collect();
    let complete: Vec<_> = lines
        .iter()
        .filter(|(body, end, _)| !end.is_empty() && !body.trim().is_empty())
        .collect();
    let json_lines = !complete.is_empty() && complete.iter().all(|line| line.2.is_some());
    if !json_lines {
        let mut text = text.to_owned();
        let changed = with.text(&at, &mut text);
        if changed {
            *bytes = text.into_bytes();
        }
        return changed;
    }
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    for (body, end, value) in lines {
        match value {
            Some(mut value) => match with.value(&at, name, &mut value) {
                true => {
                    match serde_json::to_string(&value) {
                        Ok(line) => out.push_str(&line),
                        Err(_) => out.push_str(body),
                    }
                    changed = true;
                }
                false => out.push_str(body),
            },
            None => {
                let mut line = body.to_owned();
                changed |= with.text(&at, &mut line);
                out.push_str(&line);
            }
        }
        out.push_str(end);
    }
    if changed {
        *bytes = out.into_bytes();
    }
    changed
}

/// What [`rewrite`] did.
#[derive(Clone, Debug, Default)]
pub struct Rewritten {
    /// Each log a record of which changed, by name.
    pub logs: Vec<RewrittenLog>,
    /// The store files the rewrite changed.
    pub files: Vec<String>,
    /// The blobs no record named any more, removed.
    pub blobs_removed: Vec<String>,
}

/// A log [`rewrite`] changed.
#[derive(Clone, Debug)]
pub struct RewrittenLog {
    /// Its name: an agent's id, or [`SESSION`](crate::SESSION).
    pub log: String,
    /// How many of its records changed.
    pub records: usize,
}

/// A log's records after its header, mapped through a [`Rewrite`].
pub(crate) struct Mapped {
    /// Its header and the header's stamp.
    pub header: (Stamp, Header),
    /// Each record after the header, a line each: those unchanged as they
    /// were, byte for byte.
    pub body: Vec<u8>,
    /// How many records changed.
    pub records: usize,
}

/// Maps the records of `lines`, the log `name`, through `with`. Images put
/// inline in a push or a splice are stored in `blobs`. `None` when it holds
/// no line.
pub(crate) fn map_log(
    name: &str,
    lines: &LogLines,
    with: &dyn Rewrite,
    blobs: &dyn Store,
) -> Result<Option<Mapped>, SessionError> {
    let refuse = |line, why| SessionError::Record {
        log: name.to_owned(),
        line,
        why,
    };
    let Some((_, _, first)) = lines.iter().next() else {
        return Ok(None);
    };
    let (stamp, header) = read_header(name, first)?;
    let mut mapped = Mapped {
        header: (stamp, header),
        body: Vec::new(),
        records: 0,
    };
    for (line, _, bytes) in lines.iter().skip(1) {
        let (stamp, body) = parse(bytes).map_err(|why| refuse(line, why))?;
        match map_record(name, stamp, body, with, blobs).map_err(|why| refuse(line, why))? {
            Some(record) => {
                mapped.body.extend_from_slice(&record);
                mapped.records += 1;
            }
            None => {
                mapped.body.extend_from_slice(bytes);
                mapped.body.push(b'\n');
            }
        }
    }
    Ok(Some(mapped))
}

/// The record `body` of the log `log`, stamped `stamp`, mapped through
/// `with`, as a line; `None` when it did not change.
pub(crate) fn map_record(
    log: &str,
    stamp: Stamp,
    body: Body,
    with: &dyn Rewrite,
    blobs: &dyn Store,
) -> Result<Option<Vec<u8>>, String> {
    let mut line = Vec::new();
    match body {
        Body::Meta(Meta::State {
            key,
            value: Some(mut value),
        }) if !value.is_null() => {
            let at = Place::Record {
                log,
                stamp,
                by: None,
            };
            if !with.value(&at, &key, &mut value) {
                return Ok(None);
            }
            let state = Meta::State {
                key,
                value: Some(value),
            };
            write_meta(&mut line, stamp, &state).map_err(|failure| failure.to_string())?;
        }
        Body::Meta(_) => return Ok(None),
        Body::Change(mut recorded) => {
            let at = Place::Record {
                log,
                stamp,
                by: recorded.by.as_ref(),
            };
            if !map_change(&mut recorded.change, &at, with)? {
                return Ok(None);
            }
            let put = &mut |name: String, bytes: Vec<u8>| put_blob(blobs, &name, &bytes);
            let stored = store_change(&recorded.change, put)
                .map_err(|failure| format!("storing an image failed: {failure}"))?;
            let change = stored.as_ref().unwrap_or(&recorded.change);
            write_change(&mut line, stamp, recorded.by.as_ref(), change)
                .map_err(|failure| failure.to_string())?;
        }
    }
    Ok(Some(line))
}

/// Maps the messages, facts and asides of `change` through `with`;
/// whether any changed.
pub(crate) fn map_change(
    change: &mut Change,
    at: &Place<'_>,
    with: &dyn Rewrite,
) -> Result<bool, String> {
    Ok(match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => {
            entries.iter_mut().fold(false, |changed, entry| {
                let message = with.message(at, Arc::make_mut(&mut entry.message));
                message | with.facts(at, &mut entry.facts) | changed
            })
        }
        Change::Aside {
            aside: Some(aside), ..
        } => with.aside(at, aside),
        Change::Reset { state } => map_state(state, at, with)?,
        Change::Aside { aside: None, .. } | Change::Halt { .. } => false,
    })
}

/// Maps the messages, facts and saved asides of a reset's `state` through
/// `with`, by its serde form; whether any changed.
fn map_state(state: &mut Conversation, at: &Place<'_>, with: &dyn Rewrite) -> Result<bool, String> {
    let fail = |failure: serde_json::Error| failure.to_string();
    let mut value = serde_json::to_value(&*state).map_err(fail)?;
    let mut changed = false;
    if let Some(Value::Array(messages)) = value.get_mut("messages") {
        for message in messages {
            let mut typed = Message::deserialize(&*message).map_err(fail)?;
            if with.message(at, &mut typed) {
                *message = serde_json::to_value(&typed).map_err(fail)?;
                changed = true;
            }
        }
    }
    if let Some(Value::Array(metas)) = value.get_mut("meta") {
        for facts in metas.iter_mut().filter_map(|meta| meta.get_mut("facts")) {
            let mut typed = ResultFacts::deserialize(&*facts).map_err(fail)?;
            if with.facts(at, &mut typed) {
                *facts = serde_json::to_value(&typed).map_err(fail)?;
                changed = true;
            }
        }
    }
    if let Some(Value::Array(asides)) = value.get_mut("asides") {
        for aside in asides.iter_mut().filter_map(|pair| pair.get_mut(1)) {
            let mut typed = Aside::deserialize(&*aside).map_err(fail)?;
            if with.aside(at, &mut typed) {
                *aside = serde_json::to_value(&typed).map_err(fail)?;
                changed = true;
            }
        }
    }
    if changed {
        *state = Conversation::deserialize(&value).map_err(fail)?;
    }
    Ok(changed)
}

/// Rewrites the session in `store` through `with`, as done by `by`, in
/// place: every record stays where it is with its stamp, and every value
/// in it goes through `with` (see [`Rewrite`]). Each log a record of which
/// changed gets a fresh lineage and `{t, by}` added to its header's
/// `rewrites`.
///
/// Every log is read strictly, and every
/// changed log is encoded and folded strictly in memory, before anything
/// is written. Then torn tails
/// are set aside (as [`Writer::open`](crate::Writer::open) does), new
/// blobs are stored, each changed log is replaced
/// ([`Store::replace_log`]), every file of the store goes through
/// [`Rewrite::file`] (each written atomically), and blobs no record names
/// any more are removed last.
///
/// What a reader sees: each log is the old one or the new one, whole;
/// across logs a reader during a rewrite may see some rewritten and some
/// not, and each always reads. A crash midway leaves some logs and files
/// rewritten; running the same map again finishes it. A conversation
/// read before the rewrite no longer continues a changed log: [`save`] of
/// it gives [`SessionError::Changed`], and the caller reads it again.
///
/// Not for a session a running [`Writer`](crate::Writer) logs to: no
/// process may rewrite a running session; rig-memory cannot tell, so a
/// program rewriting a rig session first checks that no rig runs it.
/// Files kept outside the store are the program's to map with the same
/// `with` ([`rewrite_value`], [`rewrite_file`]).
///
/// # Errors
///
/// As [`Session::read`](crate::Session::read) (a session of another format, or broken, is
/// refused, nothing written), or when a rewritten log would not read
/// (nothing written), or a write fails.
///
/// [`save`]: crate::save
pub fn rewrite(
    store: &dyn Store,
    by: Author,
    with: &dyn Rewrite,
) -> Result<Rewritten, SessionError> {
    let mut logs: Vec<(String, LogLines, Folded)> = Vec::new();
    for (name, lines) in read_logs(store)? {
        if let Some(folded) = fold_lines(&name, &lines, Fold::default(), &|_| {
            Err(io::ErrorKind::NotFound.into())
        })? {
            logs.push((name, lines, folded));
        }
    }
    logs.sort_by(|a, b| a.0.cmp(&b.0));
    let t = now_ms();
    let blobs = MemoryStore::default();
    let blob = |name: &str| read_blob(&blobs, name).or_else(|_| read_blob(store, name));
    let mut rewritten = Rewritten::default();
    let mut replaced: Vec<(u64, Vec<u8>)> = Vec::new();
    for (name, lines, _) in &logs {
        let Some(mapped) = map_log(name, lines, with, &blobs)? else {
            continue;
        };
        if mapped.records == 0 {
            continue;
        }
        let (stamp, mut header) = mapped.header;
        header.lineage = Lineage::fresh();
        header.rewrites.push(LogRewrite { t, by: by.clone() });
        let mut bytes = Vec::with_capacity(mapped.body.len() + 256);
        write_meta(&mut bytes, stamp, &Meta::Header(header)).map_err(io::Error::from)?;
        bytes.extend_from_slice(&mapped.body);
        let new = LogLines::parse(bytes);
        if fold_lines(name, &new, Fold::WHOLE, &blob)?.is_none() {
            return Err(SessionError::Record {
                log: name.clone(),
                line: 1,
                why: "the rewritten log is empty".to_owned(),
            });
        }
        rewritten.logs.push(RewrittenLog {
            log: name.clone(),
            records: mapped.records,
        });
        replaced.push((lines.len(), new.prefix(u64::MAX).to_vec()));
    }

    // Nothing was written so far.
    for (name, _, old) in &logs {
        if let Some((kept, cut)) = old.torn {
            set_aside_tail(store, name, kept, cut)?;
        }
    }
    for name in blob_names(&blobs)? {
        put_blob(store, &name, &read_blob(&blobs, &name)?)?;
    }
    for (log, (len, bytes)) in rewritten.logs.iter().zip(&replaced) {
        refused(&log.log, store.replace_log(&log.log, *len, bytes))?;
    }
    for name in store.files()? {
        if !name.starts_with(BLOBS) && rewrite_file(store, &name, with)? {
            rewritten.files.push(name);
        }
    }
    rewritten.files.sort();
    let mut named = BTreeSet::new();
    for (name, lines, _) in &logs {
        let log = rewritten.logs.iter().position(|log| &log.log == name);
        let bytes = match log.and_then(|at| replaced.get(at)) {
            Some((_, bytes)) => bytes.as_slice(),
            None => lines.prefix(u64::MAX),
        };
        named_blobs(bytes, &mut named);
    }
    let mut stored = blob_names(store)?;
    stored.sort();
    for name in stored {
        if !named.contains(&name) {
            store.remove_file(&format!("{BLOBS}{name}"))?;
            rewritten.blobs_removed.push(name);
        }
    }
    Ok(rewritten)
}

#[cfg(test)]
mod tests;
