//! Reading what older rigs saved: the versions of saved values and of
//! plugins' asides ([`Versioned`], [`Upgrades`]), and the rewrite of a
//! store's older logs in the current format ([`upgrade`]).
//!
//! A log is written in one format, named in its header, and read through a
//! chain of pure steps from any format since [`OLDEST`](crate::OLDEST); a
//! store is rewritten in the current one the first time anything writes to
//! it, never on a read. A saved value carries its own version, which a
//! plugin moves on with steps of its own ([`Upgrades::add`]); a value no
//! step reads is shelved under [`SHELVED`], never dropped.

use std::collections::BTreeMap;
use std::io;

use rig_core::conversation::{Aside, Author};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::format::{Header, LogLines, Meta, write_meta};
use crate::session::{Fold, fold_lines, read_header, read_logs};
use crate::writer::set_aside_tail;
use crate::{FORMAT, LogRewrite, Repair, SessionError, Store, VERSION};
use rig_core::now_ms;

/// One step of the saved format: reads a log of format `from` as format
/// `from + 1`. Pure; a step keeps every record's stamp.
pub(crate) struct Step {
    /// The format it reads.
    pub from: u32,
    /// The header's value read as the next format's; `None` when only its
    /// format moves on.
    pub header: Option<fn(Value) -> Result<Value, String>>,
    /// One record read as the next format's records: none (dropped), one,
    /// or several; `None` when records are unchanged.
    pub record: Option<fn(Value) -> Result<Vec<Value>, String>>,
}

/// The chain, a step from each format since [`OLDEST`](crate::OLDEST) to
/// the next.
// Every bump of the format first freezes a session the old writer wrote as
// `fixtures/rig-session-<N>/` (the ignored test
// `write_the_fixture_of_this_format` writes it), then adds the step from `N`
// here; a test reads every fixture with the current reader. Fixtures are
// never edited.
pub(crate) const STEPS: &[Step] = &[Step {
    from: 2,
    header: None,
    record: None,
}];

/// The steps that read a log of format `version` as this one.
fn steps(version: u32) -> impl Iterator<Item = &'static Step> {
    STEPS.iter().filter(move |step| step.from >= version)
}

/// `header`, of a log in format `version`, read in this format.
pub(crate) fn header_now(version: u32, header: Header) -> Result<Header, String> {
    let mut header = if steps(version).any(|step| step.header.is_some()) {
        let mut value = serde_json::to_value(header).map_err(|error| error.to_string())?;
        for step in steps(version).filter_map(|step| step.header) {
            value = step(value)?;
        }
        Header::deserialize(value).map_err(|error| error.to_string())?
    } else {
        header
    };
    header.format = Some(FORMAT.to_owned());
    Ok(header)
}

/// Gives `each` every record of `lines`, a log in format `version`, after
/// its header, in this format, with its line number: the line as stored,
/// or what the steps made of it. It stops at a `Break`; `refuse` makes the
/// error of a step that failed.
pub(crate) fn each_record<E>(
    lines: &LogLines,
    version: u32,
    mut each: impl FnMut(usize, &[u8]) -> Result<std::ops::ControlFlow<()>, E>,
    refuse: impl Fn(usize, String) -> E,
) -> Result<(), E> {
    let records = steps(version).filter_map(|step| step.record);
    let records: Vec<fn(Value) -> Result<Vec<Value>, String>> = records.collect();
    for (line, _, bytes) in lines.iter().skip(1) {
        // A record no step changes costs nothing.
        if records.is_empty() {
            if each(line, bytes)?.is_break() {
                return Ok(());
            }
            continue;
        }
        for record in record_now(&records, bytes).map_err(|why| refuse(line, why))? {
            if each(line, &record)?.is_break() {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// The record `bytes` through each of `records`, as lines.
fn record_now(
    records: &[fn(Value) -> Result<Vec<Value>, String>],
    bytes: &[u8],
) -> Result<Vec<Vec<u8>>, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let mut values = vec![value];
    for step in records {
        let mut next = Vec::new();
        for value in values {
            next.extend(step(value)?);
        }
        values = next;
    }
    let line = |value: Value| serde_json::to_vec(&value).map_err(|error| error.to_string());
    values.into_iter().map(line).collect()
}

/// The prefix of the key a saved value no step reads is moved to before its
/// key is written again: `rig.set_aside/<key>/v<N>`. Readers skip such keys
/// silently; they are kept in the file.
pub const SHELVED: &str = "rig.set_aside/";

/// A saved value and the version its type had when it was saved (1 for one
/// saved before versions). Its serde form is `{"v": n, "value": …}`, without
/// `v` at 1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Versioned {
    /// The version of its type it was saved at, from 1.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub v: u32,
    /// The value, as saved.
    pub value: Value,
}

impl Versioned {
    /// `value` at version `v` as an aside's value: `{"v": v, "value": value}`,
    /// or `value` alone at 1.
    pub fn to_aside_value(&self) -> Value {
        if self.v == 1 {
            return self.value.clone();
        }
        let mut envelope = Map::new();
        envelope.insert("v".to_owned(), Value::from(self.v));
        envelope.insert("value".to_owned(), self.value.clone());
        Value::Object(envelope)
    }

    /// An aside's value read as [`to_aside_value`](Self::to_aside_value)
    /// writes it: the envelope's version and value, or the value whole at 1.
    fn of_aside_value(value: &Value) -> Self {
        let enveloped = value.as_object().filter(|object| object.len() == 2);
        let read = enveloped.and_then(|object| {
            let v = object.get("v")?.as_u64()?;
            let v = u32::try_from(v).ok().filter(|v| *v > 1)?;
            Some(Self {
                v,
                value: object.get("value")?.clone(),
            })
        });
        read.unwrap_or_else(|| Self {
            v: 1,
            value: value.clone(),
        })
    }
}

fn one() -> u32 {
    1
}

fn is_one(v: &u32) -> bool {
    *v == 1
}

/// A value saved before its type had versions: version 1.
impl From<Value> for Versioned {
    fn from(value: Value) -> Self {
        Self { v: 1, value }
    }
}

/// A pure step reading version `n` of a saved value as version `n + 1`.
pub type Upgrade = fn(Value) -> Result<Value, String>;

/// How to read older versions of saved values and of plugins' asides, by
/// key (a saved value's type path, an aside's kind).
#[derive(Clone, Debug, Default)]
pub struct Upgrades(BTreeMap<String, &'static [Upgrade]>);

/// Why a saved value is set aside, not read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unread {
    /// A newer rig saved it, at a version this one does not read.
    #[error("saved at version {v} by a newer rig; this one reads up to {reads}")]
    Newer {
        /// The version it was saved at.
        v: u32,
        /// The newest version this build reads.
        reads: u32,
    },
    /// The step from version `from` failed, or there is none.
    #[error("cannot read version {from}: {why}")]
    Step {
        /// The version the step reads.
        from: u32,
        /// Why it failed.
        why: String,
    },
}

impl Upgrades {
    /// `key`'s values are at version `steps.len() + 1`; `steps[i]` reads
    /// version `i + 1` as `i + 2`. Added again, the later steps win.
    pub fn add(&mut self, key: impl Into<String>, steps: &'static [Upgrade]) {
        self.0.insert(key.into(), steps);
    }

    /// The version `key`'s values are written at: 1 when none was added.
    pub fn version(&self, key: &str) -> u32 {
        let steps = self.0.get(key).map_or(0, |steps| steps.len());
        u32::try_from(steps).map_or(u32::MAX, |steps| steps.saturating_add(1))
    }

    /// `saved` read at this version through each step from its own, or why
    /// it is set aside.
    ///
    /// # Errors
    ///
    /// [`Unread::Newer`] for a version newer than this one, [`Unread::Step`]
    /// when a step fails or the version is 0.
    pub fn upgrade(&self, key: &str, saved: Versioned) -> Result<Value, Unread> {
        let reads = self.version(key);
        if saved.v > reads {
            return Err(Unread::Newer { v: saved.v, reads });
        }
        let steps = self.0.get(key).copied().unwrap_or_default();
        let first = saved.v.checked_sub(1).ok_or_else(|| Unread::Step {
            from: 0,
            why: "there is no version 0".to_owned(),
        })?;
        let mut value = saved.value;
        let skipped = steps
            .iter()
            .skip(usize::try_from(first).unwrap_or(usize::MAX));
        for (from, step) in (saved.v..).zip(skipped) {
            value = step(value).map_err(|why| Unread::Step { from, why })?;
        }
        Ok(value)
    }

    /// A plugin aside's value at this version: `aside.value` read as
    /// `{"v": n, "value": …}` (a value without that envelope is version 1,
    /// whole), upgraded under the aside's kind.
    ///
    /// # Errors
    ///
    /// As [`upgrade`](Self::upgrade).
    pub fn read_aside(&self, aside: &Aside) -> Result<Value, Unread> {
        self.upgrade(&aside.kind, Versioned::of_aside_value(&aside.value))
    }
}

/// Why a saved value could not be read as a type.
#[derive(Debug, thiserror::Error)]
pub enum ValueError {
    /// It was set aside: no step reads its version.
    #[error(transparent)]
    Unread(#[from] Unread),
    /// At this version it does not parse as the type.
    #[error("the saved value does not parse: {0}")]
    Parse(#[from] serde_json::Error),
}

impl From<Unread> for rig_core::error::RigError {
    fn from(error: Unread) -> Self {
        Self::from_error(rig_core::error::ErrorKind::Other, &error)
    }
}

impl From<ValueError> for rig_core::error::RigError {
    fn from(error: ValueError) -> Self {
        Self::from_error(rig_core::error::ErrorKind::Other, &error)
    }
}

/// The logs [`upgrade`] rewrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Upgraded {
    /// Each log rewritten, by name, with the format version it had.
    pub logs: Vec<(String, u32)>,
    /// The torn tails it set aside first ([`Repair::TornTail`]).
    pub repairs: Vec<Repair>,
}

/// Where [`upgrade`] keeps the log `log` of format `version` until every
/// log is replaced.
fn kept_copy(log: &str, version: u32) -> String {
    format!("{UPGRADED}{log}.v{version}.jsonl")
}

/// The folder of the store's files [`upgrade`] keeps old logs in.
const UPGRADED: &str = "rig/upgraded/";

/// No images: an upgrade reads no blob.
fn no_images(_: &str) -> io::Result<Vec<u8>> {
    Err(io::ErrorKind::NotFound.into())
}

/// Rewrites every log of `store` older than this format in this format,
/// through the chain. A log's torn tail is set aside first (its bytes in
/// `recovered/`), so the upgrade never drops it. Each log is then kept as
/// `rig/upgraded/<log>.v<N>.jsonl` and replaced atomically
/// ([`Store::replace_log`]) with its lineage and stamps unchanged and a
/// rewrite by rig in its header; the kept copies are removed once every log
/// is replaced. A store whose logs are all in this format is left. Nothing
/// is written when any log fails to read, is newer than this format, or
/// would not read back. A crash midway leaves every log whole, old or new,
/// and the copies; the next call finishes.
///
/// Every write of rig-memory to a session upgrades it first
/// ([`Writer::open`](crate::Writer::open), [`Session::save`](crate::Session::save),
/// [`save`](crate::save), [`rewrite`](crate::rewrite)); reading never does.
///
/// # Errors
///
/// As [`Session::read`](crate::Session::read), or when a write fails.
pub fn upgrade(store: &dyn Store) -> Result<Upgraded, SessionError> {
    upgrade_logs(store).map(|(upgraded, _)| upgraded)
}

/// As [`upgrade`], and every log of `store` as it leaves them, in the
/// store's order.
pub(crate) fn upgrade_logs(
    store: &dyn Store,
) -> Result<(Upgraded, Vec<(String, LogLines)>), SessionError> {
    let mut logs = read_logs(store)?;
    let mut headers = Vec::new();
    for (at, (name, lines)) in logs.iter().enumerate() {
        // A log holding no complete line is set aside by its writer.
        let Some((_, _, first)) = lines.iter().next() else {
            continue;
        };
        let (stamp, header, version) = read_header(name, first)?;
        headers.push((name.as_str(), at, stamp, header, version));
    }
    if headers.iter().all(|(.., version)| *version == VERSION) {
        // What a crash after the last log was replaced left.
        remove_kept_copies(store)?;
        return Ok((Upgraded::default(), logs));
    }
    headers.sort_by_key(|(name, ..)| *name);

    // Every log read strictly, and each old one written in this format and
    // read back, before anything is written.
    let t = now_ms();
    let mut replaced = Vec::new();
    for (name, at, stamp, header, version) in headers {
        let Some((_, lines)) = logs.get(at) else {
            continue;
        };
        let Some(old) = fold_lines(name, lines, Fold::default(), &no_images)? else {
            continue;
        };
        if version == VERSION {
            continue;
        }
        let mut header = header;
        header.rewrites.push(LogRewrite {
            t,
            by: Author::rig(),
        });
        let mut bytes = Vec::with_capacity(lines.prefix(u64::MAX).len() + 64);
        write_meta(&mut bytes, stamp, &Meta::Header(header)).map_err(io::Error::from)?;
        let refuse = |line, why| SessionError::Record {
            log: name.to_owned(),
            line,
            why,
        };
        each_record(
            lines,
            version,
            |_, record| {
                bytes.extend_from_slice(record);
                bytes.push(b'\n');
                Ok(std::ops::ControlFlow::Continue(()))
            },
            refuse,
        )?;
        let new = LogLines::parse(bytes);
        // No step changes a record: the copy holds the same records.
        let same = steps(version).all(|step| step.record.is_none()) || {
            let back = fold_lines(name, &new, Fold::default(), &no_images)?;
            back.as_ref().is_some_and(|back| {
                back.conversation == old.conversation
                    && back.state == old.state
                    && back.retired == old.retired
            })
        };
        if !same {
            let why = format!("in {FORMAT} it would not read back the same");
            return Err(SessionError::Record {
                log: name.to_owned(),
                line: 1,
                why,
            });
        }
        replaced.push((name.to_owned(), at, version, new, old.torn));
    }

    // Nothing was written so far.
    let mut upgraded = Upgraded::default();
    for (name, _, _, _, torn) in &replaced {
        if let Some((kept, cut)) = *torn {
            let file = set_aside_tail(store, name, kept)?;
            upgraded.repairs.push(Repair::TornTail {
                log: name.clone(),
                kept,
                cut,
                set_aside: Some(file),
            });
        }
    }
    for (name, _, version, ..) in &replaced {
        store.write_file(&kept_copy(name, *version), &store.read_log(name)?)?;
    }
    for (name, at, version, new, _) in replaced {
        store.replace_log(&name, new.prefix(u64::MAX))?;
        upgraded.logs.push((name, version));
        if let Some((_, lines)) = logs.get_mut(at) {
            *lines = new;
        }
    }
    remove_kept_copies(store)?;
    Ok((upgraded, logs))
}

/// Removes the old logs [`upgrade`] kept.
fn remove_kept_copies(store: &dyn Store) -> io::Result<()> {
    for name in store.files()? {
        if name.starts_with(UPGRADED) {
            store.remove_file(&name)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
