//! Copying a session: [`fork`] copies it into another store, whole or as it
//! was before one of its messages was made, each copied value through a
//! [`Rewrite`], and carries over what the asides of the copied messages say
//! now.

use std::collections::BTreeSet;
use std::io;

use rig_core::conversation::{
    Aside, AsideId, Author, AuthorKind, Change, ChangeRecord, Lineage, MessageId,
};

use crate::format::{LogLines, Meta, SESSION, write_change, write_meta};
use crate::rewrite::{Place, Rewrite, map_aside, map_log};
use crate::session::{Fold, Folded, fold_lines, log_lines};
use crate::{LoadError, MemoryStore, Stamp, Store};

/// What [`fork`] copied beyond the records before the cut.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Forked {
    /// The asides carried as they are now: put, replaced or removed after
    /// the cut.
    pub carried: usize,
}

/// A log of the copy, before it is written.
struct Copy {
    name: String,
    bytes: Vec<u8>,
    /// The copy as read back, to check the asides carried after it.
    folded: Folded,
    next_seq: u64,
}

/// An aside as it is now in the session copied, to carry after the cut.
struct Carry {
    /// The copy it goes in.
    copy: usize,
    /// The record that last set it there.
    set: Stamp,
    by: Option<Author>,
    id: AsideId,
    aside: Option<Aside>,
}

/// Copies the session in `from` into `to`, an empty store, as it was
/// before message `before` (a log and the id of a message of its
/// conversation) was made: each log up to the records made before the one
/// that made that message, leaving out a log with no change of its
/// conversation by then (such as an agent spawned later). Without
/// `before`, every log whole. A torn tail is never copied.
///
/// Each copied record's values go through `with` ([`Keep`](crate::Keep)
/// for a plain copy), each copied log gets a lineage of its own, so a
/// conversation of one never passes as a continuation of the other, and
/// each is read back strictly before anything is written. After a cut,
/// every aside that follows an entry made before it (or none) and is not
/// rig's own ([`AuthorKind::Rig`]: the kernel's marks, which describe the
/// run at the cut) is carried as it is now: put, replaced or removed by a
/// record after the copied ones, with the time and author of the record
/// that last set it, so [`as_of`](crate::as_of) the cut still shows the
/// asides as they were then. Every blob, and the session's plugin files
/// (`plugins/…`) through [`Rewrite::file`], are copied too.
///
/// # Errors
///
/// As [`Session::read`](crate::Session::read): an old or broken session is
/// refused, and nothing is written. Also when a store fails, or `before`'s
/// log never made that message.
pub fn fork(
    from: &dyn Store,
    to: &dyn Store,
    before: Option<(&str, MessageId)>,
    with: &dyn Rewrite,
) -> Result<Forked, LoadError> {
    // The cut log, read and folded once: for where the message was made,
    // then as it is now.
    let (until, mut cut) = match before {
        None => (None, None),
        Some((log, id)) => {
            let read = read_now(from, log)?;
            let made = read
                .as_ref()
                .and_then(|(_, now)| now.made.get(&id).copied());
            let Some(made) = made else {
                let why = format!("the log `{log}` made no message {id:?}");
                return Err(io::Error::new(io::ErrorKind::InvalidInput, why).into());
            };
            (Some(made.n), read.map(|read| (log, read)))
        }
    };
    let blobs = MemoryStore::default();
    let mut names = from.logs()?;
    names.sort();
    let mut copies: Vec<Copy> = Vec::new();
    let mut carries: Vec<Carry> = Vec::new();
    for name in names {
        // Each log read once; folded as it is now only after a cut, for
        // the asides carried.
        let (lines, now) = match (until, cut.take_if(|(log, _)| *log == name)) {
            (_, Some((_, (lines, now)))) => (lines, Some(now)),
            (Some(_), None) => match read_now(from, &name)? {
                Some((lines, now)) => (lines, Some(now)),
                None => continue,
            },
            (None, None) => match log_lines(from, &name)? {
                Some(lines) => (lines, None),
                None => continue,
            },
        };
        let Some(mapped) = map_log(&name, &lines, until, with, &blobs)? else {
            continue;
        };
        if until.is_some() && !mapped.changes && name != SESSION {
            continue;
        }
        let (stamp, mut header) = mapped.header;
        header.lineage = Lineage::fresh();
        let mut bytes = Vec::with_capacity(mapped.body.len() + 256);
        write_meta(&mut bytes, stamp, &Meta::Header(header)).map_err(io::Error::from)?;
        bytes.extend_from_slice(&mapped.body);
        let lines = LogLines::parse(bytes);
        let folded = fold_lines(&name, &lines, Fold::default(), &no_images)?.ok_or_else(|| {
            LoadError::Record {
                log: name.clone(),
                line: 1,
                why: "the copied log is empty".to_owned(),
            }
        })?;
        let copy = copies.len();
        if let (Some(until), Some(now)) = (until, &now) {
            carries.extend(carried(copy, &name, now, &folded, until, with));
        }
        copies.push(Copy {
            next_seq: mapped.last.seq.saturating_add(1),
            bytes: lines.prefix(u64::MAX).to_vec(),
            name,
            folded,
        });
    }
    carries.sort_by_key(|carry| (carry.set.n, carry.copy));
    let forked = Forked {
        carried: carries.len(),
    };
    for (n, carry) in (until.unwrap_or_default()..).zip(carries) {
        let Some(copy) = copies.get_mut(carry.copy) else {
            continue;
        };
        let stamp = Stamp {
            n,
            seq: copy.next_seq,
            t: carry.set.t,
        };
        let recorded = ChangeRecord {
            change: Change::Aside {
                id: carry.id,
                aside: carry.aside,
            },
            by: carry.by,
            at: Some(stamp.t),
        };
        let refuse = |why: String| LoadError::Record {
            log: copy.name.clone(),
            line: usize::try_from(stamp.seq)
                .unwrap_or(usize::MAX)
                .saturating_add(1),
            why,
        };
        let conversation = &mut copy.folded.conversation;
        conversation
            .apply(&recorded)
            .map_err(|error| refuse(error.to_string()))?;
        write_change(
            &mut copy.bytes,
            stamp,
            recorded.by.as_ref(),
            &recorded.change,
        )
        .map_err(|failure| refuse(failure.to_string()))?;
        copy.next_seq = copy.next_seq.saturating_add(1);
    }

    // Nothing was written so far.
    let mut stored: BTreeSet<String> = from.blobs()?.into_iter().collect();
    stored.extend(blobs.blobs()?);
    for name in stored {
        let bytes = blobs.blob(&name).or_else(|_| from.blob(&name))?;
        to.put_blob(&name, &bytes)?;
    }
    for copy in &copies {
        to.append_log(&copy.name, &copy.bytes)?;
    }
    copy_plugin_files(from, to, with)?;
    Ok(forked)
}

/// Copies every plugin file of `from` (`plugins/` of its store) into `to`,
/// each mapped through `with`.
pub(crate) fn copy_plugin_files(
    from: &dyn Store,
    to: &dyn Store,
    with: &dyn Rewrite,
) -> io::Result<()> {
    for name in from.files()? {
        if name.starts_with("plugins/") {
            let mut bytes = from.read_file(&name)?;
            with.file(&name, &mut bytes);
            to.write_file(&name, &bytes)?;
        }
    }
    Ok(())
}

/// No images: a fork copies the blobs, and its folds never read them.
fn no_images(_: &str) -> io::Result<Vec<u8>> {
    Err(io::ErrorKind::NotFound.into())
}

/// The log `name` of `from` and its strict fold as it is now; `None` when
/// it holds no complete line.
fn read_now(from: &dyn Store, name: &str) -> Result<Option<(LogLines, Folded)>, LoadError> {
    let Some(lines) = log_lines(from, name)? else {
        return Ok(None);
    };
    let now = fold_lines(name, &lines, Fold::default(), &no_images)?;
    Ok(now.map(|now| (lines, now)))
}

/// The asides of the log `log` to carry into its copy `copy`, which ends
/// before the session's record `until`: each saved aside that differs
/// between the log `now` and the copy, follows an entry made before the
/// cut or none, and is not rig's own, as it is now through `with`.
fn carried(
    copy: usize,
    log: &str,
    now: &Folded,
    then: &Folded,
    until: u64,
    with: &dyn Rewrite,
) -> Vec<Carry> {
    let copied = |aside: &Aside| {
        let follows = aside
            .after
            .is_none_or(|after| now.made.get(&after).is_some_and(|made| made.n < until));
        follows && aside.author.kind != AuthorKind::Rig
    };
    let ids: BTreeSet<AsideId> = now
        .conversation
        .asides()
        .chain(then.conversation.asides())
        .map(|(id, _)| id)
        .collect();
    let mut carries = Vec::new();
    for id in ids {
        let Some((set, by)) = now.asides_set.get(&id) else {
            continue;
        };
        let was = then.conversation.aside(id);
        let aside = match now.conversation.aside(id) {
            Some(aside) if copied(aside) => {
                let mut aside = aside.clone();
                let at = Place::Record {
                    log,
                    stamp: *set,
                    by: by.as_ref(),
                };
                map_aside(with, &at, &mut aside);
                if Some(&aside) == was {
                    continue;
                }
                Some(aside)
            }
            None if was.is_some_and(copied) => None,
            _ => continue,
        };
        carries.push(Carry {
            copy,
            set: *set,
            by: by.clone(),
            id,
            aside,
        });
    }
    carries
}

#[cfg(test)]
mod tests;
