//! Finding and replacing secrets in JSON values, in a session's saved
//! bytes and in plain files, with no app around.

use std::fs;
use std::io;
use std::path::Path;

use rig_harness::prelude::{Session, Store};
use rig_harness::rig_memory::MemoryStore;
use serde_json::Value;

/// A secret and the placeholder it becomes.
pub(crate) type Pair = (String, String);

/// How `secret` may appear in saved bytes: as it is, and escaped once and
/// twice as inside JSON strings.
pub(crate) fn forms(secret: &str) -> Vec<String> {
    let mut forms = vec![secret.to_owned()];
    let mut last = secret.to_owned();
    for _ in 0..2 {
        let quoted = serde_json::to_string(&last).unwrap_or_default();
        let inner = quoted
            .get(1..quoted.len().saturating_sub(1))
            .unwrap_or_default();
        if inner == last {
            break;
        }
        inner.clone_into(&mut last);
        forms.push(last.clone());
    }
    forms
}

/// How often `needle` appears in `haystack`, not overlapping.
pub(crate) fn count(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let (mut at, mut found) = (0, 0);
    while let Some(rest) = haystack.get(at..) {
        match rest
            .windows(needle.len())
            .position(|window| window == needle)
        {
            Some(start) => {
                found += 1;
                at += start + needle.len();
            }
            None => break,
        }
    }
    found
}

/// `haystack` with every `needle` replaced by `with`.
fn replace(haystack: &[u8], needle: &[u8], with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(haystack.len());
    let mut at = 0;
    while let Some(rest) = haystack.get(at..) {
        match (!needle.is_empty())
            .then(|| {
                rest.windows(needle.len())
                    .position(|window| window == needle)
            })
            .flatten()
        {
            Some(start) => {
                out.extend_from_slice(rest.get(..start).unwrap_or_default());
                out.extend_from_slice(with);
                at += start + needle.len();
            }
            None => {
                out.extend_from_slice(rest);
                break;
            }
        }
    }
    out
}

/// Replaces every secret in the string values of `value`; whether any
/// was.
pub(crate) fn scrub_value(value: &mut Value, pairs: &[Pair]) -> bool {
    match value {
        Value::String(text) => {
            let mut changed = false;
            for (secret, placeholder) in pairs {
                if !secret.is_empty() && text.contains(secret.as_str()) {
                    *text = text.replace(secret.as_str(), placeholder);
                    changed = true;
                }
            }
            changed
        }
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| scrub_value(item, pairs) | changed),
        Value::Object(map) => map
            .values_mut()
            .fold(false, |changed, item| scrub_value(item, pairs) | changed),
        _ => false,
    }
}

/// Every string value of `value`.
pub(crate) fn strings(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => out.push(text.clone()),
        Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
        Value::Object(map) => map.values().for_each(|item| strings(item, out)),
        _ => {}
    }
}

/// `bytes` with every secret replaced, line by line: first in the string
/// values of a line that is JSON, then wherever one of its [`forms`] is
/// still left (a key, a line that is no JSON). `None` when none appears.
pub(crate) fn scrub_bytes(bytes: &[u8], pairs: &[Pair]) -> Option<Vec<u8>> {
    let forms: Vec<(Vec<String>, &str)> = pairs
        .iter()
        .map(|(secret, placeholder)| (forms(secret), placeholder.as_str()))
        .collect();
    let mut out = Vec::with_capacity(bytes.len());
    let mut changed = false;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        let hit = forms
            .iter()
            .any(|(forms, _)| forms.iter().any(|form| count(line, form.as_bytes()) > 0));
        if !hit {
            out.extend_from_slice(line);
            continue;
        }
        changed = true;
        let (body, newline) = match line.strip_suffix(b"\n") {
            Some(body) => (body, true),
            None => (line, false),
        };
        let json = serde_json::from_slice::<Value>(body)
            .ok()
            .and_then(|mut value| {
                scrub_value(&mut value, pairs);
                serde_json::to_vec(&value).ok()
            });
        let mut body = json.unwrap_or_else(|| body.to_vec());
        for (forms, placeholder) in &forms {
            for form in forms {
                body = replace(&body, form.as_bytes(), placeholder.as_bytes());
            }
        }
        out.extend(body);
        if newline {
            out.push(b'\n');
        }
    }
    changed.then_some(out)
}

/// What scrubbing a store rewrote.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Scrubbed {
    /// Logs rewritten.
    pub logs: usize,
    /// Named files rewritten.
    pub files: usize,
}

/// Rewrites every log and named file of `store` that holds a secret, the
/// secret replaced. The logs keep every record, its stamp and the
/// header's lineage, so a writer appending to them goes on. The rewritten
/// logs are read as a session in memory before any is replaced, so a
/// redaction never leaves a session that does not load.
pub(crate) fn scrub_store(store: &dyn Store, pairs: &[Pair]) -> Result<Scrubbed, String> {
    let failed = |failure: io::Error| failure.to_string();
    let names = store.logs().map_err(failed)?;
    let check = MemoryStore::default();
    let mut rewritten = Vec::new();
    for name in &names {
        let bytes = store.read_log(name).map_err(failed)?;
        let bytes = match scrub_bytes(&bytes, pairs) {
            Some(scrubbed) => {
                rewritten.push((name.clone(), scrubbed.clone()));
                scrubbed
            }
            None => bytes,
        };
        check.append_log(name, &bytes).map_err(failed)?;
    }
    Session::read(&check).map_err(|why| format!("the redacted session would not load: {why}"))?;
    for (name, bytes) in &rewritten {
        store.replace_log(name, bytes).map_err(failed)?;
    }
    let mut files = 0;
    for name in store.files().map_err(failed)? {
        let bytes = store.read_file(&name).map_err(failed)?;
        if let Some(scrubbed) = scrub_bytes(&bytes, pairs) {
            store.write_file(&name, &scrubbed).map_err(failed)?;
            files += 1;
        }
    }
    Ok(Scrubbed {
        logs: rewritten.len(),
        files,
    })
}

/// Every saved thing of `store` by kind and name, with its bytes.
fn saved(store: &dyn Store) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut all = Vec::new();
    for name in store.logs()? {
        all.push((format!("logs/{name}"), store.read_log(&name)?));
    }
    for name in store.files()? {
        all.push((format!("files/{name}"), store.read_file(&name)?));
    }
    for name in store.blobs()? {
        all.push((format!("blobs/{name}"), store.blob(&name)?));
    }
    Ok(all)
}

/// How often any of `secrets` appears in `bytes`, in any of its forms.
pub(crate) fn occurrences(bytes: &[u8], secrets: &[String]) -> usize {
    secrets
        .iter()
        .flat_map(|secret| forms(secret))
        .map(|form| count(bytes, form.as_bytes()))
        .sum()
}

/// The saved things of `store` that hold any of `secrets`, with how
/// often.
pub(crate) fn found_in(store: &dyn Store, secrets: &[String]) -> io::Result<Vec<(String, usize)>> {
    Ok(saved(store)?
        .into_iter()
        .map(|(name, bytes)| (name, occurrences(&bytes, secrets)))
        .filter(|(_, found)| *found > 0)
        .collect())
}

/// The text of every saved log line and named file of `store`: the
/// string values of a line that is JSON, else the line.
pub(crate) fn texts(store: &dyn Store) -> Vec<String> {
    let mut texts = Vec::new();
    let logs = store.logs().unwrap_or_default().into_iter();
    let logs = logs.filter_map(|name| store.read_log(&name).ok());
    let files = store.files().unwrap_or_default().into_iter();
    let files = files.filter_map(|name| store.read_file(&name).ok());
    for bytes in logs.chain(files) {
        for line in bytes.split(|byte| *byte == b'\n') {
            match serde_json::from_slice::<Value>(line) {
                Ok(value) => strings(&value, &mut texts),
                Err(_) => texts.push(String::from_utf8_lossy(line).into_owned()),
            }
        }
    }
    texts
}

/// Every file under `dir`, all the way down.
fn files_under(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_owned()];
    while let Some(dir) = todo.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found
}

/// Rewrites every file under `dir` that holds a secret; how many.
pub(crate) fn scrub_dir(dir: &Path, pairs: &[Pair]) -> usize {
    files_under(dir)
        .into_iter()
        .filter(|path| {
            let scrubbed = fs::read(path)
                .ok()
                .and_then(|bytes| scrub_bytes(&bytes, pairs));
            scrubbed.is_some_and(|bytes| fs::write(path, bytes).is_ok())
        })
        .count()
}

/// The files under `dir` that hold any of `secrets`, with how often.
pub(crate) fn found_under(dir: &Path, secrets: &[String]) -> Vec<(String, usize)> {
    files_under(dir)
        .into_iter()
        .filter_map(|path| {
            let found = occurrences(&fs::read(&path).ok()?, secrets);
            (found > 0).then(|| (path.display().to_string(), found))
        })
        .collect()
}
