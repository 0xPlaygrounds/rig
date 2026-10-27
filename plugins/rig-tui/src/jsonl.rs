//! Files of one JSON value per line, such as the prompt history, which
//! only grows, and the notices a reload keeps for the next build: values
//! are appended or written whole, and `tail` reads the newest back. A
//! file is one of the view's in a
//! [`PluginData`](rig_harness::prelude::PluginData) place.

use std::fmt;
use std::io;
use std::sync::Arc;

use bevy_log::warn;
use rig_harness::prelude::Files;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// A file of values: its name in a plugin's files.
#[derive(Clone)]
pub(crate) struct Kept(pub(crate) Arc<dyn Files>, pub(crate) &'static str);

impl fmt::Display for Kept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.1)
    }
}

/// `values`, one per line.
fn lines<'a, T: Serialize + 'a>(values: impl IntoIterator<Item = &'a T>) -> String {
    values
        .into_iter()
        .filter_map(|value| serde_json::to_string(value).ok())
        .map(|line| line + "\n")
        .collect()
}

/// Appends `values` to `file`, one per line.
pub(crate) fn append<'a, T: Serialize + 'a>(
    file: &Kept,
    values: impl IntoIterator<Item = &'a T>,
) -> io::Result<()> {
    let lines = lines(values);
    if lines.is_empty() {
        return Ok(());
    }
    file.0.append_file(file.1, lines.as_bytes())
}

/// Replaces `file` with `values`, one per line.
pub(crate) fn write<'a, T: Serialize + 'a>(
    file: &Kept,
    values: impl IntoIterator<Item = &'a T>,
) -> io::Result<()> {
    file.0.write_file(file.1, lines(values).as_bytes())
}

/// The last `kept` values of `file`, oldest first, skipping lines that do
/// not read; the file is rewritten with them when it grew past twice that.
pub(crate) fn tail<T: Serialize + DeserializeOwned>(file: &Kept, kept: usize) -> Vec<T> {
    let Ok(bytes) = file.0.read_file(file.1) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let values: Vec<T> = lines
        .into_iter()
        .skip(total.saturating_sub(kept))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if total > 2 * kept
        && let Err(failure) = file.0.write_file(file.1, self::lines(&values).as_bytes())
    {
        warn!("could not trim {file}: {failure}");
    }
    values
}
