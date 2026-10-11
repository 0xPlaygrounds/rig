//! Files of one JSON value per line that only grow, such as the prompt
//! history and the notices kept for the next build: values are appended,
//! and the newest are read back.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use bevy_log::warn;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Appends `values` to `file`, one per line, making its directory.
pub(crate) fn append<'a, T: Serialize + 'a>(
    file: &Path,
    values: impl IntoIterator<Item = &'a T>,
) -> std::io::Result<()> {
    let lines: String = values
        .into_iter()
        .filter_map(|value| serde_json::to_string(value).ok())
        .map(|line| line + "\n")
        .collect();
    if lines.is_empty() {
        return Ok(());
    }
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?;
    out.write_all(lines.as_bytes())
}

/// The last `kept` values of `file`, oldest first, skipping lines that do
/// not read; the file is rewritten with them when it grew past twice that.
pub(crate) fn tail<T: Serialize + DeserializeOwned>(file: &Path, kept: usize) -> Vec<T> {
    let Ok(text) = fs::read_to_string(file) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let values: Vec<T> = lines
        .into_iter()
        .skip(total.saturating_sub(kept))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if total > 2 * kept {
        let temporary = file.with_extension("jsonl.new");
        let rewritten = fs::remove_file(&temporary)
            .or(Ok(()))
            .and_then(|()| append(&temporary, &values))
            .and_then(|()| fs::rename(&temporary, file));
        if let Err(failure) = rewritten {
            warn!("could not trim {}: {failure}", file.display());
        }
    }
    values
}
