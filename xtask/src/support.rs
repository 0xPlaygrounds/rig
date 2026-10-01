//! Filesystem and process plumbing shared by every xtask command.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Every file under `dir`, recursively and in path order, keeping only files
/// with `extension` when one is given. An unreadable directory or entry is an
/// error, so a check never passes over files it could not see.
pub(crate) fn files_under(dir: &Path, extension: Option<&str>) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        for entry in entries {
            let path = entry
                .map_err(|error| format!("{}: {error}", dir.display()))?
                .path();
            if path.is_dir() {
                pending.push(path);
            } else if extension
                .is_none_or(|wanted| path.extension().is_some_and(|ext| ext == wanted))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// The stdout of `program args` run in `dir`. A spawn failure, a non-zero
/// exit (with its stderr) or non-UTF-8 output is an error.
pub(crate) fn output(dir: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let command = format!("{program} {}", args.join(" "));
    let result = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|error| format!("could not run {command}: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "{command} failed:\n{}",
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    String::from_utf8(result.stdout)
        .map_err(|error| format!("{command} produced non-UTF-8 output: {error}"))
}
