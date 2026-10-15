//! Output a tool cut, kept whole in a file the model can `read` or
//! `search`, so cut-off output is never gone for good.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::PathBuf;

/// The directory where tools keep output they cut, each in a file of its
/// own whose path, the handle, the cut output names.
#[derive(Clone, Debug)]
pub struct Spill(pub PathBuf);

impl Spill {
    /// A new empty file for `tool`'s output, `<tool>-<n>.txt` with the
    /// first free `n`, and its path.
    pub fn create(&self, tool: &str) -> io::Result<(PathBuf, File)> {
        fs::create_dir_all(&self.0)?;
        let mut n = 1_u64;
        loop {
            let path = self.0.join(format!("{tool}-{n}.txt"));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => n += 1,
                Err(error) => return Err(error),
            }
        }
    }
}
