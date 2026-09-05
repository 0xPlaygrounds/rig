//! Declared project files; immutable fixture contracts stay outside patch authority.

use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use super::super::{Error, workspace::Workspace};

#[cfg(test)]
mod tests;

pub(super) type Image = BTreeMap<String, String>;

pub(super) fn initial() -> Image {
    [
        (
            "Cargo.toml",
            include_str!("../../fixtures/repository_repair/Cargo.toml"),
        ),
        (
            "Cargo.lock",
            include_str!("../../fixtures/repository_repair/Cargo.lock"),
        ),
        (
            "README.md",
            include_str!("../../fixtures/repository_repair/README.md"),
        ),
        (
            "src/lib.rs",
            include_str!("../../fixtures/repository_repair/src/lib.rs"),
        ),
        (
            "tests/pagination.rs",
            include_str!("../../fixtures/repository_repair/tests/pagination.rs"),
        ),
    ]
    .into_iter()
    .map(|(path, content)| (path.into(), content.into()))
    .collect()
}

pub(super) fn digest(image: &Image) -> Result<String, Error> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(image)?)))
}

pub(super) fn content_digest(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}

pub(super) struct Project {
    workspace: Workspace,
}

impl Project {
    pub fn new() -> Result<Self, Error> {
        Self::restore(&initial(), 0)
    }

    pub fn restore(image: &Image, writes: usize) -> Result<Self, Error> {
        let fixture = initial();
        for path in image.keys() {
            if !fixture.contains_key(path) && path != "tests/regression.rs" {
                return Err(Error::Invariant(format!("undeclared project file {path}")));
            }
        }
        for (path, contents) in &fixture {
            if path != "src/lib.rs" && image.get(path) != Some(contents) {
                return Err(Error::Invariant(format!(
                    "immutable project file changed: {path}"
                )));
            }
        }
        if !image.contains_key("src/lib.rs") {
            return Err(Error::Invariant("project source is missing".into()));
        }
        Ok(Self {
            workspace: Workspace::from_files(image, writes)?,
        })
    }

    pub fn root(&self) -> &Path {
        self.workspace.root()
    }
    pub fn writes(&self) -> usize {
        self.workspace.writes
    }

    fn path(&self, relative: &str) -> Result<PathBuf, Error> {
        if !initial().contains_key(relative) && relative != "tests/regression.rs" {
            return Err(Error::Invariant(format!(
                "undeclared project path {relative}"
            )));
        }
        let mut path = self.root().to_path_buf();
        for component in Path::new(relative).components() {
            path.push(component);
            match std::fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(Error::Invariant("project symlink refused".into()));
                }
                Ok(_) => (),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                Err(error) => return Err(error.into()),
            }
        }
        Ok(path)
    }

    pub fn read(&self, path: &str) -> Result<String, Error> {
        Ok(std::fs::read_to_string(self.path(path)?)?)
    }

    pub fn image(&self) -> Result<Image, Error> {
        fn inspect(root: &Path, relative: &str, allowed: &Image) -> Result<(), Error> {
            for entry in std::fs::read_dir(root.join(relative))? {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| Error::Invariant("non-UTF8 project path".into()))?;
                let path = if relative.is_empty() {
                    name
                } else {
                    format!("{relative}/{name}")
                };
                let kind = entry.file_type()?;
                if kind.is_dir() && matches!(path.as_str(), "src" | "tests") {
                    inspect(root, &path, allowed)?;
                } else if !kind.is_file()
                    || (!allowed.contains_key(&path) && path != "tests/regression.rs")
                {
                    return Err(Error::Invariant(format!(
                        "undeclared or non-regular project path: {path}"
                    )));
                }
            }
            Ok(())
        }
        let fixture = initial();
        inspect(self.root(), "", &fixture)?;
        let mut image = Image::new();
        for path in initial()
            .keys()
            .map(String::as_str)
            .chain(["tests/regression.rs"])
        {
            let full = self.path(path)?;
            if path == "tests/regression.rs" && !full.exists() {
                continue;
            }
            image.insert(path.into(), std::fs::read_to_string(full)?);
        }
        for (path, content) in &fixture {
            if path != "src/lib.rs" && image.get(path) != Some(content) {
                return Err(Error::Invariant(format!(
                    "immutable project file changed: {path}"
                )));
            }
        }
        Ok(image)
    }

    /// The policy layer checks operation approval; this boundary independently
    /// rejects changed input, undeclared writes and symlink replacement.
    pub fn apply(&mut self, path: &str, content: &str, before: &str) -> Result<String, Error> {
        if !matches!(path, "src/lib.rs" | "tests/regression.rs") {
            return Err(Error::Invariant(
                "patch may change only source or the new regression file".into(),
            ));
        }
        if content.len() > 16 * 1024 || !content.ends_with('\n') {
            return Err(Error::Invariant(
                "patch content must end in a newline and fit 16 KiB".into(),
            ));
        }
        let image = self.image()?;
        if digest(&image)? != before {
            return Err(Error::Invariant(
                "stale patch: project digest changed".into(),
            ));
        }
        // No validation subprocess runs concurrently with an approved edit;
        // model subprocesses cannot write this project or leave child writers.
        let target = self.path(path)?;
        let temporary = self.root().join(".repair-patch.tmp");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> std::io::Result<()> {
            file.write_all(content.as_bytes())?;
            file.sync_all()?;
            std::fs::rename(&temporary, &target)
        })();
        let _ = std::fs::remove_file(&temporary);
        result?;
        self.workspace.writes += 1;
        digest(&self.image()?)
    }
}
