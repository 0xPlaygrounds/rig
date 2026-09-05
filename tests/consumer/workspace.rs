//! A disposable project and its explicit mutation ledger.

use super::Error;

pub(super) const INITIAL: &str = "Helo, Rig!\n";
pub(super) const TARGET: &str = "Hello, Rig!\n";

pub(super) struct Workspace {
    directory: assert_fs::TempDir,
    pub writes: usize,
}

impl Workspace {
    pub fn new() -> Result<Self, Error> {
        let directory = assert_fs::TempDir::new()
            .map_err(|e| Error::Invariant(format!("create disposable project: {e}")))?;
        std::fs::write(directory.path().join("greeting.txt"), INITIAL)?;
        Ok(Self {
            directory,
            writes: 0,
        })
    }

    pub fn from_files(
        files: &std::collections::BTreeMap<String, String>,
        writes: usize,
    ) -> Result<Self, Error> {
        for path in files.keys() {
            if path.is_empty()
                || !std::path::Path::new(path)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
            {
                return Err(Error::Invariant(
                    "project image contains an unsafe path".into(),
                ));
            }
        }
        let directory = assert_fs::TempDir::new()
            .map_err(|error| Error::Invariant(format!("create disposable project: {error}")))?;
        for (path, contents) in files {
            let path = directory.path().join(path);
            let parent = path
                .parent()
                .ok_or_else(|| Error::Invariant("project file has no parent".into()))?;
            std::fs::create_dir_all(parent)?;
            std::fs::write(path, contents)?;
        }
        Ok(Self { directory, writes })
    }

    pub fn root(&self) -> &std::path::Path {
        self.directory.path()
    }

    pub fn read(&self) -> Result<String, Error> {
        Ok(std::fs::read_to_string(
            self.directory.path().join("greeting.txt"),
        )?)
    }

    /// Restore data into a fresh disposable workspace. Restoring its image is
    /// not another invocation of the mutation tool; the operation ledger persists.
    pub fn restore(content: &str, writes: usize) -> Result<Self, Error> {
        let mut workspace = Self::new()?;
        std::fs::write(workspace.directory.path().join("greeting.txt"), content)?;
        workspace.writes = writes;
        Ok(workspace)
    }

    pub fn apply(&mut self, content: &str) -> Result<(), Error> {
        if self.writes != 0 {
            return Err(Error::Invariant(
                "duplicate write operation greeting-fix-v1".into(),
            ));
        }
        std::fs::write(self.directory.path().join("greeting.txt"), content)?;
        self.writes += 1;
        Ok(())
    }
}
