//! The coding tools' remote handlers: the same names, arguments and result
//! formats as rig-tools' `read`, `write`, `edit`, `search` and `shell`,
//! acting on the remote workspace over the [`Link`].

use std::fmt::Write as _;

use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::tool::{
    ExitStatus, FileRead, FilesChanged, FilesRead, Scope, Tool, ToolContext, ToolErrorKind,
    ToolExecutionError, Touches, content_hash,
};
use rig_tools::edit::apply;
use rig_tools::fs::file_change;
use rig_tools::process::Launcher;
use rig_tools::{
    Edit, EditArgs, Found, Hit, MAX_BYTES, MAX_FILE_BYTES, MAX_LINES, MAX_MATCHES, Read, ReadArgs,
    Search, SearchArgs, Shell, ShellArgs, Spill, Write, WriteArgs, blocking, clip, render,
};

use crate::link::{Link, Ran, lost};
use crate::{Target, quote, quote_path};

/// How long a searched line is shown, as rig-tools' `search` cuts it.
const MAX_HIT_CHARS: usize = 300;

/// What every remote tool works with: the link, the target its call was
/// admitted for, and how processes start.
#[derive(Clone)]
pub struct Remote {
    /// The link.
    pub link: Link,
    /// The workspace the call's paths resolve in.
    pub target: Target,
    /// How the local `ssh` starts.
    pub launcher: Launcher,
}

impl Remote {
    /// `path` resolved against the remote workspace, refused outside it.
    fn place(&self, path: &str) -> Result<String, ToolExecutionError> {
        let resolved = self.target.join(path);
        if !self.target.contains(&resolved) {
            return Err(ToolExecutionError::refused(format!(
                "{path} is outside {}, the only directory this tool works in",
                self.target.dir
            )));
        }
        Ok(resolved)
    }

    /// How a file of the remote workspace is named in published facts, so
    /// no local plugin takes it for a local file.
    fn fact(&self, path: &str) -> String {
        format!("ssh://{}{path}", self.target.host)
    }

    /// Runs `script` remotely on a thread of its own.
    async fn run(&self, script: String, stdin: Option<String>) -> Result<Ran, ToolExecutionError> {
        let remote = self.clone();
        blocking(move || {
            let stdin = stdin.as_deref().map(str::as_bytes);
            remote.link.exec(&remote.launcher, &script, stdin)
        })
        .await
    }

    /// The text of the remote file at `path`, refused when it is no
    /// regular file.
    async fn read_text(&self, path: &str) -> Result<String, ToolExecutionError> {
        let script = format!(
            "p={}; if [ ! -f \"$p\" ]; then echo \"$p is no regular file\" >&2; exit 2; fi; \
             head -c {} -- \"$p\"",
            quote_path(path),
            MAX_FILE_BYTES
        );
        let ran = self.run(script, None).await?;
        match ran.code {
            Some(0) => Ok(ran.out),
            Some(2) => Err(ToolExecutionError::new(
                ToolErrorKind::NotFound,
                ran.err.trim(),
            )),
            _ => Err(ToolExecutionError::other(ran.err.trim())),
        }
    }

    /// The remote file's text before a write: `None` when there is none.
    async fn previous(&self, path: &str) -> Result<Option<String>, ToolExecutionError> {
        let script = format!("cat -- {} 2>/dev/null", quote_path(path));
        let ran = self.run(script, None).await?;
        Ok((ran.code == Some(0)).then_some(ran.out))
    }

    /// Replaces the remote file at `path` with `text`, through a temporary
    /// file renamed over it, creating its directory.
    async fn put(&self, path: &str, text: String) -> Result<(), ToolExecutionError> {
        let script = format!(
            "p={}; d=$(dirname -- \"$p\") && mkdir -p -- \"$d\" && t=\"$d/.rig-remote.$$.tmp\" \
             && cat > \"$t\" && mv -f -- \"$t\" \"$p\"",
            quote_path(path)
        );
        let ran = self.run(script, Some(text)).await?;
        match ran.code {
            Some(0) => Ok(()),
            _ => Err(ToolExecutionError::other(format!(
                "{path}: {}",
                ran.err.trim()
            ))),
        }
    }
}

/// `read`, of a remote file.
pub struct RemoteRead(pub Remote);

impl Tool for RemoteRead {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Read::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        Read::default().parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        Read::default().annotations()
    }

    fn footprint(&self, args: &ReadArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![self.0.target.join(&args.path).into()]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReadArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self.0.place(&args.path)?;
        let text = self.0.read_text(&path).await?;
        let read = FileRead::new(self.0.fact(&path), content_hash(text.as_bytes()));
        context
            .insert_result(FilesRead(vec![read]))
            .map_err(ToolExecutionError::from_error)?;
        Ok(args.show(&text))
    }
}

/// `write`, of a remote file.
pub struct RemoteWrite(pub Remote);

impl Tool for RemoteWrite {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Write::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        Write::default().parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        Write::default().annotations()
    }

    fn footprint(&self, args: &WriteArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.0.target.join(&args.path).into()]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self.0.place(&args.path)?;
        let before = self.0.previous(&path).await?;
        self.0.put(&path, args.content.clone()).await?;
        let change = file_change(&self.0.fact(&path), before.as_deref(), &args.content)
            .with_after(content_hash(args.content.as_bytes()));
        context
            .insert_result(FilesChanged(vec![change]))
            .map_err(ToolExecutionError::from_error)?;
        Ok(format!(
            "Wrote {} bytes to {}.",
            args.content.len(),
            args.path
        ))
    }
}

/// `edit`, of a remote file.
pub struct RemoteEdit(pub Remote);

impl Tool for RemoteEdit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Edit::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        Edit::default().parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        Edit::default().annotations()
    }

    fn footprint(&self, args: &EditArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.0.target.join(&args.path).into()]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: EditArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self.0.place(&args.path)?;
        let raw = self.0.read_text(&path).await?;
        let applied = apply(&self.0.fact(&path), &raw, &args.edits)?;
        self.0.put(&path, applied.text).await?;
        let replaced = applied.replaced;
        let mut out = format!(
            "Edited {}: {replaced} replacement{}.",
            args.path,
            if replaced == 1 { "" } else { "s" }
        );
        for note in applied.notes {
            out.push_str(&format!("\nNote: {note}"));
        }
        out.push('\n');
        out.push_str(&shown(applied.change.diff.as_deref().unwrap_or_default()));
        context
            .insert_result(FilesChanged(vec![applied.change]))
            .map_err(ToolExecutionError::from_error)?;
        Ok(out)
    }
}

/// A diff as rig-tools' `edit` shows it (its private `shown`, copied).
fn shown(diff: &str) -> String {
    let lines: Vec<&str> = diff.lines().collect();
    let mut out = String::new();
    for (count, line) in lines.iter().enumerate() {
        if count >= MAX_LINES / 4 || out.len() + line.len() > MAX_BYTES / 2 {
            let rest = lines.len() - count;
            let _ = writeln!(out, "[{rest} more diff lines; read the file to see them]");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `search`, with the remote `grep`.
pub struct RemoteSearch(pub Remote);

impl Tool for RemoteSearch {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Search::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        Search::default().parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        Search::default().annotations()
    }

    fn footprint(&self, args: &SearchArgs) -> Touches {
        let path = args.path.as_deref().unwrap_or(".");
        Touches::Reads(Scope::Paths(vec![self.0.target.join(path).into()]))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: SearchArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self.0.place(args.path.as_deref().unwrap_or("."))?;
        let glob = args
            .glob
            .as_deref()
            .map(|glob| format!(" --include={}", quote(glob)))
            .unwrap_or_default();
        let script = format!(
            "grep -rnIE --exclude='.*' --exclude-dir='.*' --exclude-dir=target{glob} -e {} -- {} \
             | head -n {}",
            quote(&args.pattern),
            quote_path(&path),
            MAX_MATCHES + 1
        );
        let ran = self.0.run(script, None).await?;
        let mut hits: Vec<Hit> = ran.out.lines().filter_map(hit).collect();
        let stopped = hits.len() > MAX_MATCHES;
        hits.truncate(MAX_MATCHES);
        if hits.is_empty() && !ran.err.trim().is_empty() {
            return Err(ToolExecutionError::invalid_args(ran.err.trim()));
        }
        Ok(render(&Found {
            hits,
            stopped,
            files: Vec::new(),
        }))
    }
}

/// A `grep -n` line, `path:line:text`, as a hit.
fn hit(line: &str) -> Option<Hit> {
    let mut parts = line.splitn(3, ':');
    let (path, number, text) = (parts.next()?, parts.next()?, parts.next()?);
    Some(Hit {
        path: path.into(),
        line: number.parse().ok()?,
        text: clip(text, MAX_HIT_CHARS).to_owned(),
    })
}

/// `shell`, run in the remote workspace by rig-tools' own [`Shell`]: the
/// command it starts locally is `ssh`, which runs the model's command in
/// the remote user's login shell environment.
pub struct RemoteShell {
    /// The link and where it points.
    pub remote: Remote,
    /// Where cut output is kept whole.
    pub spill: Option<Spill>,
}

impl Tool for RemoteShell {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Shell::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        Shell::default().parameters()
    }

    fn annotations(&self) -> ToolAnnotations {
        Shell::default().annotations()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ShellArgs,
    ) -> Result<String, ToolExecutionError> {
        let remote = &self.remote;
        let dir = remote.place(args.cwd.as_deref().unwrap_or("."))?;
        let target = {
            let link = remote.link.clone();
            blocking(move || link.ready()).await?
        };
        let script = format!(
            "stty -onlcr 2>/dev/null; cd {} && exec sh -lc {}",
            quote_path(&dir),
            quote(&args.command)
        );
        let mut command = String::from("exec");
        for word in remote.link.argv(&target, true) {
            command.push(' ');
            command.push_str(&quote(&word));
        }
        command.push(' ');
        command.push_str(&quote(&script));
        let shell = Shell {
            launcher: remote.launcher.clone(),
            spill: self.spill.clone(),
            ..Shell::default()
        };
        let ran = shell
            .run(ShellArgs {
                command,
                timeout_secs: args.timeout_secs,
                cwd: None,
            })
            .await?;
        if ran.status == ExitStatus::Exited(255) {
            remote.link.lost(&remote.launcher, &target);
            return Err(lost(&target.host));
        }
        context
            .insert_result(ran.status)
            .map_err(ToolExecutionError::from_error)?;
        Ok(ran.output)
    }
}
