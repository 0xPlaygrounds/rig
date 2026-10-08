//! `rig`: the launcher of the rig coding agent. It generates a small Cargo
//! project for the agent from `plugins.toml`, builds it, runs it, restarts
//! it when it exits with the reload code, and rolls back to the last build
//! that started when a new build crashes during startup.
//!
//! `rig [-j N]` runs the agent. `rig build` regenerates and builds the agent
//! project for `/reload`, with cargo's JSON messages on stdout and its
//! progress bar on stderr. `-j N` sets `CARGO_BUILD_JOBS` for every build.
//!
//! With `RIG_HOME` set, every file lives under it; otherwise the XDG base
//! directories are used. `RIG_CODE_SOURCE=<rig repository>` builds the agent
//! from that checkout instead of crates.io, which is also the default when
//! the launcher itself was installed from a checkout.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread::sleep;
use std::time::Duration;

/// The launcher's version, which the agent crate shares.
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The one Bevy version the agent and every plugin must use.
const BEVY: &str = "0.20.0-rc.2";
/// The exit code with which the agent asks to be restarted.
const RELOAD_EXIT_CODE: i32 = 75;
/// The generated package, and the name of its binary.
const PACKAGE: &str = "rig-code-app";
/// The plugin list written on first run.
const PLUGINS_TEMPLATE: &str = "\
# Plugins of the rig coding agent. /reload rebuilds the agent with them.
# Each plugin is a crate and a Bevy plugin in it. Every key besides `crate`
# and `plugin` goes into the crate's Cargo dependency: path, git, rev,
# branch, tag, version or package.
#
# [[plugin]]
# crate = \"rig-code-hello\"
# path = \"/home/me/rig-code-hello\"
# plugin = \"rig_code_hello::HelloPlugin\"
";

fn main() -> ExitCode {
    match Launcher::from_args(std::env::args_os().skip(1)) {
        Ok((launcher, Mode::Run)) => launcher.run(),
        Ok((launcher, Mode::Build)) => match launcher.build(true) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(&error),
        },
        Err(error) => fail(&error),
    }
}

fn fail(error: &Error) -> ExitCode {
    eprintln!("error: {error}");
    ExitCode::FAILURE
}

/// What went wrong.
#[derive(Debug)]
enum Error {
    /// The command line was not understood.
    Usage(String),
    /// A file or process operation failed.
    Io(String, io::Error),
    /// `plugins.toml` is malformed.
    Plugins {
        path: PathBuf,
        line: usize,
        reason: &'static str,
    },
    /// A plugin pulls in a second Bevy.
    SecondBevy {
        plugins: PathBuf,
        dependency: String,
        version: String,
    },
    /// A cargo command failed; cargo has said why.
    Cargo(&'static str),
    /// There is no built agent to run.
    NoBinary,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(reason) => write!(f, "{reason}\nusage: rig [-j N] [build]"),
            Self::Io(action, error) => write!(f, "cannot {action}: {error}"),
            Self::Plugins { path, line, reason } => {
                write!(f, "{}:{line}: {reason}", path.display())
            }
            Self::SecondBevy {
                plugins,
                dependency,
                version,
            } => write!(
                f,
                "the {dependency} uses Bevy {version}, but rig needs exactly Bevy \
                 {BEVY}, and two Bevy versions cannot share one app. Update the plugin to \
                 depend on Bevy ={BEVY}, or remove it from {}.",
                plugins.display()
            ),
            Self::Cargo(step) => write!(f, "{step} failed; see cargo's messages above"),
            Self::NoBinary => write!(f, "there is no working build of the agent to run"),
        }
    }
}

/// Wraps an I/O error with what was being done.
fn io(action: impl FnOnce() -> String) -> impl FnOnce(io::Error) -> Error {
    move |error| Error::Io(action(), error)
}

enum Mode {
    Run,
    Build,
}

/// The launcher's settings: where its files live, the agent's source, and
/// the build job count.
struct Launcher {
    config: PathBuf,
    cache: PathBuf,
    data: PathBuf,
    /// The rig repository to build the agent from, in local-source mode.
    source: Option<PathBuf>,
    jobs: Option<String>,
}

/// A plugin of `plugins.toml`.
struct Plugin {
    /// The crate's package name.
    krate: String,
    /// The Rust expression of the Bevy plugin.
    expression: String,
    /// The crate's Cargo dependency fields, such as `path`.
    dependency: Vec<(String, String)>,
}

/// A `[[plugin]]` table while it is parsed.
#[derive(Default)]
struct Table {
    line: usize,
    krate: Option<String>,
    expression: Option<String>,
    dependency: Vec<(String, String)>,
}

impl Launcher {
    fn from_args(mut args: impl Iterator<Item = OsString>) -> Result<(Self, Mode), Error> {
        let mut mode = Mode::Run;
        let mut jobs = None;
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("-j" | "--jobs") => {
                    let count = args.next().and_then(|count| count.into_string().ok());
                    match count {
                        Some(count) if count.parse::<u16>().is_ok_and(|count| count > 0) => {
                            jobs = Some(count);
                        }
                        _ => return Err(Error::Usage("-j takes a positive number".to_owned())),
                    }
                }
                Some("build") => mode = Mode::Build,
                _ => {
                    return Err(Error::Usage(format!(
                        "unknown argument {}",
                        arg.to_string_lossy()
                    )));
                }
            }
        }
        let (config, cache, data) = directories();
        let launcher = Self {
            config,
            cache,
            data,
            source: source(),
            jobs,
        };
        Ok((launcher, mode))
    }

    fn agent_dir(&self) -> PathBuf {
        self.config.join("agent")
    }

    fn plugins_path(&self) -> PathBuf {
        self.config.join("plugins.toml")
    }

    fn bin(&self, name: &str) -> PathBuf {
        self.cache.join("bin").join(name)
    }

    /// `cargo`, with the job count when one was given.
    fn cargo(&self) -> Command {
        let mut cargo = Command::new("cargo");
        if let Some(jobs) = &self.jobs {
            cargo.env("CARGO_BUILD_JOBS", jobs);
        }
        cargo
    }

    /// Builds the agent, then runs it until it quits: a reload starts the
    /// newest build, and a new build that crashes before it is ready is
    /// replaced by the last one that was.
    fn run(&self) -> ExitCode {
        let current = self.bin("current");
        let candidate = self.bin("candidate");
        let ready = self.bin("ready");
        let mut notice = None;
        if let Err(error) = self.build(false) {
            eprintln!("error: {error}");
            if !current.exists() {
                return ExitCode::FAILURE;
            }
            notice = Some(format!(
                "The build failed ({error}). This is the last working build; fix the problem \
                 and /reload."
            ));
            let _ = fs::remove_file(&candidate);
        }
        let launcher = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("rig"));
        loop {
            let trying = candidate.exists();
            let binary = if trying { &candidate } else { &current };
            if !binary.exists() {
                return fail(&Error::NoBinary);
            }
            let _ = fs::remove_file(&ready);
            let mut agent = Command::new(binary);
            agent
                .env("RIG_LAUNCHER", &launcher)
                .env("RIG_READY_FILE", &ready)
                .env_remove("RIG_NOTICE");
            if let Some(jobs) = &self.jobs {
                agent.env("CARGO_BUILD_JOBS", jobs);
            }
            if let Some(notice) = notice.take() {
                agent.env("RIG_NOTICE", notice);
            }
            let status = match self.supervise(&mut agent, trying) {
                Ok(status) => status,
                Err(error) => return fail(&error),
            };
            match status.code() {
                Some(0) => return ExitCode::SUCCESS,
                Some(RELOAD_EXIT_CODE) => continue,
                _ if trying && candidate.exists() && current.exists() => {
                    let _ = fs::remove_file(&candidate);
                    let text = format!(
                        "The new build crashed during startup ({status}); rolled back to the \
                         last working build."
                    );
                    eprintln!("rig: {text}");
                    notice = Some(text);
                }
                code => {
                    eprintln!(
                        "rig: the agent exited ({status}); its logs are in {}",
                        self.data.join("sessions").display()
                    );
                    let code = code.and_then(|code| u8::try_from(code).ok()).unwrap_or(1);
                    return ExitCode::from(code);
                }
            }
        }
    }

    /// Runs the agent to its exit. A candidate build that becomes ready is
    /// promoted to the current build.
    fn supervise(
        &self,
        agent: &mut Command,
        trying: bool,
    ) -> Result<std::process::ExitStatus, Error> {
        let ready = self.bin("ready");
        let mut child = agent.spawn().map_err(io(|| "start the agent".to_owned()))?;
        let mut promoted = !trying;
        loop {
            let exited = child
                .try_wait()
                .map_err(io(|| "wait for the agent".to_owned()))?;
            if !promoted && ready.exists() {
                fs::rename(self.bin("candidate"), self.bin("current"))
                    .map_err(io(|| "promote the new build".to_owned()))?;
                promoted = true;
            }
            if let Some(status) = exited {
                return Ok(status);
            }
            sleep(Duration::from_millis(100));
        }
    }

    /// Regenerates the agent project, checks that it has one Bevy, builds
    /// it and stages the binary as the candidate. With `json`, cargo writes
    /// JSON messages and a progress bar for a reader that is not a terminal.
    fn build(&self, json: bool) -> Result<(), Error> {
        let plugins = self.generate()?;
        self.check_bevy(&plugins)?;
        if !json {
            eprintln!("rig: building the agent (the first build takes a few minutes)");
        }
        let agent = self.agent_dir();
        let target = self.cache.join("target");
        let mut cargo = self.cargo();
        cargo
            .arg("build")
            .arg("--manifest-path")
            .arg(agent.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(&target);
        if json {
            cargo
                .arg("--message-format=json")
                .env("CARGO_TERM_PROGRESS_WHEN", "always")
                .env("CARGO_TERM_PROGRESS_WIDTH", "200");
        }
        let status = cargo
            .status()
            .map_err(io(|| "run cargo build".to_owned()))?;
        if !status.success() {
            return Err(Error::Cargo("the agent build"));
        }
        let built = target
            .join("debug")
            .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX));
        let staged = self.bin("candidate.partial");
        let bin = self.cache.join("bin");
        fs::create_dir_all(&bin).map_err(io(|| format!("create {}", bin.display())))?;
        fs::copy(&built, &staged).map_err(io(|| format!("copy {}", built.display())))?;
        fs::rename(&staged, self.bin("candidate")).map_err(io(|| "stage the new build".to_owned()))
    }

    /// Writes the agent project from `plugins.toml`, creating the list on
    /// first run. Files are only written when their content changes, so an
    /// unchanged project stays built.
    fn generate(&self) -> Result<Vec<Plugin>, Error> {
        let path = self.plugins_path();
        if !path.exists() {
            write_if_changed(&path, PLUGINS_TEMPLATE)?;
        }
        let text = fs::read_to_string(&path).map_err(io(|| format!("read {}", path.display())))?;
        let plugins = parse_plugins(&text).map_err(|(line, reason)| Error::Plugins {
            path: path.clone(),
            line,
            reason,
        })?;
        let agent = self.agent_dir();
        write_if_changed(&agent.join("Cargo.toml"), &self.manifest(&plugins))?;
        write_if_changed(&agent.join("src").join("main.rs"), &main_rs(&plugins))?;
        // A fresh local-source project starts from the repository's lockfile,
        // so it builds with the dependency versions the repository tested.
        let lock = agent.join("Cargo.lock");
        if let Some(source) = &self.source
            && !lock.exists()
        {
            let _ = fs::copy(source.join("Cargo.lock"), &lock);
        }
        Ok(plugins)
    }

    /// The agent project's `Cargo.toml`.
    fn manifest(&self, plugins: &[Plugin]) -> String {
        let mut manifest = format!(
            "# Generated by rig {VERSION} from {}. Edits are overwritten.\n\
             [package]\nname = \"{PACKAGE}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             publish = false\n\n[dependencies]\n",
            self.plugins_path().display()
        );
        match &self.source {
            Some(source) => manifest.push_str(&format!(
                "rig-code = {{ path = {} }}\n",
                toml_string(&source.join("crates/rig-code").to_string_lossy())
            )),
            None => manifest.push_str(&format!("rig-code = \"={VERSION}\"\n")),
        }
        manifest.push_str(&format!(
            "bevy_app = {{ version = \"={BEVY}\", default-features = false }}\n\
             bevy_ecs = {{ version = \"={BEVY}\", default-features = false }}\n"
        ));
        for plugin in plugins {
            let fields: Vec<String> = plugin
                .dependency
                .iter()
                .map(|(key, value)| format!("{key} = {}", toml_string(value)))
                .collect();
            manifest.push_str(&format!(
                "{} = {{ {} }}\n",
                toml_string(&plugin.krate),
                fields.join(", ")
            ));
        }
        if let Some(source) = &self.source {
            // Plugins that ask for the published crates get the local ones.
            manifest.push_str("\n[patch.crates-io]\n");
            for krate in ["rig-code", "rig-core"] {
                manifest.push_str(&format!(
                    "{krate} = {{ path = {} }}\n",
                    toml_string(&source.join("crates").join(krate).to_string_lossy())
                ));
            }
        }
        manifest.push_str("\n[workspace]\n");
        manifest
    }

    /// Fails with a plain explanation when a plugin pulls in a Bevy other
    /// than [`BEVY`], naming the dependency of the agent project it came
    /// through.
    fn check_bevy(&self, plugins: &[Plugin]) -> Result<(), Error> {
        let output = self
            .cargo()
            .args([
                "tree", "-e", "normal", "--prefix", "depth", "--format", "{p}",
            ])
            .arg("--manifest-path")
            .arg(self.agent_dir().join("Cargo.toml"))
            .stderr(Stdio::inherit())
            .output()
            .map_err(io(|| "run cargo tree".to_owned()))?;
        if !output.status.success() {
            return Err(Error::Cargo("resolving the agent's dependencies"));
        }
        let tree = String::from_utf8_lossy(&output.stdout);
        let mut top = "";
        for line in tree.lines() {
            let package = line.trim_start_matches(|character: char| character.is_ascii_digit());
            let depth = line.get(..line.len() - package.len()).unwrap_or_default();
            let mut words = package.split_whitespace();
            let (Some(name), Some(version)) = (words.next(), words.next()) else {
                continue;
            };
            if depth == "1" {
                top = name;
            }
            let version = version.trim_start_matches('v');
            if name == "bevy_ecs" && version != BEVY {
                let plugin = plugins.iter().any(|plugin| plugin.krate == top);
                return Err(Error::SecondBevy {
                    plugins: self.plugins_path(),
                    dependency: if plugin {
                        format!("plugin crate \"{top}\"")
                    } else {
                        format!("crate \"{top}\"")
                    },
                    version: version.to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// The config, cache and data directories: under `RIG_HOME` when it is
/// set, else the XDG base directories, each with a `rig` subdirectory.
fn directories() -> (PathBuf, PathBuf, PathBuf) {
    if let Some(home) = std::env::var_os("RIG_HOME").filter(|home| !home.is_empty()) {
        let home = PathBuf::from(home);
        return (home.join("config"), home.join("cache"), home.join("data"));
    }
    let user = std::env::home_dir().unwrap_or_default();
    let base = |variable: &str, default: &str| {
        std::env::var_os(variable)
            .filter(|value| !value.is_empty())
            .map_or_else(|| user.join(default), PathBuf::from)
            .join("rig")
    };
    (
        base("XDG_CONFIG_HOME", ".config"),
        base("XDG_CACHE_HOME", ".cache"),
        base("XDG_DATA_HOME", ".local/share"),
    )
}

/// The rig repository to build the agent from: `RIG_CODE_SOURCE`, or the
/// checkout this launcher was built from when it has the agent crate. A
/// launcher installed from crates.io has no such checkout.
fn source() -> Option<PathBuf> {
    if let Some(source) = std::env::var_os("RIG_CODE_SOURCE").filter(|source| !source.is_empty()) {
        return Some(PathBuf::from(source));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    root.join("crates/rig-code/Cargo.toml")
        .is_file()
        .then(|| root.to_path_buf())
}

/// The agent project's `main.rs`: the agent app plus one `add_plugins`
/// call per plugin.
fn main_rs(plugins: &[Plugin]) -> String {
    let mut main = format!(
        "// Generated by rig {VERSION} from plugins.toml. Edits are overwritten.\n\
         fn main() -> rig_code::bevy_app::AppExit {{\n"
    );
    if plugins.is_empty() {
        main.push_str("    rig_code::app().run()\n}\n");
        return main;
    }
    main.push_str("    let mut app = rig_code::app();\n");
    for plugin in plugins {
        main.push_str(&format!("    app.add_plugins({});\n", plugin.expression));
    }
    main.push_str("    app.run()\n}\n");
    main
}

/// Parses `plugins.toml`, a strict subset of TOML: `[[plugin]]` headers,
/// `key = "string"` lines and `#` comments. Each plugin is its crate name,
/// its plugin expression and its dependency fields. An error names the
/// line and what is wrong with it.
fn parse_plugins(text: &str) -> Result<Vec<Plugin>, (usize, &'static str)> {
    let mut tables: Vec<Table> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("[[plugin]]") {
            if !is_comment(rest) {
                return Err((number, "unexpected text after [[plugin]]"));
            }
            tables.push(Table {
                line: number,
                ..Table::default()
            });
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err((number, "expected [[plugin]] or key = \"value\""));
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "-_".contains(character))
        {
            return Err((number, "a key is letters, digits, `-` and `_`"));
        }
        let (value, rest) = parse_string(value.trim()).ok_or((
            number,
            "a value is a double-quoted string; only \\\" and \\\\ escapes are supported",
        ))?;
        if !is_comment(rest) {
            return Err((number, "unexpected text after the value"));
        }
        let table = tables
            .last_mut()
            .ok_or((number, "a key must follow a [[plugin]] header"))?;
        let duplicate = match key {
            "crate" => table.krate.replace(value).is_some(),
            "plugin" => table.expression.replace(value).is_some(),
            _ => {
                let duplicate = table.dependency.iter().any(|(name, _)| name == key);
                table.dependency.push((key.to_owned(), value));
                duplicate
            }
        };
        if duplicate {
            return Err((number, "this key is already set for this plugin"));
        }
    }
    tables
        .into_iter()
        .map(|table| match (table.krate, table.expression) {
            (Some(krate), Some(expression)) if !table.dependency.is_empty() => Ok(Plugin {
                krate,
                expression,
                dependency: table.dependency,
            }),
            _ => Err((
                table.line,
                "a plugin needs `crate`, `plugin` and a source: path, git or version",
            )),
        })
        .collect()
}

/// Whether `rest` is empty or a comment.
fn is_comment(rest: &str) -> bool {
    let rest = rest.trim();
    rest.is_empty() || rest.starts_with('#')
}

/// Reads a double-quoted string at the start of `text`, returning it and
/// what follows.
fn parse_string(text: &str) -> Option<(String, &str)> {
    let mut chars = text.strip_prefix('"')?.char_indices();
    let mut value = String::new();
    while let Some((index, character)) = chars.next() {
        match character {
            '"' => return Some((value, text.get(index + 2..)?)),
            '\\' => match chars.next()?.1 {
                escaped @ ('"' | '\\') => value.push(escaped),
                _ => return None,
            },
            _ => value.push(character),
        }
    }
    None
}

/// `value` as a TOML basic string.
fn toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Writes `content` to `path`, creating its directory, unless the file
/// already holds exactly that.
fn write_if_changed(path: &Path, content: &str) -> Result<(), Error> {
    if fs::read_to_string(path).is_ok_and(|current| current == content) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io(|| format!("create {}", parent.display())))?;
    }
    fs::write(path, content).map_err(io(|| format!("write {}", path.display())))
}
