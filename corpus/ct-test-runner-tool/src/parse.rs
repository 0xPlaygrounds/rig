//! Reading a test run's output line by line: [`Parser`], one per [`Kind`]
//! of runner, which knows the counts and the failures so far at any line.

use serde::Deserialize;

/// The lines of a failure's own output a result keeps.
pub const FAILURE_LINES: usize = 30;

/// Which test runner prints the output, and so how it is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// `cargo test` (libtest's output).
    Cargo,
    /// `pytest -v`.
    Pytest,
    /// `go test -v`.
    Go,
    /// `npm test`, whose output depends on the project's runner: never
    /// read, always the fallback.
    Npm,
    /// Anything else: always the fallback.
    #[default]
    Other,
}

impl Kind {
    /// The kind a configured command line runs, from its words.
    pub fn guess(command: &str) -> Self {
        let words: Vec<&str> = command.split_whitespace().collect();
        let has = |word: &str| words.contains(&word);
        if has("nextest") {
            Self::Other
        } else if has("cargo") {
            Self::Cargo
        } else if has("pytest") {
            Self::Pytest
        } else if has("go") && has("test") {
            Self::Go
        } else if ["npm", "yarn", "pnpm", "jest"].into_iter().any(has) {
            Self::Npm
        } else {
            Self::Other
        }
    }

    /// The command this kind runs when a project configures none.
    pub fn default_command(self) -> Option<&'static str> {
        match self {
            Self::Cargo => Some("cargo test"),
            Self::Pytest => Some("python -m pytest -v"),
            Self::Go => Some("go test -v ./..."),
            Self::Npm => Some("npm test"),
            Self::Other => None,
        }
    }

    /// The arguments that run only `filter`'s tests, after the command.
    pub fn filter_args(self, filter: &Filter) -> Vec<String> {
        let names = match filter {
            Filter::All => return Vec::new(),
            Filter::Matching(pattern) => {
                return match self {
                    Self::Cargo | Self::Other => vec![pattern.clone()],
                    Self::Pytest => vec!["-k".to_owned(), pattern.clone()],
                    Self::Go => vec!["-run".to_owned(), pattern.clone()],
                    Self::Npm => vec!["--".to_owned(), pattern.clone()],
                };
            }
            Filter::Exactly(names) => names,
        };
        match self {
            Self::Cargo => ["--", "--exact"]
                .into_iter()
                .map(str::to_owned)
                .chain(names.iter().cloned())
                .collect(),
            Self::Go => {
                let mut top: Vec<&str> = names
                    .iter()
                    .map(|name| name.split('/').next().unwrap_or(name))
                    .collect();
                top.dedup();
                vec!["-run".to_owned(), format!("^({})$", top.join("|"))]
            }
            Self::Pytest | Self::Npm | Self::Other => names.clone(),
        }
    }
}

/// Which tests a run runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Every test.
    All,
    /// Those the runner's own filter matches.
    Matching(String),
    /// Exactly these, by full name.
    Exactly(Vec<String>),
}

/// A failed test.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Failure {
    /// Its full name, as the runner takes it back as a filter.
    pub name: String,
    /// `file:line` of the failure, when the output says.
    pub location: Option<String>,
    /// The assertion's message, when the output says.
    pub message: Option<String>,
    /// Its first [`FAILURE_LINES`] lines of output.
    pub output: Vec<String>,
    /// Lines of output left out after those.
    pub cut: usize,
    /// Whether the line after the next is the message (libtest's
    /// `panicked at` line comes first).
    message_next: bool,
}

impl Failure {
    fn named(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            ..Self::default()
        }
    }

    fn push(&mut self, line: &str) {
        if self.output.len() < FAILURE_LINES {
            self.output.push(line.to_owned());
        } else {
            self.cut += 1;
        }
    }
}

/// Where in the output a parser is.
#[derive(Clone, Debug, Default)]
enum Section {
    /// Between failures.
    #[default]
    None,
    /// Inside the output of this failure.
    Failure(usize),
    /// Inside a Go test that has not said how it ended: its name and its
    /// lines so far.
    GoRun(String, Vec<String>),
}

/// Reads a run's output a line at a time.
#[derive(Clone, Debug, Default)]
pub struct Parser {
    kind: Kind,
    passed: usize,
    failed: usize,
    skipped: usize,
    /// The counts of the runner's own summary lines, added up, once seen.
    summary: Option<[usize; 3]>,
    failures: Vec<Failure>,
    section: Section,
}

impl Parser {
    /// A parser of `kind`'s output.
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// Passed, failed and skipped: the runner's summary once it printed
    /// one, else the tests seen so far.
    pub fn counts(&self) -> [usize; 3] {
        self.summary
            .unwrap_or([self.passed, self.failed, self.skipped])
    }

    /// Whether the output read so far looked like test results.
    pub fn parsed(&self) -> bool {
        self.summary.is_some() || self.passed + self.failed + self.skipped > 0
    }

    /// The failures so far.
    pub fn failures(&self) -> &[Failure] {
        &self.failures
    }

    /// Reads the next line, without its line break.
    pub fn feed(&mut self, line: &str) {
        match self.kind {
            Kind::Cargo => self.cargo(line),
            Kind::Pytest => self.pytest(line),
            Kind::Go => self.go(line),
            Kind::Npm | Kind::Other => {}
        }
    }

    fn fail(&mut self, name: &str) -> usize {
        self.failed += 1;
        self.failure(name)
    }

    /// The index of the failure `name`, added if new.
    fn failure(&mut self, name: &str) -> usize {
        if let Some(at) = self.failures.iter().position(|f| f.name == name) {
            return at;
        }
        self.failures.push(Failure::named(name));
        self.failures.len() - 1
    }

    fn add_summary(&mut self, [passed, failed, skipped]: [usize; 3]) {
        let [p, f, s] = self.summary.unwrap_or_default();
        self.summary = Some([p + passed, f + failed, s + skipped]);
    }

    fn cargo(&mut self, line: &str) {
        if let Some((name, status)) = line
            .strip_prefix("test ")
            .and_then(|rest| rest.rsplit_once(" ... "))
        {
            match status.trim() {
                "ok" => self.passed += 1,
                "FAILED" => {
                    self.fail(name);
                }
                status if status.starts_with("ignored") => self.skipped += 1,
                _ => {}
            }
            return;
        }
        if let Some(result) = line.strip_prefix("test result: ") {
            self.section = Section::None;
            self.add_summary(counts(result, ';', &["passed"], &["failed"], &["ignored"]));
            return;
        }
        if let Some(name) = line
            .strip_prefix("---- ")
            .and_then(|rest| rest.strip_suffix(" stdout ----"))
        {
            self.section = Section::Failure(self.failure(name));
            return;
        }
        if line == "failures:" {
            self.section = Section::None;
            return;
        }
        let Section::Failure(at) = self.section else {
            return;
        };
        let Some(failure) = self.failures.get_mut(at) else {
            return;
        };
        failure.push(line);
        if let Some((_, place)) = line.split_once("panicked at ") {
            // `src/lib.rs:12:9:`, or before Rust 1.73 `'message', src/lib.rs:12:9`.
            let (message, place) = match place.rsplit_once("', ") {
                Some((message, place)) => (message.strip_prefix('\'').map(str::to_owned), place),
                None => (None, place),
            };
            let place = place.trim_end_matches(':');
            let place = match place.rsplit_once(':') {
                Some((head, column)) if column.chars().all(|c| c.is_ascii_digit()) => head,
                _ => place,
            };
            failure.location = Some(place.to_owned());
            failure.message_next = message.is_none();
            failure.message = failure.message.take().or(message);
        } else if failure.message_next && !line.trim().is_empty() {
            failure.message_next = false;
            failure.message = Some(line.trim().to_owned());
        }
    }

    fn pytest(&mut self, line: &str) {
        if let Some((node, status)) = line.split_once(' ')
            && node.contains("::")
        {
            let status = status.split_whitespace().next().unwrap_or_default();
            match status {
                "PASSED" | "XFAIL" | "XPASS" => self.passed += 1,
                "FAILED" | "ERROR" => {
                    self.fail(node);
                }
                "SKIPPED" => self.skipped += 1,
                _ => {}
            }
            if !status.is_empty() {
                return;
            }
        }
        if let Some(rest) = line.strip_prefix("FAILED ") {
            // The short summary: `FAILED tests/x.py::test_a - assert 1 == 2`.
            let (node, message) = rest.split_once(" - ").unwrap_or((rest, ""));
            let at = self.failure(node.trim());
            if let Some(failure) = self.failures.get_mut(at)
                && failure.message.is_none()
                && !message.is_empty()
            {
                failure.message = Some(message.trim().to_owned());
            }
            return;
        }
        if line.starts_with('=') {
            self.section = Section::None;
            let inner = line.trim_matches(|c| c == '=' || c == ' ');
            if let Some((counted, _)) = inner.rsplit_once(" in ")
                && (counted.contains("passed") || counted.contains("failed"))
            {
                let summary = counts(
                    counted,
                    ',',
                    &["passed"],
                    &["failed", "error", "errors"],
                    &["skipped"],
                );
                self.add_summary(summary);
            }
            return;
        }
        if line.starts_with("___") && line.ends_with("___") {
            let name = line.trim_matches(|c| c == '_' || c == ' ');
            let dotted = name.replace('.', "::");
            let known = self.failures.iter().position(|failure| {
                failure.name == name || failure.name.ends_with(&format!("::{dotted}"))
            });
            let at = match known {
                Some(at) => at,
                None => self.failure(name),
            };
            self.section = Section::Failure(at);
            return;
        }
        let Section::Failure(at) = self.section else {
            return;
        };
        let Some(failure) = self.failures.get_mut(at) else {
            return;
        };
        failure.push(line);
        if let Some(message) = line.strip_prefix("E ") {
            if failure.message.is_none() {
                failure.message = Some(message.trim().to_owned());
            }
        } else if let Some(place) = file_line(line) {
            failure.location = Some(place);
        }
    }

    fn go(&mut self, line: &str) {
        let trimmed = line.trim_start();
        if let Some(name) = trimmed.strip_prefix("=== RUN") {
            self.section = Section::GoRun(name.trim().to_owned(), Vec::new());
            return;
        }
        let ended = |prefix: &str| {
            let rest = trimmed.strip_prefix(prefix)?;
            Some(rest.split(" (").next().unwrap_or(rest).trim().to_owned())
        };
        if ended("--- PASS:").is_some() {
            self.passed += 1;
            self.section = Section::None;
            return;
        }
        if ended("--- SKIP:").is_some() {
            self.skipped += 1;
            self.section = Section::None;
            return;
        }
        if let Some(name) = ended("--- FAIL:") {
            let earlier = match std::mem::take(&mut self.section) {
                Section::GoRun(running, lines) if running == name => lines,
                _ => Vec::new(),
            };
            let at = self.fail(&name);
            for earlier in earlier {
                self.go_line(at, &earlier);
            }
            self.section = Section::Failure(at);
            return;
        }
        if !line.starts_with(' ') {
            // `FAIL`, `ok  pkg 0.1s`, `PASS`: a package's end.
            self.section = Section::None;
            return;
        }
        match &mut self.section {
            Section::GoRun(_, lines) => lines.push(line.to_owned()),
            Section::Failure(at) => {
                let at = *at;
                self.go_line(at, line);
            }
            Section::None => {}
        }
    }

    /// A line of the output of failed Go test `at`: `    x_test.go:12: got 1`.
    fn go_line(&mut self, at: usize, line: &str) {
        let Some(failure) = self.failures.get_mut(at) else {
            return;
        };
        failure.push(line);
        let trimmed = line.trim_start();
        if failure.location.is_none()
            && let Some(place) = file_line(trimmed)
        {
            failure.message = trimmed
                .get(place.len() + 1..)
                .map(|message| message.trim().to_owned());
            failure.location = Some(place);
        }
    }
}

/// `path:line` when `line` starts with one followed by `:`, as in
/// `tests/test_math.py:7: AssertionError`.
fn file_line(line: &str) -> Option<String> {
    let (path, rest) = line.split_once(':')?;
    let (number, _) = rest.split_once(':')?;
    let looks = !path.is_empty()
        && !path.contains(char::is_whitespace)
        && path.contains('.')
        && !number.is_empty()
        && number.chars().all(|c| c.is_ascii_digit());
    looks.then(|| format!("{path}:{number}"))
}

/// The passed, failed and skipped counts of a summary such as
/// `8 passed; 2 failed; 0 ignored` split at `separator`.
fn counts(
    summary: &str,
    separator: char,
    passed: &[&str],
    failed: &[&str],
    skipped: &[&str],
) -> [usize; 3] {
    let mut found = [0; 3];
    for part in summary.split(separator) {
        let mut words = part.split_whitespace().rev();
        let (Some(word), Some(number)) = (words.next(), words.next()) else {
            continue;
        };
        let Ok(number) = number.parse::<usize>() else {
            continue;
        };
        let word = word.trim_matches(|c: char| !c.is_alphabetic());
        for (slot, names) in found.iter_mut().zip([passed, failed, skipped]) {
            if names.contains(&word) {
                *slot += number;
            }
        }
    }
    found
}
