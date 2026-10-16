//! The rules, judged against the repository each git command would run in:
//! its options read as git reads them, its aliases expanded, and its
//! branches, upstream and uncommitted files asked of git there.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use rig_harness::rig_tools::process::{Launcher, Spawn};
use rig_harness::rig_tools::workspace::Workspace;

use crate::git_guardrails::shell::{self, GitCall, Word};
use crate::git_guardrails::{GitGuardConfig, Rule};

/// How long one question to git may take.
const TIMEOUT: Duration = Duration::from_secs(10);

/// How many aliases one command may go through.
const ALIASES: usize = 8;

/// Git commands no rule is about. Built-in commands win over aliases, so
/// a command named here is never an alias either.
const HARMLESS: &[&str] = &[
    "add",
    "am",
    "apply",
    "archive",
    "bisect",
    "blame",
    "bundle",
    "cat-file",
    "cherry-pick",
    "clone",
    "config",
    "describe",
    "diff",
    "fetch",
    "for-each-ref",
    "format-patch",
    "gc",
    "grep",
    "help",
    "init",
    "log",
    "ls-files",
    "ls-remote",
    "merge",
    "merge-base",
    "mv",
    "notes",
    "pull",
    "reflog",
    "remote",
    "rev-list",
    "rev-parse",
    "revert",
    "rm",
    "shortlog",
    "show",
    "show-ref",
    "status",
    "submodule",
    "switch",
    "symbolic-ref",
    "tag",
    "version",
    "worktree",
];

/// Git commands a rule is about.
const RULED: &[&str] = &[
    "branch", "checkout", "clean", "commit", "push", "rebase", "reset", "restore", "stash",
];

/// A rule a command breaks, and what the model reads about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// The rule.
    pub rule: Rule,
    /// What the command would do, and what to do instead.
    pub text: String,
}

fn finding(rule: Rule, what: String) -> Finding {
    Finding {
        rule,
        text: format!("{what}. {}", rule.instead()),
    }
}

/// Whether a rule may be about `call`: a ruled command, or one that may
/// be an alias.
pub fn may_matter(call: &GitCall) -> bool {
    Parsed::of(call).is_some_and(|parsed| !HARMLESS.contains(&parsed.sub.as_str()))
}

/// What judges commands: where git runs, how it starts, the rules, and
/// the files the agent changed this session.
pub struct Judge {
    /// The project's workspace.
    pub workspace: Arc<dyn Workspace>,
    /// How git is started.
    pub launcher: Launcher,
    /// The rules and the protected branches.
    pub config: GitGuardConfig,
    /// Every file the agent's calls changed, as absolute paths.
    pub agent_changed: Vec<String>,
}

impl Judge {
    /// The rules `calls` break, at every level.
    pub fn judge(&self, calls: &[GitCall], depth: u8) -> Vec<Finding> {
        calls
            .iter()
            .flat_map(|call| self.call(call, depth))
            .collect()
    }

    fn call(&self, call: &GitCall, depth: u8) -> Vec<Finding> {
        let Some(mut parsed) = Parsed::of(call) else {
            return Vec::new();
        };
        let repo = Repo {
            judge: self,
            dir: call.dir.clone(),
            globals: parsed.globals.clone(),
        };
        let mut aliases: Option<HashMap<String, String>> = None;
        for _ in 0..ALIASES {
            if HARMLESS.contains(&parsed.sub.as_str()) || RULED.contains(&parsed.sub.as_str()) {
                break;
            }
            let value = parsed.inline.get(&parsed.sub).cloned().or_else(|| {
                let aliases = aliases.get_or_insert_with(|| repo.aliases());
                aliases.get(&parsed.sub).cloned()
            });
            let Some(value) = value else { break };
            if let Some(line) = value.strip_prefix('!') {
                // A shell alias runs from the top of the work tree.
                let rest: Vec<&str> = parsed.rest.iter().map(|word| word.text.as_str()).collect();
                let line = format!("{line} {}", rest.join(" "));
                let top = repo.toplevel().unwrap_or_else(|| call.dir.clone());
                return if depth < 4 {
                    self.judge(&shell::git_calls(&line, &top), depth + 1)
                } else {
                    Vec::new()
                };
            }
            let mut words = shell::words(&value).into_iter();
            let Some(sub) = words.next() else { break };
            parsed.sub = sub.text;
            parsed.rest = words.chain(std::mem::take(&mut parsed.rest)).collect();
        }
        repo.check(&parsed)
    }
}

/// A git command's words: the options before its subcommand, the
/// subcommand, the rest, and aliases `-c alias.x=…` defines.
struct Parsed {
    globals: Vec<String>,
    sub: String,
    rest: Vec<Word>,
    inline: HashMap<String, String>,
}

impl Parsed {
    fn of(call: &GitCall) -> Option<Self> {
        let mut globals = Vec::new();
        let mut inline = HashMap::new();
        let mut words = call.args.iter();
        while let Some(word) = words.next() {
            let text = word.text.as_str();
            if !text.starts_with('-') {
                return Some(Self {
                    globals,
                    sub: text.to_owned(),
                    rest: words.cloned().collect(),
                    inline,
                });
            }
            match text {
                "-p" | "--paginate" => {}
                "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace" | "--config-env" => {
                    let value = words.next()?;
                    if let Some((key, alias)) = value.text.split_once('=')
                        && let Some(name) = key.strip_prefix("alias.")
                        && text == "-c"
                    {
                        inline.insert(name.to_owned(), alias.to_owned());
                    }
                    globals.extend([text.to_owned(), value.text.clone()]);
                }
                _ => globals.push(text.to_owned()),
            }
        }
        None
    }

    /// The command as the model reads it, after aliases.
    fn shown(&self) -> String {
        let rest: Vec<&str> = self.rest.iter().map(|word| word.text.as_str()).collect();
        format!("git {} {}", self.sub, rest.join(" "))
            .trim_end()
            .to_owned()
    }
}

/// A branch a push updates or deletes; `None` when the line only knows it
/// when it runs.
struct Target {
    branch: Option<String>,
    forced: bool,
    deletes: bool,
}

/// The repository a command runs in, asked through git with the
/// command's own options before the subcommand.
struct Repo<'a> {
    judge: &'a Judge,
    dir: PathBuf,
    globals: Vec<String>,
}

impl Repo<'_> {
    /// What `git <globals> <args>` printed, when it succeeded.
    fn git(&self, args: &[&str]) -> Option<String> {
        let mut all: Vec<OsString> = self.globals.iter().map(OsString::from).collect();
        all.extend(args.iter().map(OsString::from));
        let launcher = self
            .judge
            .launcher
            .clone()
            .with_env("GIT_TERMINAL_PROMPT", "0")
            .with_env("GIT_OPTIONAL_LOCKS", "0");
        let process = self
            .judge
            .workspace
            .spawn(
                &launcher,
                Some(&self.dir),
                OsStr::new("git"),
                &all,
                Spawn::default(),
            )
            .ok()?;
        let output = process.output(Some(TIMEOUT)).ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        output
            .status
            .success()
            .then(|| text.trim_end_matches('\n').to_owned())
    }

    fn succeeds(&self, args: &[&str]) -> bool {
        self.git(args).is_some()
    }

    fn aliases(&self) -> HashMap<String, String> {
        let listed = self
            .git(&["config", "--get-regexp", r"^alias\."])
            .unwrap_or_default();
        listed
            .lines()
            .filter_map(|line| {
                let (key, value) = line.split_once(' ')?;
                Some((key.strip_prefix("alias.")?.to_owned(), value.to_owned()))
            })
            .collect()
    }

    fn toplevel(&self) -> Option<PathBuf> {
        self.git(&["rev-parse", "--show-toplevel"])
            .map(PathBuf::from)
    }

    fn current_branch(&self) -> Option<String> {
        self.git(&["symbolic-ref", "--short", "-q", "HEAD"])
    }

    /// The remote branch a bare `git push` updates.
    fn push_branch(&self) -> Option<String> {
        let push = self.git(&[
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{push}",
        ]);
        push.and_then(|name| Some(name.split_once('/')?.1.to_owned()))
            .or_else(|| self.current_branch())
    }

    /// Whether moving `head` so only what `base` reaches stays (`None`:
    /// nothing stays) drops a commit already on `head`'s upstream: that
    /// commit, shortened, and the upstream.
    fn pushed_rewrite(&self, base: Option<&str>, head: &str) -> Option<(String, String)> {
        let upstream_ref = if head == "HEAD" {
            "@{u}".to_owned()
        } else {
            format!("{head}@{{u}}")
        };
        let upstream = self.git(&[
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            &upstream_ref,
        ])?;
        let shared = self.git(&["merge-base", head, &upstream_ref])?;
        if let Some(base) = base
            && let Some(kept) = self.git(&["merge-base", base, head])
            && self.succeeds(&["merge-base", "--is-ancestor", &shared, &kept])
        {
            return None;
        }
        let short = self.git(&["rev-parse", "--short", &shared]);
        Some((short.unwrap_or(shared), upstream))
    }

    /// The uncommitted files the agent did not change, from the top of the
    /// work tree.
    fn user_changes(&self) -> Vec<String> {
        let (Some(top), Some(status)) = (
            self.toplevel(),
            self.git(&["status", "--porcelain", "-z", "-uall"]),
        ) else {
            return Vec::new();
        };
        let workspace = &self.judge.workspace;
        let mine: Vec<PathBuf> = self
            .judge
            .agent_changed
            .iter()
            .map(|path| workspace.canonical(path.as_ref()))
            .collect();
        let mut fields = status.split('\0');
        let mut theirs = Vec::new();
        while let Some(entry) = fields.next() {
            let Some((code, path)) = entry.split_at_checked(3) else {
                continue;
            };
            if code.starts_with(['R', 'C']) {
                fields.next();
            }
            if !mine.contains(&workspace.canonical(&top.join(path))) {
                theirs.push(path.to_owned());
            }
        }
        theirs
    }

    fn discards(&self, shown: &str) -> Option<Finding> {
        let lost = self.user_changes();
        (!lost.is_empty()).then(|| {
            let what = format!(
                "`{shown}` would throw away uncommitted changes you did not make: {}",
                lost.join(", ")
            );
            finding(Rule::DiscardChanges, what)
        })
    }

    fn protected(&self, branch: &str) -> bool {
        self.judge.config.protects(branch)
    }

    fn check(&self, parsed: &Parsed) -> Vec<Finding> {
        let shown = parsed.shown();
        let args = parsed.rest.as_slice();
        match parsed.sub.as_str() {
            "push" => self.push(args, &shown),
            "commit" => self.commit(args, &shown),
            "rebase" => self.rebase(args, &shown).into_iter().collect(),
            "reset" => self.reset(args, &shown),
            "branch" => self.branch(args, &shown),
            "checkout" | "restore" | "clean" | "stash" if discarding(&parsed.sub, args) => {
                self.discards(&shown).into_iter().collect()
            }
            _ => Vec::new(),
        }
    }

    fn push(&self, args: &[Word], shown: &str) -> Vec<Finding> {
        let (mut force, mut no_verify, mut delete, mut every) = (false, false, false, false);
        let mut positional: Vec<&Word> = Vec::new();
        let mut words = args.iter();
        let mut dashes = false;
        while let Some(word) = words.next() {
            let text = word.text.as_str();
            if dashes || !text.starts_with('-') || text == "-" {
                positional.push(word);
                continue;
            }
            match text {
                "--" => dashes = true,
                "--force" => force = true,
                "--no-verify" => no_verify = true,
                "--delete" => delete = true,
                "--all" | "--branches" | "--mirror" => every = true,
                "--repo" | "--push-option" | "--receive-pack" | "--exec" | "-o" => {
                    words.next();
                }
                _ if text.starts_with("--force-with-lease") => force = true,
                _ if text.starts_with("--") => {}
                _ => {
                    force |= text.contains('f');
                    delete |= text.contains('d');
                }
            }
        }
        let mut findings = Vec::new();
        if no_verify {
            let what = format!("`{shown}` skips the pre-push hook with --no-verify");
            findings.push(finding(Rule::NoVerify, what));
        }
        let refspecs = positional.get(1..).unwrap_or_default();
        let targets: Vec<Target> = if refspecs.is_empty() {
            if !(force || delete) {
                return findings;
            }
            if every {
                let branches = self
                    .git(&["for-each-ref", "--format=%(refname:short)", "refs/heads"])
                    .unwrap_or_default();
                let branches = branches.lines().map(|branch| Target {
                    branch: Some(branch.to_owned()),
                    forced: force,
                    deletes: false,
                });
                branches.collect()
            } else {
                vec![Target {
                    branch: self.push_branch(),
                    forced: force,
                    deletes: delete,
                }]
            }
        } else {
            refspecs
                .iter()
                .map(|spec| self.target(spec, force, delete))
                .collect()
        };
        for target in targets {
            let protected = target
                .branch
                .as_deref()
                .is_none_or(|branch| self.protected(branch));
            if !protected {
                continue;
            }
            let named = target.branch.as_deref().map_or_else(
                || "a branch only known when the line runs (name it plainly)".to_owned(),
                |branch| format!("the protected branch `{branch}`"),
            );
            if target.deletes {
                let what = format!("`{shown}` deletes {named} on the remote");
                findings.push(finding(Rule::DeleteProtected, what));
            } else if target.forced {
                let what = format!("`{shown}` force-pushes to {named}");
                findings.push(finding(Rule::ForcePush, what));
            }
        }
        findings
    }

    /// The branch a push's `refspec` updates.
    fn target(&self, refspec: &Word, force: bool, delete: bool) -> Target {
        let plus = refspec.text.starts_with('+');
        let forced = force || plus;
        if refspec.dynamic {
            return Target {
                branch: None,
                forced,
                deletes: delete,
            };
        }
        let text = refspec.text.trim_start_matches('+');
        let (destination, removes) = match text.split_once(':') {
            Some(("", destination)) => (destination, true),
            Some((source, "")) => (source, false),
            Some((_, destination)) => (destination, false),
            None => (text, false),
        };
        let branch = if matches!(destination, "HEAD" | "@") {
            self.current_branch()
        } else {
            Some(destination.trim_start_matches("refs/heads/").to_owned())
        };
        Target {
            branch,
            forced,
            deletes: delete || removes,
        }
    }

    fn commit(&self, args: &[Word], shown: &str) -> Vec<Finding> {
        let (mut no_verify, mut amend) = (false, false);
        let mut words = args.iter();
        while let Some(word) = words.next() {
            let text = word.text.as_str();
            match text {
                "--" => break,
                "--no-verify" => no_verify = true,
                "--amend" => amend = true,
                "--message" | "--file" | "--author" | "--date" | "--reuse-message"
                | "--reedit-message" | "--fixup" | "--squash" | "--template" | "--cleanup"
                | "--trailer" => {
                    words.next();
                }
                _ if text.starts_with("--") || !text.starts_with('-') => {}
                _ => {
                    let flags = text.get(1..).unwrap_or_default();
                    for (at, flag) in flags.char_indices() {
                        no_verify |= flag == 'n';
                        // An option that takes a value takes the rest.
                        if "mFCct".contains(flag) {
                            if at + flag.len_utf8() == flags.len() {
                                words.next();
                            }
                            break;
                        }
                    }
                }
            }
        }
        let mut findings = Vec::new();
        if no_verify {
            let what = format!("`{shown}` skips the commit hooks with --no-verify");
            findings.push(finding(Rule::NoVerify, what));
        }
        if amend {
            let parent = self
                .succeeds(&["rev-parse", "--verify", "-q", "HEAD^"])
                .then_some("HEAD^");
            if let Some((commit, upstream)) = self.pushed_rewrite(parent, "HEAD") {
                let what = format!("`{shown}` rewrites commit {commit}, already on {upstream}");
                findings.push(finding(Rule::RewritePushed, what));
            }
        }
        findings
    }

    fn rebase(&self, args: &[Word], shown: &str) -> Option<Finding> {
        let mut root = false;
        let mut positional: Vec<&str> = Vec::new();
        let mut words = args.iter();
        while let Some(word) = words.next() {
            match word.text.as_str() {
                "--continue"
                | "--abort"
                | "--skip"
                | "--quit"
                | "--edit-todo"
                | "--show-current-patch" => return None,
                "--root" => root = true,
                "--onto" | "--strategy" | "--strategy-option" | "--exec" | "-s" | "-X" | "-x" => {
                    words.next();
                }
                text if text.starts_with('-') => {}
                text => positional.push(text),
            }
        }
        let head = positional.get(1).copied().unwrap_or("HEAD");
        let base = (!root).then(|| positional.first().copied().unwrap_or("@{u}"));
        let (commit, upstream) = self.pushed_rewrite(base, head)?;
        let what = format!("`{shown}` rewrites commit {commit}, already on {upstream}");
        Some(finding(Rule::RewritePushed, what))
    }

    fn reset(&self, args: &[Word], shown: &str) -> Vec<Finding> {
        let (mut hard, mut paths) = (false, false);
        let mut target = None;
        for word in args {
            match word.text.as_str() {
                "--hard" => hard = true,
                "--" => paths = true,
                text if text.starts_with('-') => {}
                text if target.is_none() && !paths => {
                    let commit = format!("{text}^{{commit}}");
                    if self.succeeds(&["rev-parse", "--verify", "-q", &commit]) {
                        target = Some(text);
                    } else {
                        paths = true;
                    }
                }
                _ => {}
            }
        }
        let mut findings = Vec::new();
        if let Some(target) = target.filter(|_| !paths)
            && let Some((commit, upstream)) = self.pushed_rewrite(Some(target), "HEAD")
        {
            let what = format!(
                "`{shown}` moves the branch back past commit {commit}, already on {upstream}"
            );
            findings.push(finding(Rule::RewritePushed, what));
        }
        if hard {
            findings.extend(self.discards(shown));
        }
        findings
    }

    fn branch(&self, args: &[Word], shown: &str) -> Vec<Finding> {
        let short = |flag: char| {
            args.iter().any(|word| {
                word.text.starts_with('-')
                    && !word.text.starts_with("--")
                    && word.text.contains(flag)
            })
        };
        let long = |flag: &str| args.iter().any(|word| word.text == flag);
        let delete = long("--delete") || short('d') || short('D');
        let remote = long("--remotes") || short('r');
        if !delete {
            return Vec::new();
        }
        let names = args.iter().filter(|word| !word.text.starts_with('-'));
        names
            .filter_map(|word| {
                let name = if remote {
                    word.text
                        .split_once('/')
                        .map_or(word.text.as_str(), |(_, name)| name)
                } else {
                    word.text.as_str()
                };
                (word.dynamic || self.protected(name)).then(|| {
                    let what = format!("`{shown}` deletes the protected branch `{name}`");
                    finding(Rule::DeleteProtected, what)
                })
            })
            .collect()
    }
}

/// Whether a `checkout`, `restore`, `clean` or `stash` command throws
/// away uncommitted work.
fn discarding(sub: &str, args: &[Word]) -> bool {
    let has = |flag: &str| args.iter().any(|word| word.text == flag);
    let short = |flag: char| {
        args.iter().any(|word| {
            word.text.starts_with('-') && !word.text.starts_with("--") && word.text.contains(flag)
        })
    };
    match sub {
        "checkout" => {
            let paths = args
                .iter()
                .position(|word| word.text == "--")
                .is_some_and(|at| args.len() > at + 1);
            paths || has(".") || has("-f") || has("--force")
        }
        "restore" => !(has("--staged") || has("-S")) || has("--worktree") || has("-W"),
        "clean" => (has("--force") || short('f')) && !(has("--dry-run") || short('n')),
        "stash" => args
            .iter()
            .find(|word| !word.text.starts_with('-'))
            .is_some_and(|word| matches!(word.text.as_str(), "drop" | "clear")),
        _ => false,
    }
}
