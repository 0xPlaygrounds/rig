//! Finding the git commands a `shell` line would run: through `&&`, `||`,
//! `;`, `|` and `&` chains, `( … )` subshells, `$( … )` and backtick
//! substitutions, `sh -c` and `eval`, and the `cd` before each, so every
//! one is judged in the directory it would start in.

use std::path::{Path, PathBuf};

/// How deep `sh -c`, `eval`, substitutions and shell aliases nest before
/// the rest is left alone.
const DEPTH: u8 = 4;

/// Words that run the command after them, with their own options first.
const WRAPPERS: &[&str] = &["sudo", "env", "command", "exec", "nohup", "time", "nice"];

/// A word of a command line, its quotes removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Word {
    /// The text as the command would get it, expansions left as written.
    pub text: String,
    /// Whether it holds a `$` expansion or a substitution, so its value is
    /// only known when the line runs.
    pub dynamic: bool,
}

/// One git command a line would run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitCall {
    /// The directory it starts in, before any `-C`.
    pub dir: PathBuf,
    /// Its words after `git`.
    pub args: Vec<Word>,
}

enum Token {
    Word(Word),
    /// `;`, `&&`, `||`, `|`, `&` or a new line.
    Separator,
    Open,
    Close,
    /// `>` or `<`: the next word is a file.
    Redirect,
    /// A substitution's command line.
    Nested(String),
}

/// The git commands `line` would run, started in `dir`.
pub fn git_calls(line: &str, dir: &Path) -> Vec<GitCall> {
    let mut calls = Vec::new();
    walk(line, dir, 0, &mut calls);
    calls
}

/// The words of `text` as a shell splits them, expansions left as written.
pub fn words(text: &str) -> Vec<Word> {
    lex(text)
        .into_iter()
        .filter_map(|token| match token {
            Token::Word(word) => Some(word),
            _ => None,
        })
        .collect()
}

fn walk(line: &str, dir: &Path, depth: u8, calls: &mut Vec<GitCall>) {
    if depth > DEPTH {
        return;
    }
    let mut dirs = vec![dir.to_path_buf()];
    let mut command: Vec<Word> = Vec::new();
    let mut redirected = false;
    for token in lex(line) {
        let here = dirs.last().cloned().unwrap_or_else(|| dir.to_path_buf());
        match token {
            Token::Word(_) if redirected => redirected = false,
            Token::Word(word) => command.push(word),
            Token::Redirect => redirected = true,
            Token::Nested(inner) => walk(&inner, &here, depth + 1, calls),
            Token::Separator | Token::Open | Token::Close => {
                let mut current = here;
                finish(&std::mem::take(&mut command), &mut current, depth, calls);
                match token {
                    Token::Open => dirs.push(current),
                    Token::Close if dirs.len() > 1 => {
                        dirs.pop();
                    }
                    _ => {
                        if let Some(top) = dirs.last_mut() {
                            *top = current;
                        }
                    }
                }
            }
        }
    }
    let mut current = dirs.last().cloned().unwrap_or_else(|| dir.to_path_buf());
    finish(&command, &mut current, depth, calls);
}

/// One simple command: a `cd` moves `dir`, a git command is kept, and a
/// shell given a line is searched.
fn finish(command: &[Word], dir: &mut PathBuf, depth: u8, calls: &mut Vec<GitCall>) {
    let mut rest = command;
    while let Some((first, after)) = rest.split_first() {
        let assignment = first
            .text
            .split_once('=')
            .is_some_and(|(name, _)| !name.is_empty() && !name.starts_with('-'));
        let wrapper = WRAPPERS.contains(&first.text.as_str());
        let option = first.text.starts_with('-') && rest.len() < command.len();
        if !(assignment || wrapper || option) {
            break;
        }
        rest = after;
    }
    let Some((program, args)) = rest.split_first() else {
        return;
    };
    let name = program.text.rsplit('/').next().unwrap_or_default();
    match name {
        "cd" | "pushd" => {
            let target = args.first().map_or("~", |word| word.text.as_str());
            *dir = dir.join(target);
        }
        "git" => calls.push(GitCall {
            dir: dir.clone(),
            args: args.to_vec(),
        }),
        "sh" | "bash" | "zsh" | "dash" => {
            let script = args
                .iter()
                .position(|word| word.text == "-c")
                .and_then(|at| args.get(at + 1));
            if let Some(script) = script {
                walk(&script.text, dir, depth + 1, calls);
            }
        }
        "eval" => {
            let line: Vec<&str> = args.iter().map(|word| word.text.as_str()).collect();
            walk(&line.join(" "), dir, depth + 1, calls);
        }
        _ => {}
    }
}

/// Splits `line` into words and the operators between them.
fn lex(line: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut word = Word::default();
    let mut started = false;
    let mut chars = line.chars().peekable();
    let end = |tokens: &mut Vec<Token>, word: &mut Word, started: &mut bool| {
        if *started {
            tokens.push(Token::Word(std::mem::take(word)));
            *started = false;
        }
    };
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' => end(&mut tokens, &mut word, &mut started),
            '\n' | ';' => {
                end(&mut tokens, &mut word, &mut started);
                tokens.push(Token::Separator);
            }
            '&' | '|' => {
                end(&mut tokens, &mut word, &mut started);
                if chars.peek() == Some(&c) {
                    chars.next();
                }
                tokens.push(Token::Separator);
            }
            '(' => {
                end(&mut tokens, &mut word, &mut started);
                tokens.push(Token::Open);
            }
            ')' => {
                end(&mut tokens, &mut word, &mut started);
                tokens.push(Token::Close);
            }
            '<' | '>' => {
                // `2>` names a stream, not a word.
                if started && word.text.chars().all(|c| c.is_ascii_digit()) {
                    word = Word::default();
                    started = false;
                }
                end(&mut tokens, &mut word, &mut started);
                while chars.peek().is_some_and(|next| matches!(next, '>' | '&')) {
                    chars.next();
                }
                tokens.push(Token::Redirect);
            }
            '#' if !started => {
                while chars.peek().is_some_and(|next| *next != '\n') {
                    chars.next();
                }
            }
            '\'' => {
                started = true;
                for next in chars.by_ref() {
                    if next == '\'' {
                        break;
                    }
                    word.text.push(next);
                }
            }
            '"' => {
                started = true;
                while let Some(next) = chars.next() {
                    match next {
                        '"' => break,
                        '\\' => match chars.next() {
                            Some(escaped @ ('$' | '`' | '"' | '\\')) => word.text.push(escaped),
                            Some('\n') | None => {}
                            Some(other) => {
                                word.text.push('\\');
                                word.text.push(other);
                            }
                        },
                        '$' | '`' => expansion(next, &mut chars, &mut word, &mut tokens),
                        other => word.text.push(other),
                    }
                }
            }
            '\\' => match chars.next() {
                Some('\n') | None => {}
                Some(escaped) => {
                    started = true;
                    word.text.push(escaped);
                }
            },
            '$' | '`' => {
                started = true;
                expansion(c, &mut chars, &mut word, &mut tokens);
            }
            other => {
                started = true;
                word.text.push(other);
            }
        }
    }
    end(&mut tokens, &mut word, &mut started);
    tokens
}

/// A `$` or backtick in a word: a substitution's line goes out as
/// [`Token::Nested`], and the word is dynamic.
fn expansion(
    first: char,
    chars: &mut std::iter::Peekable<std::str::Chars>,
    word: &mut Word,
    tokens: &mut Vec<Token>,
) {
    word.dynamic = true;
    word.text.push(first);
    if first == '`' {
        let inner: String = chars.by_ref().take_while(|next| *next != '`').collect();
        word.text.push_str(&inner);
        word.text.push('`');
        tokens.push(Token::Nested(inner));
        return;
    }
    if chars.peek() != Some(&'(') {
        return;
    }
    chars.next();
    let mut depth = 1usize;
    let mut inner = String::new();
    for next in chars.by_ref() {
        match next {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            break;
        }
        inner.push(next);
    }
    word.text.push('(');
    word.text.push_str(&inner);
    word.text.push(')');
    tokens.push(Token::Nested(inner));
}
