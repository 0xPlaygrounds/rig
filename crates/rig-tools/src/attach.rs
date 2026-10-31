//! Files a message names as `@path`, read to go with it: an image as an
//! image, any other file as its text, capped the way `read` shows it, so
//! the model needs no `read` call for it. [`Reference`] is the one way to
//! write and read such a name: `@notes.md`, or `@"my notes.md"` for one a
//! space or a trailing punctuation mark would cut.

use std::borrow::Cow;
use std::fmt;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::completion::Message;
use rig_core::conversation::{Conversation, EntryRef, FILE_CLOSE, FILE_OPEN, attached_file};
use rig_core::message::{ImageMediaType, UserContent};

use crate::workspace::{Stat, Workspace};
use crate::{MAX_LINES, show_lines};

/// The largest image attached: providers refuse one whose base64 is over
/// 5 MB, which this many bytes make.
pub(crate) const MOST_BYTES: u64 = 5 * 1024 * 1024 / 4 * 3;

/// The most bytes of a text file attached; the model reads on with `read`.
const MOST_TEXT_BYTES: usize = 32 * 1024;

/// The file extensions looked at as images.
const EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// What may follow an unquoted reference without being part of its path,
/// as in `see @notes.md, then`.
const TRAILING: [char; 10] = [',', '.', ';', ':', '!', '?', ')', '"', '\'', '`'];

/// Whether `path` names an image by its extension.
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// A file a message names with `@`, at the start of the text or after
/// whitespace. Unquoted, the path runs to the next whitespace, its
/// trailing punctuation left out: `@src/lib.rs,` names `src/lib.rs`.
/// Quoted, it is everything up to the closing quote, a doubled `""`
/// standing for one `"`: `@"my notes.md"`. [`format`](Self::format)
/// writes a path so that [`parse`](Self::parse) reads it back whole.
///
/// ```
/// use rig_tools::attach::Reference;
///
/// let text = format!("compare {} with @src/lib.rs.", Reference::format("my notes.md"));
/// assert_eq!(text, r#"compare @"my notes.md" with @src/lib.rs."#);
/// let paths: Vec<_> = Reference::parse(&text).map(|reference| reference.path).collect();
/// assert_eq!(paths, ["my notes.md", "src/lib.rs"]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference<'a> {
    /// Where it is in the text: from its `@` to its end, a closing quote
    /// included.
    pub span: Range<usize>,
    /// The path it names.
    pub path: Cow<'a, str>,
    /// Whether it is written quoted.
    pub quoted: bool,
}

impl<'a> Reference<'a> {
    /// The references in `text`, in order. A quote that is not closed
    /// runs to the end of the text.
    pub fn parse(text: &'a str) -> impl Iterator<Item = Reference<'a>> {
        let mut from = 0;
        std::iter::from_fn(move || {
            loop {
                let at = next_start(text, from)?;
                let after = text.get(at + 1..)?;
                if let Some(rest) = after.strip_prefix('"') {
                    let (path, len) = quoted(rest).unwrap_or_else(|| unclosed(rest));
                    from = at + 2 + len;
                    if !path.is_empty() {
                        return Some(Self::quoted(at..from, path));
                    }
                    continue;
                }
                let token = after.split(char::is_whitespace).next().unwrap_or_default();
                from = at + 1 + token.len();
                let path = token.trim_end_matches(TRAILING);
                if !path.is_empty() {
                    return Some(Self {
                        span: at..from,
                        path: Cow::Borrowed(path),
                        quoted: false,
                    });
                }
            }
        })
    }

    /// The reference being typed at the end of `text`, for completion: an
    /// unquoted one with its punctuation, or a quoted one not yet closed.
    pub fn typing(text: &'a str) -> Option<Reference<'a>> {
        let mut from = 0;
        loop {
            let at = next_start(text, from)?;
            let after = text.get(at + 1..)?;
            if let Some(rest) = after.strip_prefix('"') {
                match quoted(rest) {
                    Some((_, len)) => from = at + 2 + len,
                    None => return Some(Self::quoted(at..text.len(), unclosed(rest).0)),
                }
                continue;
            }
            let token = after.split(char::is_whitespace).next().unwrap_or_default();
            from = at + 1 + token.len();
            if from == text.len() {
                return Some(Self {
                    span: at..from,
                    path: Cow::Borrowed(token),
                    quoted: false,
                });
            }
        }
    }

    /// `path` written as a reference: `@path`, quoted when it is empty,
    /// holds whitespace, starts with `"` or ends in punctuation.
    pub fn format(path: &str) -> String {
        let plain = !path.is_empty()
            && !path.starts_with('"')
            && !path.contains(char::is_whitespace)
            && !path.ends_with(TRAILING);
        match plain {
            true => format!("@{path}"),
            false => format!("@\"{}\"", path.replace('"', "\"\"")),
        }
    }

    fn quoted(span: Range<usize>, path: Cow<'a, str>) -> Self {
        Self {
            span,
            path,
            quoted: true,
        }
    }
}

/// Where the next `@` that starts a word is, from byte `from` of `text` on.
fn next_start(text: &str, from: usize) -> Option<usize> {
    text.get(from..)?
        .match_indices('@')
        .map(|(at, _)| from + at)
        .find(|&at| {
            text.get(..at)
                .and_then(|before| before.chars().next_back())
                .is_none_or(char::is_whitespace)
        })
}

/// The quoted path at the start of `rest`, the text after an opening
/// quote, with `""` read as `"`, and its length with the closing quote;
/// `None` when no quote closes it.
fn quoted(rest: &str) -> Option<(Cow<'_, str>, usize)> {
    let mut at = 0;
    loop {
        let close = at + rest.get(at..)?.find('"')?;
        if rest.get(close + 1..)?.starts_with('"') {
            at = close + 2;
            continue;
        }
        let inside = rest.get(..close)?;
        let path = match inside.contains("\"\"") {
            true => Cow::Owned(inside.replace("\"\"", "\"")),
            false => Cow::Borrowed(inside),
        };
        return Some((path, close + 1));
    }
}

/// The path of a quote that is not closed: all of `rest`, with `""` read
/// as `"`, and its length.
fn unclosed(rest: &str) -> (Cow<'_, str>, usize) {
    (Cow::Owned(rest.replace("\"\"", "\"")), rest.len())
}

/// What a message's `@path` references came to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attached {
    /// The files read, each with its label: its path as the message named
    /// it, or where the workspace joins it when that is not below the base,
    /// such as `~/` made the home directory.
    pub files: Vec<(String, UserContent)>,
    /// A line for the user about each file that is there but could not be
    /// attached, and why.
    pub notes: Vec<String>,
    /// A line for the user about each reference that names a path with no
    /// file there, or one that could not be looked at, and why.
    pub warnings: Vec<String>,
}

/// The files `text` names ([`Reference::parse`]), each once, resolved from
/// `base` in `workspace` as a file tool resolves a path
/// (`Workspace::resolve`: `bench:~/x` is on the host `bench`
/// of a [`Hosts`](crate::remote::Hosts), `~/x` in the workspace's home)
/// and read from it. Every reference that names a file there is attached,
/// whatever its shape (`@notes.md`, `@src/lib.rs`, `@Makefile`, `@.env`);
/// one with no file there, or whose path could not be looked at, is
/// warned about when it [looks like a path](is_path_reference), and
/// otherwise stays text, such as `@alice`, `@alice.smith` or a pasted
/// C# string, `@"hello world"`.
///
/// Reads `workspace`, which may route a path to another machine: call it
/// off any thread that must not wait.
pub fn attachments(text: &str, base: &Path, workspace: &dyn Workspace) -> Attached {
    let mut attached = Attached::default();
    let mut seen: Vec<PathBuf> = Vec::new();
    for reference in Reference::parse(text) {
        let named = &*reference.path;
        // As typed, unless the workspace puts it elsewhere than below the
        // base, such as in a home directory or on a host.
        let joined = workspace.join(base, Path::new(named));
        let label = match joined == base.join(named) {
            true => named.to_owned(),
            false => joined.display().to_string(),
        };
        let path = workspace.resolve(Some(base), named);
        let warn = is_path_reference(named);
        match workspace.stat(&path) {
            Ok(Stat::File { .. }) => {}
            Ok(Stat::Missing) if warn => {
                let missing = format!("{}: no such file", Reference::format(named));
                if !attached.warnings.contains(&missing) {
                    attached.warnings.push(missing);
                }
                continue;
            }
            Ok(Stat::Dir | Stat::Other) if warn => {
                let note = format!("{label} is not attached: it is not a file.");
                if !attached.notes.contains(&note) {
                    attached.notes.push(note);
                }
                continue;
            }
            Err(failure) if warn => {
                let failed = format!("could not read {label}: {failure}");
                if !attached.warnings.contains(&failed) {
                    attached.warnings.push(failed);
                }
                continue;
            }
            Ok(_) | Err(_) => continue,
        }
        if seen.contains(&path) {
            continue;
        }
        let read = if is_image_path(Path::new(named)) {
            read_image(workspace, &path)
        } else {
            read_file(workspace, &path, &label)
        };
        match read {
            Ok(content) => attached.files.push((label, content)),
            Err(why) => attached
                .notes
                .push(format!("{label} is not attached: {why}.")),
        }
        seen.push(path);
    }
    attached
}

/// Whether `reference`, an `@` token without its `@`, looks like a path by
/// its shape alone: it holds a `/`, or ends in a dot and one to four
/// letters or digits. [`attachments`] warns about such a token when no
/// file is there; any other token (a mention, an address, or a file with
/// no extension) is attached only when its file is there, and is
/// otherwise left as text.
///
/// ```
/// use rig_tools::attach::is_path_reference;
///
/// assert!(is_path_reference("src/main.rs"));
/// assert!(is_path_reference("notes.md"));
/// assert!(!is_path_reference("reviewer"));
/// ```
pub fn is_path_reference(reference: &str) -> bool {
    reference.contains('/')
        || reference.rsplit_once('.').is_some_and(|(name, extension)| {
            !name.is_empty()
                && (1..=4).contains(&extension.len())
                && extension.chars().all(|c| c.is_ascii_alphanumeric())
        })
}

/// The image at `path` in `workspace` as message content, its type told
/// by its first bytes; why not, for a file that cannot be read, is not a
/// PNG, JPEG, GIF or WebP image, or is larger than a provider takes.
pub fn read_image(workspace: &dyn Workspace, path: &Path) -> Result<UserContent, String> {
    let image = image(workspace, path).map_err(|failure| failure.to_string())?;
    Ok(UserContent::image_base64(
        image.base64(),
        Some(image.media_type),
        None,
    ))
}

/// An image file, read whole.
pub(crate) struct Image {
    /// Its bytes.
    pub bytes: Vec<u8>,
    /// Its type, told by its first bytes.
    pub media_type: ImageMediaType,
}

impl Image {
    /// Its bytes in base64, as message content carries them.
    pub fn base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.bytes)
    }
}

/// Why [`image`] read no image.
#[derive(Debug)]
pub(crate) enum NoImage {
    /// The file could not be read.
    Read(io::Error),
    /// Its first bytes are not a PNG, JPEG, GIF or WebP image's.
    Other,
    /// It is an image of this many bytes, more than a provider takes.
    TooLarge(u64),
}

impl fmt::Display for NoImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(failure) => write!(formatter, "{failure}"),
            Self::Other => write!(formatter, "it is not a PNG, JPEG, GIF or WebP image"),
            Self::TooLarge(size) => write!(
                formatter,
                "it is an image of {} KB, more than the {} KB a provider takes",
                size.div_ceil(1024),
                MOST_BYTES / 1024
            ),
        }
    }
}

/// The image at `path` in `workspace`, whose first bytes tell its type.
pub(crate) fn image(workspace: &dyn Workspace, path: &Path) -> Result<Image, NoImage> {
    let contents = workspace.read(path, MOST_BYTES).map_err(NoImage::Read)?;
    let media_type = ImageMediaType::sniff(&contents.bytes).ok_or(NoImage::Other)?;
    if contents.size > MOST_BYTES {
        return Err(NoImage::TooLarge(contents.size));
    }
    Ok(Image {
        bytes: contents.bytes,
        media_type,
    })
}

/// The text file at `path` as message content: its lines the way `read`
/// shows them, at most [`MOST_TEXT_BYTES`], with a line saying how
/// many were left out, between a `<file path="label">` line and `</file>`.
fn read_file(workspace: &dyn Workspace, path: &Path, label: &str) -> Result<UserContent, String> {
    let text = workspace
        .read_text(path)
        .map_err(|failure| failure.to_string())?;
    if text.contains('\0') {
        return Err("it is not a text file".to_owned());
    }
    let lines = show_lines(&text, 1, MAX_LINES, MOST_TEXT_BYTES);
    Ok(UserContent::text(format!(
        "{FILE_OPEN}{label}\">\n{lines}{FILE_CLOSE}"
    )))
}

/// What the user typed in message `at` of `conversation`: the text items
/// of a message of the user's own
/// ([`EntryRef::is_users_own`]), never of
/// one an agent or a plugin delivered, such as a note, nor a file
/// attached, each trimmed, joined by a blank line; `None` when it has
/// none.
pub fn typed(conversation: &Conversation, at: usize) -> Option<String> {
    typed_in(conversation.entries().nth(at)?)
}

/// What the user typed in each message of `conversation` that has some
/// ([`typed`]), oldest first.
pub fn typed_prompts(conversation: &Conversation) -> impl Iterator<Item = String> + '_ {
    conversation.entries().filter_map(typed_in)
}

/// What the user typed in `entry` ([`typed`]).
fn typed_in(entry: EntryRef<'_>) -> Option<String> {
    let Message::User { content } = entry.message else {
        return None;
    };
    if !entry.is_users_own() {
        return None;
    }
    let typed: Vec<&str> = content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) if attached_file(&text.text).is_none() => {
                Some(text.text.trim())
            }
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect();
    (!typed.is_empty()).then(|| typed.join("\n\n"))
}

/// What the user typed first in `conversation` ([`typed`]).
pub fn first_typed(conversation: &Conversation) -> Option<String> {
    (0..conversation.messages().len()).find_map(|at| typed(conversation, at))
}

#[cfg(test)]
mod tests;
