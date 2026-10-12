//! A plan file's checklist: its top-level `- [ ]` items with their nested
//! lines and annotations, the text above them, and how one item is marked
//! in the file as it is now, so the user's edits to other lines stay.

/// How an item is marked in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// `- [ ]`: still to do.
    Open,
    /// `- [x]`: done, by an agent or by hand.
    Done,
    /// `- [!]`: an agent failed it.
    Failed,
}

impl Mark {
    /// The character between the brackets.
    fn char(self) -> char {
        match self {
            Self::Open => ' ',
            Self::Done => 'x',
            Self::Failed => '!',
        }
    }

    fn of(c: char) -> Option<Self> {
        match c {
            ' ' => Some(Self::Open),
            'x' | 'X' => Some(Self::Done),
            '!' => Some(Self::Failed),
            _ => None,
        }
    }
}

/// An item an item waits for, as written in `(after: ..)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum After {
    /// By its number among the top-level items, from 1.
    Number(usize),
    /// The first other item whose title starts with this.
    Prefix(String),
}

/// A top-level item of the checklist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Its line in the file, from 0.
    line: usize,
    /// The leading whitespace of its line.
    indent: usize,
    /// How it is marked.
    pub mark: Mark,
    /// Its text without its annotations.
    pub title: String,
    /// Its nested lines, without the result lines under it.
    pub body: String,
    /// What it waits for, as written.
    pub after: Vec<After>,
    /// Whether it starts without waiting for the item before it.
    pub parallel: bool,
}

/// A plan: the text above its checklist and its top-level items.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checklist {
    /// The text above the first item.
    pub context: String,
    /// The top-level items, in order.
    pub items: Vec<Entry>,
}

/// A checklist line's indent, mark and text.
fn checkbox(line: &str) -> Option<(usize, Mark, &str)> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    let rest = trimmed
        .strip_prefix("- [")
        .or_else(|| trimmed.strip_prefix("* ["))?;
    let mut chars = rest.chars();
    let mark = Mark::of(chars.next()?)?;
    let text = chars.as_str().strip_prefix(']')?;
    if !text.is_empty() && !text.starts_with(' ') {
        return None;
    }
    Some((indent, mark, text.trim()))
}

/// Whether `line`, nested deeper than `indent`, is a result line.
fn is_result(line: &str, indent: usize) -> bool {
    let trimmed = line.trim_start();
    line.len() - trimmed.len() > indent && trimmed.starts_with('>')
}

/// `text` without `(parallel)` and `(after: ..)`, and what they said.
fn annotations(text: &str) -> (String, Vec<After>, bool) {
    let mut title = text.to_owned();
    let parallel = title.contains("(parallel)");
    title = title.replace("(parallel)", "");
    let mut after = Vec::new();
    if let Some(start) = title.find("(after:")
        && let Some(len) = title.get(start..).and_then(|rest| rest.find(')'))
    {
        let inside = title
            .get(start + "(after:".len()..start + len)
            .unwrap_or_default();
        for part in inside
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            let unquoted = part.trim_matches('"');
            match part.parse() {
                Ok(number) => after.push(After::Number(number)),
                Err(_) => after.push(After::Prefix(unquoted.to_owned())),
            }
        }
        title.replace_range(start..=start + len, "");
    }
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    (title, after, parallel)
}

/// Reads the checklist of `text`. The first checklist line sets the
/// top-level indent; deeper lines until the next top-level item are the
/// item's, and a shallower line ends it.
pub fn parse(text: &str) -> Checklist {
    let lines: Vec<&str> = text.lines().collect();
    let Some((first, top)) = lines
        .iter()
        .enumerate()
        .find_map(|(at, line)| checkbox(line).map(|(indent, ..)| (at, indent)))
    else {
        return Checklist {
            context: text.trim().to_owned(),
            items: Vec::new(),
        };
    };
    let context = lines.get(..first).unwrap_or_default().join("\n");
    let mut items: Vec<Entry> = Vec::new();
    let mut open = false;
    for (at, line) in lines.iter().enumerate().skip(first) {
        let indent = line.len() - line.trim_start().len();
        if let Some((indent, mark, text)) = checkbox(line).filter(|(indent, ..)| *indent <= top) {
            let (title, after, parallel) = annotations(text);
            items.push(Entry {
                line: at,
                indent,
                mark,
                title,
                body: String::new(),
                after,
                parallel,
            });
            open = true;
        } else if open && (line.trim().is_empty() || indent > top) {
            if let Some(item) = items.last_mut()
                && !is_result(line, top)
            {
                item.body.push_str(line.trim_end());
                item.body.push('\n');
            }
        } else {
            open = false;
        }
    }
    for item in &mut items {
        item.body = item.body.trim_end().to_owned();
    }
    Checklist {
        context: context.trim().to_owned(),
        items,
    }
}

impl Checklist {
    /// Each item's dependencies as item indices, from its `(after: ..)`.
    ///
    /// # Errors
    ///
    /// When one names no other item.
    pub fn dependencies(&self) -> Result<Vec<Vec<usize>>, String> {
        let resolve = |own: usize, after: &After| match after {
            After::Number(number) => Some(number.wrapping_sub(1))
                .filter(|&index| index < self.items.len() && index != own)
                .ok_or_else(|| {
                    format!(
                        "item {} waits for item {number}, which is not in the plan",
                        own + 1
                    )
                }),
            After::Prefix(prefix) => self
                .items
                .iter()
                .enumerate()
                .find(|(index, item)| *index != own && item.title.starts_with(prefix.as_str()))
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    format!(
                        "item {} waits for \"{prefix}\", which no other item starts with",
                        own + 1
                    )
                }),
        };
        self.items
            .iter()
            .enumerate()
            .map(|(own, item)| item.after.iter().map(|after| resolve(own, after)).collect())
            .collect()
    }
}

/// `text` with item `number` (from 1), titled `title`, marked `mark` and
/// `note` as the one result line under it, in place of those it had. The
/// item is found by its title, so lines the user added or changed
/// elsewhere stay. `None` when no item has that title any more.
pub fn mark(text: &str, number: usize, title: &str, mark: Mark, note: &str) -> Option<String> {
    let checklist = parse(text);
    let by_number = checklist
        .items
        .get(number.wrapping_sub(1))
        .filter(|item| item.title == title);
    let item = by_number.or_else(|| checklist.items.iter().find(|item| item.title == title))?;
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let line = lines.get_mut(item.line)?;
    let bracket = line.find('[')?;
    let (head, tail) = (line.get(..=bracket)?, line.get(bracket + 2..)?);
    let indent = line.get(..item.indent)?.to_owned();
    *line = format!("{head}{}{tail}", mark.char());
    let below = item.line + 1;
    while lines
        .get(below)
        .is_some_and(|line| is_result(line, item.indent))
    {
        lines.remove(below);
    }
    let note = note.split_whitespace().collect::<Vec<_>>().join(" ");
    lines.insert(below, format!("{indent}  > {note}"));
    let mut marked = lines.join("\n");
    if text.ends_with('\n') {
        marked.push('\n');
    }
    Some(marked)
}
