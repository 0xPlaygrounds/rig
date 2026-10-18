//! A verb's command line: its options, as [`guide::OPTIONS`] declares them,
//! and its other words. An option a verb does not take is a usage error
//! naming `rig help VERB`.

use std::time::Duration;

use rig_harness_protocol::guide;

use crate::out::Failure;

/// A verb's options and words, in the order given.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Args {
    /// Each option by its long name, with its value if it takes one.
    pub options: Vec<(&'static str, Option<String>)>,
    /// Every other word.
    pub words: Vec<String>,
}

impl Args {
    /// Reads `args` as verb `verb`'s.
    pub fn parse(verb: &str, args: &[String]) -> Result<Args, Failure> {
        let mut parsed = Args::default();
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if arg == "--" {
                parsed.words.extend(args.by_ref().cloned());
                break;
            }
            if arg == "-" || !arg.starts_with('-') {
                parsed.words.push(arg.clone());
                continue;
            }
            let Some(option) = guide::option(arg).filter(|option| option.verbs.contains(&verb))
            else {
                return Err(Failure::usage(
                    format!("rig {verb} takes no option `{arg}`"),
                    verb,
                ));
            };
            let inline = arg.split_once('=').map(|(_, value)| value.to_owned());
            // On rig answer, --allow and --deny are switches.
            let takes_value = option.value.is_some()
                && !(verb == "answer" && matches!(option.name, "allow" | "deny"));
            let value = match (takes_value, inline) {
                (true, Some(value)) => Some(value),
                (true, None) => Some(args.next().cloned().ok_or_else(|| {
                    Failure::usage(
                        format!(
                            "--{} needs a {}",
                            option.name,
                            option.value.unwrap_or("value")
                        ),
                        verb,
                    )
                })?),
                (false, Some(_)) => {
                    return Err(Failure::usage(
                        format!("--{} takes no value", option.name),
                        verb,
                    ));
                }
                (false, None) => None,
            };
            parsed.options.push((option.name, value));
        }
        Ok(parsed)
    }

    /// Whether switch `name` was given.
    pub fn has(&self, name: &str) -> bool {
        self.options.iter().any(|(option, _)| *option == name)
    }

    /// The last value of option `name`.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.options
            .iter()
            .rev()
            .find(|(option, _)| *option == name)
            .and_then(|(_, value)| value.as_deref())
    }

    /// Every value of option `name`, in order.
    pub fn values<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.options
            .iter()
            .filter(move |(option, _)| *option == name)
            .filter_map(|(_, value)| value.as_deref())
    }

    /// The first word, which names the session: a usage error without one.
    pub fn session(&self, verb: &str) -> Result<&str, Failure> {
        self.words
            .first()
            .map(String::as_str)
            .ok_or_else(|| Failure::usage(format!("rig {verb} needs a session id"), verb))
    }

    /// Option `name` as a number.
    pub fn number(&self, name: &str, verb: &str) -> Result<Option<u64>, Failure> {
        self.value(name)
            .map(|value| {
                value.parse().map_err(|_| {
                    Failure::usage(format!("--{name} needs a number, not `{value}`"), verb)
                })
            })
            .transpose()
    }

    /// Option `name` as a duration; `0` is `None`, waiting forever.
    pub fn duration(&self, name: &str, verb: &str) -> Result<Option<Option<Duration>>, Failure> {
        self.value(name)
            .map(|value| {
                parse_duration(value)
                    .map(|duration| (!duration.is_zero()).then_some(duration))
                    .ok_or_else(|| {
                        Failure::usage(
                            format!(
                                "--{name} needs a duration such as 30s, 10m or 1h, not `{value}`"
                            ),
                            verb,
                        )
                    })
            })
            .transpose()
    }
}

/// `30s`, `10m`, `1h`, `500ms`, or plain seconds.
pub fn parse_duration(text: &str) -> Option<Duration> {
    let text = text.trim();
    let split = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let number: u64 = number.parse().ok()?;
    match unit {
        "" | "s" => Some(Duration::from_secs(number)),
        "ms" => Some(Duration::from_millis(number)),
        "m" => Some(Duration::from_secs(number.checked_mul(60)?)),
        "h" => Some(Duration::from_secs(number.checked_mul(3600)?)),
        _ => None,
    }
}
