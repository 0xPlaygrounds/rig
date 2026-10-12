//! Keystrokes as bytes a terminal program reads: the agent's notation
//! (`<Enter>`, `<C-c>`) and the keys the user presses in the panel.

use rig_tui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// The bytes of a named key, such as `Enter` or `Up`.
fn named(name: &str) -> Option<Vec<u8>> {
    let bytes: &[u8] = match name.to_ascii_lowercase().as_str() {
        "enter" | "cr" | "return" => b"\r",
        "tab" => b"\t",
        "esc" | "escape" => b"\x1b",
        "space" => b" ",
        "bs" | "backspace" => b"\x7f",
        "del" | "delete" => b"\x1b[3~",
        "up" => b"\x1b[A",
        "down" => b"\x1b[B",
        "right" => b"\x1b[C",
        "left" => b"\x1b[D",
        "home" => b"\x1b[H",
        "end" => b"\x1b[F",
        "pageup" | "pgup" => b"\x1b[5~",
        "pagedown" | "pgdn" => b"\x1b[6~",
        "insert" => b"\x1b[2~",
        "lt" => b"<",
        "f1" => b"\x1bOP",
        "f2" => b"\x1bOQ",
        "f3" => b"\x1bOR",
        "f4" => b"\x1bOS",
        "f5" => b"\x1b[15~",
        "f6" => b"\x1b[17~",
        "f7" => b"\x1b[18~",
        "f8" => b"\x1b[19~",
        "f9" => b"\x1b[20~",
        "f10" => b"\x1b[21~",
        "f11" => b"\x1b[23~",
        "f12" => b"\x1b[24~",
        _ => {
            let lower = name.to_ascii_lowercase();
            if let Some(key) = lower.strip_prefix("c-") {
                let mut chars = key.chars();
                return match (chars.next(), chars.next()) {
                    (Some(key), None) => control(key).map(|byte| vec![byte]),
                    _ => None,
                };
            }
            if let Some(key) = name.get(2..).filter(|_| lower.starts_with("m-")) {
                let mut bytes = vec![0x1b];
                bytes.extend(named(key).unwrap_or_else(|| key.as_bytes().to_vec()));
                return Some(bytes);
            }
            return None;
        }
    };
    Some(bytes.to_vec())
}

/// Ctrl+`key`.
fn control(key: char) -> Option<u8> {
    match key {
        'a'..='z' | '@' | '[' | '\\' | ']' | '^' | '_' => u8::try_from(key.to_ascii_uppercase())
            .ok()
            .map(|byte| byte & 0x1f),
        '?' => Some(0x7f),
        _ => None,
    }
}

/// `input` as the bytes it types: text as it is, `<Name>` as that key.
/// An unknown `<…>` is typed as it is.
pub fn parse(input: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut rest = input;
    while let Some(open) = rest.find('<') {
        let (text, from) = rest.split_at(open);
        bytes.extend_from_slice(text.as_bytes());
        let key = from
            .get(1..)
            .and_then(|after| after.find('>').map(|close| (after, close)))
            .and_then(|(after, close)| {
                let name = after.get(..close)?;
                Some((named(name)?, after.get(close + 1..)?))
            });
        match key {
            Some((key, after)) => {
                bytes.extend(key);
                rest = after;
            }
            None => {
                bytes.push(b'<');
                rest = from.get(1..).unwrap_or_default();
            }
        }
    }
    bytes.extend_from_slice(rest.as_bytes());
    bytes
}

/// The bytes a key the user pressed types.
pub fn of_key(key: &KeyEvent) -> Vec<u8> {
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let mut bytes = if alt { vec![0x1b] } else { Vec::new() };
    let name = match key.code {
        KeyCode::Char(c) if ctrl => {
            bytes.extend(control(c.to_ascii_lowercase()));
            return bytes;
        }
        KeyCode::Char(c) => {
            bytes.extend(c.to_string().into_bytes());
            return bytes;
        }
        KeyCode::Enter => "enter",
        KeyCode::Tab => "tab",
        KeyCode::BackTab => {
            bytes.extend(b"\x1b[Z");
            return bytes;
        }
        KeyCode::Esc => "esc",
        KeyCode::Backspace => "bs",
        KeyCode::Delete => "del",
        KeyCode::Up => "up",
        KeyCode::Down => "down",
        KeyCode::Left => "left",
        KeyCode::Right => "right",
        KeyCode::Home => "home",
        KeyCode::End => "end",
        KeyCode::PageUp => "pageup",
        KeyCode::PageDown => "pagedown",
        KeyCode::Insert => "insert",
        KeyCode::F(number) => {
            bytes.extend(named(&format!("f{number}")).unwrap_or_default());
            return bytes;
        }
        _ => return Vec::new(),
    };
    bytes.extend(named(name).unwrap_or_default());
    bytes
}
