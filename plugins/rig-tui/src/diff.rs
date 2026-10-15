//! The colouring of a unified diff a tool returned, in the transcript.

use ratatui::style::Stylize;
use ratatui::text::Line;

/// `text` coloured as a unified diff when it holds a hunk header, else
/// `None`.
pub fn unified_lines(text: &str) -> Option<Vec<Line<'static>>> {
    if !text.lines().any(|line| line.starts_with("@@")) {
        return None;
    }
    let lines: Vec<Line<'static>> = text
        .lines()
        .map(|line| {
            let line = line.replace('\t', "    ");
            if line.starts_with("+++") || line.starts_with("---") {
                Line::from(line).bold()
            } else if line.starts_with("@@") {
                Line::from(line).cyan()
            } else if line.starts_with('+') {
                Line::from(line).green()
            } else if line.starts_with('-') {
                Line::from(line).red()
            } else {
                Line::from(line).dim()
            }
        })
        .collect();
    Some(lines)
}
