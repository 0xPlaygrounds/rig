// =====================================================
// Utilities to print results from a Neo4j vector search
// =====================================================

use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug)]
pub struct SearchResult {
    pub title: String,
    pub id: String,
    pub description: String,
    pub score: f64,
}

pub struct SearchResults<'a>(pub &'a Vec<SearchResult>);

impl<'a> Display for SearchResults<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = term_size::dimensions().map(|(w, _)| w).unwrap_or(150);
        let title_width = 40;
        let id_width = 10;
        let description_width = width - title_width - id_width - 2; // 2 for spaces

        writeln!(
            f,
            "{:<title_width$} {:<id_width$} {:<description_width$}",
            "Title", "ID", "Description"
        )?;
        writeln!(f, "{}", "-".repeat(width))?;
        for result in self.0 {
            let wrapped_title = textwrap::fill(&result.title, title_width);
            let wrapped_description = textwrap::fill(&result.description, description_width);
            let title_lines: Vec<&str> = wrapped_title.lines().collect();
            let description_lines: Vec<&str> = wrapped_description.lines().collect();
            let max_lines = title_lines.len().max(description_lines.len());

            for i in 0..max_lines {
                let title_line = title_lines.get(i).unwrap_or(&"");
                let description_line = description_lines.get(i).unwrap_or(&"");
                if i == 0 {
                    writeln!(
                        f,
                        "{:<title_width$} {:<id_width$} {:<description_width$}",
                        title_line, result.id, description_line
                    )?;
                } else {
                    writeln!(
                        f,
                        "{:<title_width$} {:<id_width$} {:<description_width$}",
                        title_line, "", description_line
                    )?;
                }
            }
        }
        Ok(())
    }
}
