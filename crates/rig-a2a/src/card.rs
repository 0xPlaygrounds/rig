//! Projection of a remote A2A agent card into Rig agent metadata.

use a2a::{AgentCard, AgentSkill};

/// Budget for metadata copied into each completion request when the agent is
/// converted into a tool.
pub(crate) const DESCRIPTION_LIMIT: usize = 8 * 1024;

/// Longest single skill rendering before it is elided.
const SKILL_LIMIT: usize = 512;

/// Room reserved for the omitted-skills note.
const NOTE_BUDGET: usize = 64;

/// Render a remote agent's card as the Rig agent description.
pub(crate) fn describe_card(card: &AgentCard) -> String {
    let mut out = card.description.clone();
    out.push('\n');

    if card.skills.is_empty() {
        return truncated(out, DESCRIPTION_LIMIT);
    }

    out.push_str("Skills:\n");
    let mut rendered_count = 0usize;
    for (index, skill) in card.skills.iter().enumerate() {
        let rendered = truncated(describe_skill(skill), SKILL_LIMIT);
        let reserved = if index + 1 == card.skills.len() {
            0
        } else {
            NOTE_BUDGET
        };
        if out.len() + rendered.len() + reserved > DESCRIPTION_LIMIT {
            break;
        }
        out.push_str(&rendered);
        rendered_count += 1;
    }

    let elided = card.skills.len() - rendered_count;
    if elided > 0 {
        tracing::warn!(
            target: "rig_a2a",
            agent = %card.name,
            elided,
            limit = DESCRIPTION_LIMIT,
            "remote agent card exceeds the agent description budget; some skills were omitted"
        );
        out.push_str(&truncated(
            format!("  … and {elided} further skill(s) omitted.\n"),
            NOTE_BUDGET,
        ));
    }
    truncated(out, DESCRIPTION_LIMIT)
}

fn describe_skill(skill: &AgentSkill) -> String {
    let mut line = format!("  - {} ({}): {}", skill.name, skill.id, skill.description);
    if !skill.tags.is_empty() {
        line.push_str(&format!(" [tags: {}]", skill.tags.join(", ")));
    }
    if let Some(examples) = skill
        .examples
        .as_ref()
        .filter(|examples| !examples.is_empty())
    {
        line.push_str(&format!(" [e.g. {}]", examples.join("; ")));
    }
    line.push('\n');
    line
}

fn truncated(mut text: String, limit: usize) -> String {
    const MARKER: &str = "…";
    if text.len() <= limit {
        return text;
    }
    let mut cut = limit.saturating_sub(MARKER.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
    text.push_str(MARKER);
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a::{AgentCapabilities, AgentInterface};

    fn card(name: &str, skills: Vec<AgentSkill>) -> AgentCard {
        AgentCard {
            name: name.to_string(),
            description: "A stub agent.".to_string(),
            version: "1.0".to_string(),
            supported_interfaces: vec![AgentInterface {
                url: "http://127.0.0.1:1".to_string(),
                protocol_binding: a2a::TRANSPORT_PROTOCOL_JSONRPC.to_string(),
                protocol_version: a2a::VERSION.to_string(),
                tenant: None,
            }],
            capabilities: AgentCapabilities::default(),
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            skills,
            provider: None,
            documentation_url: None,
            icon_url: None,
            security_schemes: None,
            security_requirements: None,
            signatures: None,
        }
    }

    fn skill(id: &str, name: &str) -> AgentSkill {
        AgentSkill {
            id: id.to_string(),
            name: name.to_string(),
            description: format!("Does {name}."),
            tags: vec!["demo".to_string()],
            examples: Some(vec![format!("please {name}")]),
            input_modes: None,
            output_modes: None,
            security_requirements: None,
        }
    }

    #[test]
    fn description_renders_card_skills() {
        let rendered = describe_card(&card("greeter", vec![skill("greet", "greet")]));
        assert!(rendered.contains("A stub agent."), "{rendered}");
        assert!(
            rendered.contains("- greet (greet): Does greet."),
            "{rendered}"
        );
        assert!(rendered.contains("[tags: demo]"), "{rendered}");
        assert!(rendered.contains("[e.g. please greet]"), "{rendered}");
    }

    #[test]
    fn description_without_skills_omits_the_section() {
        let rendered = describe_card(&card("bare", vec![]));
        assert!(!rendered.contains("Skills:"), "{rendered}");
    }

    #[test]
    fn description_is_capped() {
        let skills = (0..2000)
            .map(|index| skill(&format!("skill-{index}"), &format!("do thing {index}")))
            .collect();
        let rendered = describe_card(&card("verbose", skills));
        assert!(rendered.len() <= DESCRIPTION_LIMIT, "{}", rendered.len());
        assert!(rendered.contains("further skill(s) omitted"), "{rendered}");
    }

    #[test]
    fn overlong_single_skill_is_elided() {
        let mut long = skill("big", "big");
        long.description = "x".repeat(4096);
        let rendered = describe_card(&card("verbose", vec![long]));
        assert!(rendered.len() <= DESCRIPTION_LIMIT);
        assert!(rendered.contains('…'), "{rendered}");
    }
}
