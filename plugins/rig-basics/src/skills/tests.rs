use std::path::{Path, PathBuf};

use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness::harness_protocol::Home;
use rig_harness::services::{ProjectRoot, RigHome};
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, connect, log_ended, reply, run_until, scratch_home,
};

use super::{Skill, SkillsPlugin, discover, listing, message, roots};

/// Writes `SKILL.md` with `text` in `dir`.
fn skill(dir: &Path, text: &str) -> Option<()> {
    std::fs::create_dir_all(dir).ok()?;
    std::fs::write(dir.join("SKILL.md"), text).ok()
}

/// Two skill roots, the first winning, with a skill of each kind.
fn roots_with_skills(name: &str) -> Option<[PathBuf; 2]> {
    let scratch = scratch_home(name).root().to_owned();
    let (first, second) = (scratch.join("first"), scratch.join("second"));
    skill(
        &first.join("review"),
        "---\nname: review\ndescription: Review a diff.\n---\nRead `checklist.md`.\n",
    )?;
    skill(
        &second.join("review"),
        "---\nname: review\ndescription: Another review.\n---\n",
    )?;
    skill(
        &second.join("group/deploy"),
        "---\ndescription: Deploy <the> app & check.\nlicense: MIT\n---\nRun it.\n",
    )?;
    skill(
        &second.join("by-hand"),
        "---\ndescription: Only when asked.\ndisable-model-invocation: true\n---\n",
    )?;
    skill(&second.join("no-description"), "---\nname: nothing\n---\n")?;
    skill(
        &second.join("Odd_Name"),
        "---\ndescription: Oddly named.\n---\n",
    )?;
    skill(
        &second.join(".hidden/secret"),
        "---\ndescription: Hidden.\n---\n",
    )?;
    Some([first, second])
}

/// A skill folder whose links lead back up (`loop/{a,b} -> .`) is searched
/// once: the walk ends, finds the skill beside the links once, and names
/// the loop once.
#[cfg(unix)]
#[test]
fn a_link_back_up_is_searched_once_and_named_once() {
    let root = scratch_home("skills-loop").root().join("skills");
    let dir = root.join("loop");
    assert!(skill(&dir.join("inside"), "---\ndescription: Inside.\n---\n").is_some());
    for link in ["a", "b"] {
        assert!(std::os::unix::fs::symlink(".", dir.join(link)).is_ok());
    }
    let (skills, problems) = discover(&[root]);
    let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
    assert_eq!(names, ["inside"]);
    let looped: Vec<&String> = problems
        .iter()
        .filter(|problem| problem.contains("leads back"))
        .collect();
    assert_eq!(looped.len(), 1, "{problems:?}");
}

#[test]
fn discovery_takes_the_first_of_a_name_and_skips_skills_without_a_description() {
    let made = roots_with_skills("skills-discover");
    assert!(made.is_some(), "scratch skills");
    let Some(roots) = made else {
        return;
    };
    let (skills, problems) = discover(&roots);
    let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
    assert_eq!(names, ["review", "Odd_Name", "by-hand", "deploy"]);
    let first = skills.first().map(|skill| skill.description.as_str());
    assert_eq!(first, Some("Review a diff."));
    let said = problems.join("\n");
    for why in [
        "has the name `review` already",
        "no description",
        "`Odd_Name`",
    ] {
        assert!(said.contains(why), "{why} in {said}");
    }
    assert_eq!(problems.len(), 3, "{said}");
}

#[test]
fn the_listing_leaves_out_skills_only_sent_by_hand() {
    let made = roots_with_skills("skills-listing");
    assert!(made.is_some(), "scratch skills");
    let Some(roots) = made else {
        return;
    };
    let (skills, _) = discover(&roots);
    let text = listing(&skills);
    assert!(text.contains("<name>deploy</name>"), "{text}");
    assert!(
        text.contains("Deploy &lt;the&gt; app &amp; check."),
        "{text}"
    );
    assert!(!text.contains("by-hand"), "{text}");
    assert!(listing(&[]).is_empty());
    let manual: Vec<Skill> = skills.into_iter().filter(|skill| skill.manual).collect();
    assert!(listing(&manual).is_empty());
}

#[test]
fn project_roots_come_first_from_the_working_directory_up_to_the_repository() {
    let scratch = scratch_home("skills-roots").root().to_owned();
    let (repository, home) = (scratch.join("repo"), scratch.join("home"));
    let dir = repository.join("sub");
    assert!(std::fs::create_dir_all(repository.join(".git")).is_ok());
    let found = roots(&dir, &home);
    let expected = [
        dir.join(".rig/skills"),
        dir.join(".agents/skills"),
        repository.join(".rig/skills"),
        repository.join(".agents/skills"),
        home.join("skills"),
    ];
    assert_eq!(found.get(..5), Some(&expected[..]));
    assert!(!found.contains(&scratch.join(".agents/skills")));
}

/// The prompt lists the project's skill, and `/skill:<name>` sends its
/// body, without the frontmatter, with the user's text.
#[test]
fn a_skill_is_listed_and_sent_with_its_command() {
    let scratch = scratch_home("skills-command").root().to_owned();
    let dir = scratch.join("repo");
    assert!(std::fs::create_dir_all(dir.join(".git")).is_ok());
    let body = "---\nname: haiku\ndescription: Answer in a haiku.\n---\nWrite three lines.\n";
    assert!(skill(&dir.join(".agents/skills/haiku"), body).is_some());
    let model = MockCompletionModel::from_stream_turns([reply("1")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .insert_resource(RigHome(Home::at(scratch.join("home"))))
        .add_plugins((AgentPlugin, SessionLogPlugin, SkillsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let agent = app.world_mut().spawn((Agent, connection)).id();
    let line = Deliver::new(agent, "/skill:haiku about rust", DeliveryMode::Steer);
    app.world_mut().trigger(line);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let requests = model.requests();
    let request = requests.first();
    let prompt = request.and_then(|request| request.system_instructions());
    let prompt = prompt.unwrap_or_default();
    assert!(prompt.contains("<name>haiku</name>"), "{prompt}");
    let sent = request.map(|request| format!("{:?}", request.chat_history));
    let sent = sent.unwrap_or_default();
    for part in ["Write three lines.", "about rust", "<skill name="] {
        assert!(sent.contains(part), "{part} in {sent}");
    }
    assert!(!sent.contains("description:"), "{sent}");
}

#[test]
fn the_message_says_where_the_skill_is() {
    let dir = scratch_home("skills-message").root().join("tidy");
    assert!(skill(&dir, "Tidy up.\n").is_some());
    let tidy = Skill {
        name: "tidy".to_owned(),
        description: "Tidy.".to_owned(),
        file: dir.join("SKILL.md"),
        manual: false,
    };
    let text = message(&tidy, "").unwrap_or_default();
    assert!(
        text.contains(&format!("relative to {}", dir.display())),
        "{text}"
    );
    assert!(text.ends_with("Tidy up.\n</skill>"), "{text}");
}
