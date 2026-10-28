use serde_json::json;

use super::*;

#[test]
fn servers_read_as_other_agents_write_them_and_the_rest_is_refused_by_name() {
    let path = std::env::var("PATH").unwrap_or_default();
    let http = |url: &str, headers: &[(&str, &str)]| {
        let headers = headers
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()));
        Ok(Some(ServerSpec::Http {
            url: url.to_owned(),
            headers: headers.collect(),
        }))
    };
    let cases = [
        (
            json!({ "command": "npx", "args": ["-y", "${RIG_MCP_TEST_UNSET_7E1D:-pkg}"], "env": { "T": "${PATH}" } }),
            Ok(Some(ServerSpec::Stdio {
                command: "npx".to_owned(),
                args: vec!["-y".to_owned(), "pkg".to_owned()],
                env: BTreeMap::from([("T".to_owned(), path.clone())]),
                cwd: None,
            })),
        ),
        (
            json!({ "type": "http", "url": "https://h/mcp", "headers": { "Authorization": "Bearer ${PATH}" } }),
            http("https://h/mcp", &[("Authorization", &format!("Bearer {path}"))]),
        ),
        (json!({ "type": "streamableHttp", "url": "https://h" }), http("https://h", &[])),
        (json!({ "url": "https://h" }), http("https://h", &[])),
        (json!({ "command": "x", "disabled": true }), Ok(None)),
        (
            json!({ "command": "x", "timeout": 5 }),
            Err("`timeout`: unknown field `timeout`, expected one of `type`, `command`, `args`, `env`, `cwd`, `url`, `headers`, `disabled`".to_owned()),
        ),
        (
            json!({ "type": "sse", "url": "https://h/sse" }),
            Err("`type` \"sse\" is not supported: rig speaks MCP over stdio and streamable HTTP (\"type\": \"http\")".to_owned()),
        ),
        (
            json!({ "url": "https://h", "command": "x" }),
            Err("`command` is a stdio server's; an http server has `url` and `headers`".to_owned()),
        ),
        (
            json!({ "command": "x", "env": { "PORT": 8080 } }),
            Err("`env.PORT`: invalid type: integer `8080`, expected a string".to_owned()),
        ),
        (
            json!({ "command": "x", "env": { "K": "${RIG_MCP_TEST_UNSET_7E1D}" } }),
            Err("`env.K`: ${RIG_MCP_TEST_UNSET_7E1D} is not set".to_owned()),
        ),
    ];
    for (written, parsed) in cases {
        assert_eq!(ServerSpec::parse(&written), parsed, "{written}");
    }
    // A name `__` is in would make two servers' tools alike.
    let names = [
        "fs",
        "my-files_2",
        "_x",
        "ops__audit",
        "ops_",
        "--user",
        "1password",
        "files.prod",
        "",
    ];
    let fine: Vec<bool> = names.iter().map(|name| check_name(name).is_ok()).collect();
    assert_eq!(
        fine,
        [true, true, true, false, false, false, false, false, false]
    );
    assert!(fits_providers("fs__read_file") && !fits_providers("fs__read.file"));
    assert!(!fits_providers(&format!("fs__{}", "x".repeat(61))));
}

/// An empty directory of the test `name`, beside the test binary.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let run = format!("rig-mcp-client-config-{}", std::process::id());
    let dir = exe.with_file_name(run).join(name);
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).ok();
    dir
}

#[test]
fn a_change_touches_only_the_bytes_of_its_server() {
    let dir = scratch("splice");
    let [user, _] = files(&dir, &dir);
    let before = "{\r\n\t\"other\": {\"a\": [1,2]},\r\n\t\"mcpServers\": {\r\n\t\t\"x\": {\"command\": \"x\"}\r\n\t}\r\n}\r\n";
    fs::write(user.path(), before).ok();
    assert_eq!(
        user.add("y", json!({ "command": "y", "args": ["-v"] })),
        Ok(())
    );
    let added = fs::read_to_string(user.path()).unwrap_or_default();
    assert_eq!(
        added,
        "{\r\n\t\"other\": {\"a\": [1,2]},\r\n\t\"mcpServers\": {\r\n\t\t\"x\": {\"command\": \"x\"},\r\n\t\t\"y\": {\r\n\t\t\t\"command\": \"y\",\r\n\t\t\t\"args\": [\r\n\t\t\t\t\"-v\"\r\n\t\t\t]\r\n\t\t}\r\n\t}\r\n}\r\n"
    );
    assert_eq!(user.remove("y"), Ok(true));
    assert_eq!(fs::read_to_string(user.path()).unwrap_or_default(), before);
    assert_eq!(user.remove("y"), Ok(false));
    // The first and the only server go as cleanly.
    assert_eq!(user.add("y", json!({ "url": "https://h" })), Ok(()));
    assert_eq!(user.remove("x"), Ok(true));
    assert_eq!(user.remove("y"), Ok(true));
    let emptied = fs::read_to_string(user.path()).unwrap_or_default();
    assert_eq!(
        emptied,
        "{\r\n\t\"other\": {\"a\": [1,2]},\r\n\t\"mcpServers\": {}\r\n}\r\n"
    );
    assert_eq!(user.add("x", json!({ "command": "x" })), Ok(()));
    let refilled = fs::read_to_string(user.path()).unwrap_or_default();
    assert_eq!(
        refilled,
        "{\r\n\t\"other\": {\"a\": [1,2]},\r\n\t\"mcpServers\": {\r\n\t\t\"x\": {\r\n\t\t\t\"command\": \"x\"\r\n\t\t}\r\n\t}\r\n}\r\n"
    );
    let again = user.add("x", json!({ "command": "x" }));
    assert!(again.is_err_and(|why| why.ends_with("names `x` already")));
    fs::remove_dir_all(dir).ok();
}

#[cfg(unix)]
#[test]
fn a_project_file_that_is_a_link_is_never_read_or_written() {
    let dir = scratch("link");
    let users = dir.join("users-mcp.json");
    let secret = "{\"mcpServers\":{\"gh\":{\"url\":\"https://h\",\"headers\":{\"Authorization\":\"Bearer SECRET\"}}}}\n";
    fs::write(&users, secret).ok();
    let [_, project] = files(&dir.join("home"), &dir);
    fs::create_dir_all(dir.join(".rig")).ok();
    // A checkout's link to the user's file.
    std::os::unix::fs::symlink(&users, project.path()).ok();
    let read = project.servers();
    assert!(read.is_err_and(|why| why.ends_with("is a link, so rig does not read it")));
    assert!(project.add("x", json!({ "command": "x" })).is_err());
    assert_eq!(fs::read_to_string(&users).unwrap_or_default(), secret);
    let metadata = fs::symlink_metadata(project.path());
    assert!(metadata.is_ok_and(|metadata| metadata.file_type().is_symlink()));
    let mut written = fs::read_dir(dir.join(".rig"))
        .into_iter()
        .flatten()
        .flatten();
    assert!(written.all(|entry| !entry.file_type().is_ok_and(|kind| kind.is_file())));
    fs::remove_dir_all(dir).ok();
}

#[cfg(unix)]
#[test]
fn the_users_file_keeps_its_permissions_and_its_link() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("users-link");
    let dotfiles = dir.join("dotfiles/mcp.json");
    fs::create_dir_all(dir.join("dotfiles")).ok();
    fs::create_dir_all(dir.join("home")).ok();
    std::os::unix::fs::symlink(&dotfiles, dir.join("home/mcp.json")).ok();
    let [user, _] = files(&dir.join("home"), &dir);
    // A link to a file not made yet is followed too.
    assert_eq!(user.add("first", json!({ "command": "first" })), Ok(()));
    assert!(
        fs::read_to_string(&dotfiles)
            .unwrap_or_default()
            .contains("\"first\"")
    );
    fs::set_permissions(&dotfiles, fs::Permissions::from_mode(0o600)).ok();
    assert_eq!(user.add("x", json!({ "command": "x" })), Ok(()));
    let link = fs::symlink_metadata(user.path());
    assert!(link.is_ok_and(|link| link.file_type().is_symlink()));
    assert!(
        fs::read_to_string(&dotfiles)
            .unwrap_or_default()
            .contains("\"x\"")
    );
    let mode = fs::metadata(&dotfiles).map(|metadata| metadata.permissions().mode() & 0o777);
    assert_eq!(mode.ok(), Some(0o600));
    fs::remove_dir_all(dir).ok();
}

#[cfg(unix)]
#[test]
fn a_project_write_never_leaves_the_project() {
    let dir = scratch("dir-link");
    let home = dir.join("home");
    fs::create_dir_all(&home).ok();
    fs::write(home.join("mcp.json"), "{}\n").ok();
    let repo = dir.join("repo");
    fs::create_dir_all(&repo).ok();
    std::os::unix::fs::symlink(&home, repo.join(".rig")).ok();
    let [_, project] = files(&home, &repo);
    assert!(project.servers().is_err());
    let refused = project.add("x", json!({ "command": "x" }));
    assert!(refused.is_err_and(|why| {
        why.ends_with("leads out of the project, so rig does not read or write it")
    }));
    assert_eq!(
        fs::read_to_string(home.join("mcp.json")).unwrap_or_default(),
        "{}\n"
    );
    fs::remove_dir_all(dir).ok();
}

#[cfg(unix)]
#[test]
fn a_write_keeps_the_files_group() {
    use std::os::unix::fs::MetadataExt;
    // Another group the user is in than the one a new file gets, if any.
    let dir = scratch("group");
    let probe = dir.join("probe");
    fs::write(&probe, "").ok();
    let own = fs::metadata(&probe)
        .map(|metadata| metadata.gid())
        .unwrap_or_default();
    let groups = std::process::Command::new("id").arg("-G").output();
    let groups = groups.map(|out| String::from_utf8_lossy(&out.stdout).into_owned());
    let other = groups
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|gid| gid.parse().ok())
        .find(|gid: &u32| *gid != own);
    let Some(other) = other else {
        fs::remove_dir_all(dir).ok();
        return;
    };
    let [user, _] = files(&dir, &dir);
    fs::write(user.path(), "{}\n").ok();
    let given = std::os::unix::fs::chown(user.path(), None, Some(other));
    assert!(given.is_ok(), "{given:?}");
    assert_eq!(user.add("x", json!({ "command": "x" })), Ok(()));
    let gid = fs::metadata(user.path()).map(|metadata| metadata.gid());
    assert_eq!(gid.ok(), Some(other));
    fs::remove_dir_all(dir).ok();
}
