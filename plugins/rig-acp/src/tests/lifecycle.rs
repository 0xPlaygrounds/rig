//! What runs and what is left: a session process killed while its tool
//! runs leaves nothing and the others go on; the router's end, by EOF or a
//! signal, ends every session and its tools. Linux, which shows each
//! process's environment.

use super::*;

/// The live processes of `place`: the sessions, and the tools they started.
fn processes_of(place: &Place) -> Vec<u32> {
    let root = place.root.to_string_lossy().into_owned();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| {
            std::fs::read(format!("/proc/{pid}/environ"))
                .is_ok_and(|environ| String::from_utf8_lossy(&environ).contains(&root))
        })
        .filter(|pid| alive(*pid))
        .collect()
}

/// The process running `session`.
fn process_of(place: &Place, session: &acp::SessionId) -> Option<u32> {
    let mark = format!("RIG_SESSION={session}");
    processes_of(place).into_iter().find(|pid| {
        std::fs::read(format!("/proc/{pid}/environ")).is_ok_and(|environ| {
            environ
                .split(|byte| *byte == 0)
                .any(|entry| entry == mark.as_bytes())
        })
    })
}

/// Whether a tool's `sleep` of `place` runs.
fn sleeping(place: &Place) -> bool {
    processes_of(place).into_iter().any(|pid| {
        std::fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|line| String::from_utf8_lossy(&line).contains("sleep"))
    })
}

#[cfg(target_os = "linux")]
#[test]
fn a_killed_session_leaves_no_tool_and_the_others_go_on_until_the_client_leaves() {
    let place = Place::new("killed");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let (first, second) = (
            driven.new_session(Vec::new()).await,
            driven.new_session(Vec::new()).await,
        );
        assert!(first.is_some() && second.is_some(), "no sessions");
        let (Some(first), Some(second)) = (first, second) else {
            return;
        };
        let (first, second) = (first.session_id, second.session_id);
        assert!(driven.model(&first, "test/wait").await);
        let request = acp::PromptRequest::new(first.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|_| sleeping(place)).await);
        let pid = process_of(place, &first);
        assert!(pid.is_some(), "no process runs {first}");
        let Some(pid) = pid else { return };
        std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .ok();
        assert!(prompt.block_task().await.is_err());
        assert!(eventually(|| !sleeping(place)));
        assert!(driven.model(&second, "test/reply").await);
        assert_eq!(
            driven.prompt(&second, "Hello.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        let prompt = acp::PromptRequest::new(first.clone(), vec!["Hello.".into()]);
        assert!(driven.cx.send_request(prompt).block_task().await.is_err());
    });
    assert!(router.is_some(), "rig acp did not run");
    let Some(mut router) = router else { return };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let status = runtime
        .ok()
        .and_then(|runtime| runtime.block_on(router.process.wait()).ok());
    assert!(status.is_some_and(|status| status.success()));
    assert!(eventually(|| processes_of(place).is_empty()));
}

#[cfg(target_os = "linux")]
#[test]
fn sigterm_ends_the_router_its_sessions_and_their_tools() {
    let place = Place::new("sigterm");
    let place = &place;
    let mut pid = None;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|_| sleeping(place)).await);
        pid = driven.router;
        if let Some(pid) = pid {
            std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .ok();
        }
        // The router ends the session, which answers the prompt or closes.
        prompt.block_task().await.ok();
    });
    assert!(pid.is_some() && router.is_some());
    assert!(eventually(|| processes_of(place).is_empty()));
}
