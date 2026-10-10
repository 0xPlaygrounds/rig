use super::*;

#[cfg(unix)]
#[test]
fn only_a_session_with_a_conversation_is_offered_to_resume() {
    use std::os::unix::process::ExitStatusExt;

    let (status, session) = (ExitStatus::from_raw(2 << 8), SessionId::generate());
    let log = Path::new("/home/.rig/sessions/s/agent.log");
    let saved = stopped(status, &session, true, log);
    assert!(
        saved.contains(&format!("`rig --resume {session}`")),
        "{saved}"
    );
    let empty = stopped(status, &session, false, log);
    let expected = "The agent stopped (exit status: 2). Log: /home/.rig/sessions/s/agent.log";
    assert_eq!(empty, expected);
}
