# rig shell companion (fish): `?? [question]` asks the project's rig session
# about the last command, `?? --cmd <what>` puts the command it suggests in
# the edit buffer, Alt-? asks why the last command failed. Run a command as
# `rig-run <command>` to send its output's tail too.
if set -q RIG_SHELL_STATE_DIR
    set -g __rig_state $RIG_SHELL_STATE_DIR
else if set -q XDG_RUNTIME_DIR
    set -g __rig_state $XDG_RUNTIME_DIR/rig-shell/$fish_pid
else
    set -g __rig_state $HOME/.cache/rig-shell/$fish_pid
end
mkdir -p $__rig_state; and chmod 700 $__rig_state
function __rig_record
    printf '%s\n' $argv[2] >$__rig_state/command
    printf '%s\n' $argv[1] >$__rig_state/exit
    pwd >$__rig_state/cwd
end
function __rig_postexec --on-event fish_postexec
    set -l code $status
    if not string match -qr '^(\?\?|rig-run|rig_ask|__rig|$)' -- $argv[1]
        __rig_record $code $argv[1]
        true >$__rig_state/output
    end
end
function rig-run
    $argv 2>&1 | tee $__rig_state/output.full
    set -l code $pipestatus[1]
    tail -n 200 $__rig_state/output.full >$__rig_state/output
    rm -f $__rig_state/output.full
    __rig_record $code "$argv"
    return $code
end
function rig_ask
    rm -f $__rig_state/suggested
    set -l rig rig
    set -q RIG_BIN; and set rig $RIG_BIN
    $rig shell-ask --state $__rig_state $argv
    set -l code $status
    if test -s $__rig_state/suggested
        commandline -r -- (string collect <$__rig_state/suggested)
    end
    return $code
end
abbr -a -- '??' rig_ask
bind \e? 'rig_ask; commandline -f repaint'
