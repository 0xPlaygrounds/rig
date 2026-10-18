//! What Linux's `/proc` says about processes: which carry a variable, their
//! parents, start times, CPU time, memory and listening ports, and the
//! signals that stop them. Elsewhere there is no `/proc`, so every scan is
//! empty and nothing is ever signalled.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::thread;
use std::time::{Duration, Instant};

use crate::process_reaper::Member;

/// How often a stop checks whether the processes are gone.
const POLL: Duration = Duration::from_millis(20);

/// How long a stop waits for killed processes to vanish.
const AFTER_KILL: Duration = Duration::from_secs(1);

/// A live process, as its `/proc/<pid>` entries say.
#[derive(Clone, Debug)]
pub(crate) struct Proc {
    /// Its pid and start time.
    pub member: Member,
    /// Its parent's pid.
    pub ppid: u32,
    /// Its command line, its arguments joined by spaces.
    pub command: String,
}

/// The fields of `/proc/<pid>/stat` after the command's name: its state,
/// parent, CPU ticks used and start time in ticks since boot.
struct Stat {
    state: char,
    ppid: u32,
    ticks: u64,
    start: u64,
}

fn stat(pid: u32) -> Option<Stat> {
    parse_stat(&fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// The fields of a `stat` file's `text`.
fn parse_stat(text: &str) -> Option<Stat> {
    // The name may hold spaces and parentheses; the fields follow its last `)`.
    let fields: Vec<&str> = text.rsplit_once(')')?.1.split_whitespace().collect();
    let number = |at: usize| fields.get(at).and_then(|field| field.parse::<u64>().ok());
    Some(Stat {
        state: fields.first()?.chars().next()?,
        ppid: u32::try_from(number(1)?).ok()?,
        ticks: number(11)? + number(12)?,
        start: number(19)?,
    })
}

/// Whether `member` still runs: its pid names a process started at its
/// start time, a thread of which is not a zombie.
pub(crate) fn alive(member: Member) -> bool {
    stat(member.pid).is_some_and(|stat| stat.start == member.start && runs(member.pid))
}

/// Whether a thread of process `pid` is not a zombie. A process whose
/// first thread ended shows as a zombie while its other threads, and the
/// files and ports they hold, live on.
pub(crate) fn runs(pid: u32) -> bool {
    let Ok(tasks) = fs::read_dir(format!("/proc/{pid}/task")) else {
        return false;
    };
    tasks.flatten().any(|task| {
        let text = fs::read_to_string(task.path().join("stat")).unwrap_or_default();
        parse_stat(&text).is_some_and(|stat| stat.state != 'Z')
    })
}

/// The process `pid` names now.
pub(crate) fn member(pid: u32) -> Option<Member> {
    stat(pid).map(|stat| Member {
        pid,
        start: stat.start,
    })
}

/// This process.
pub(crate) fn this_process() -> Member {
    let pid = std::process::id();
    member(pid).unwrap_or(Member { pid, start: 0 })
}

/// Every live process of this user whose environment holds `variable`
/// (`NAME=value`), oldest first.
pub(crate) fn scan(variable: &str) -> Vec<Proc> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    let wanted = variable.as_bytes();
    let mut found: Vec<Proc> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter_map(|pid| {
            let environ = fs::read(format!("/proc/{pid}/environ")).ok()?;
            if !environ.split(|byte| *byte == 0).any(|pair| pair == wanted) {
                return None;
            }
            let stat = stat(pid).filter(|_| runs(pid))?;
            let cmdline = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let words: Vec<String> = cmdline
                .split(|byte| *byte == 0)
                .filter(|word| !word.is_empty())
                .map(|word| String::from_utf8_lossy(word).into_owned())
                .collect();
            Some(Proc {
                member: Member {
                    pid,
                    start: stat.start,
                },
                ppid: stat.ppid,
                command: words.join(" "),
            })
        })
        .collect();
    found.sort_by_key(|proc| proc.member.start);
    found
}

/// What a group of processes uses, for a listing.
#[derive(Clone, Debug, Default)]
pub(crate) struct Usage {
    /// The TCP ports they listen on.
    pub ports: BTreeSet<u16>,
    /// The CPU time they used.
    pub cpu: Duration,
    /// Their resident memory, in KiB.
    pub rss_kib: u64,
    /// How long the oldest has run.
    pub age: Duration,
}

/// What the live ones of `members` use.
pub(crate) fn usage(members: &[Member]) -> Usage {
    let tick = clock_tick();
    let uptime = fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse::<f64>().ok())
        .unwrap_or_default();
    let listening = listening_sockets();
    let mut usage = Usage::default();
    let mut oldest = None::<u64>;
    for member in members.iter().copied().filter(|member| alive(*member)) {
        if let Some(stat) = stat(member.pid) {
            usage.cpu += tick.mul_f64(stat.ticks as f64);
            oldest = Some(oldest.map_or(stat.start, |at| at.min(stat.start)));
        }
        usage.rss_kib += rss_kib(member.pid);
        let fds = fs::read_dir(format!("/proc/{}/fd", member.pid))
            .into_iter()
            .flatten();
        for fd in fds.flatten() {
            let Ok(link) = fs::read_link(fd.path()) else {
                continue;
            };
            let inode = link.to_str().and_then(|link| link.strip_prefix("socket:["));
            let inode = inode.and_then(|inode| inode.strip_suffix(']')?.parse::<u64>().ok());
            if let Some(port) = inode.and_then(|inode| listening.get(&inode)) {
                usage.ports.insert(*port);
            }
        }
    }
    if let Some(start) = oldest {
        let started = tick.mul_f64(start as f64).as_secs_f64();
        usage.age = Duration::from_secs_f64((uptime - started).max(0.0));
    }
    usage
}

fn rss_kib(pid: u32) -> u64 {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:"));
    let kib = line.and_then(|line| line.split_whitespace().next()?.parse().ok());
    kib.unwrap_or_default()
}

/// The listening TCP sockets of this machine, by inode, with their ports.
fn listening_sockets() -> HashMap<u64, u16> {
    let mut sockets = HashMap::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let text = fs::read_to_string(table).unwrap_or_default();
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            // `0A` is LISTEN.
            if fields.get(3) != Some(&"0A") {
                continue;
            }
            let port = fields.get(1).and_then(|local| local.rsplit_once(':'));
            let port = port.and_then(|(_, port)| u16::from_str_radix(port, 16).ok());
            let inode = fields.get(9).and_then(|inode| inode.parse::<u64>().ok());
            if let (Some(port), Some(inode)) = (port, inode) {
                sockets.insert(inode, port);
            }
        }
    }
    sockets
}

/// The length of a clock tick, the unit of `/proc`'s times.
fn clock_tick() -> Duration {
    #[cfg(unix)]
    {
        // SAFETY: `sysconf` takes a constant and returns a number.
        let hertz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if let Ok(hertz) = u32::try_from(hertz)
            && hertz > 0
        {
            return Duration::from_secs(1) / hertz;
        }
    }
    Duration::from_millis(10)
}

/// Stops the live ones of `members`: asks each to stop (`SIGTERM`), waits
/// at most `grace` for them to end, then kills (`SIGKILL`) the rest. A
/// process is signalled only while its pid names the process started at
/// its start time. Returns how many were running. A blocking call.
pub(crate) fn stop(members: &[Member], grace: Duration) -> usize {
    let running: Vec<Member> = members.iter().copied().filter(|m| alive(*m)).collect();
    let mut seen = HashSet::new();
    for member in running.iter().filter(|member| seen.insert(member.pid)) {
        signal(*member, Signal::Terminate);
    }
    let left = |members: &[Member]| members.iter().copied().filter(|m| alive(*m)).collect();
    let deadline = Instant::now() + grace;
    let mut rest: Vec<Member> = left(&running);
    while !rest.is_empty() && Instant::now() < deadline {
        thread::sleep(POLL);
        rest = left(&rest);
    }
    for member in &rest {
        signal(*member, Signal::Kill);
    }
    let deadline = Instant::now() + AFTER_KILL;
    while !rest.is_empty() && Instant::now() < deadline {
        thread::sleep(POLL);
        rest = left(&rest);
    }
    running.len()
}

#[derive(Clone, Copy)]
enum Signal {
    Terminate,
    Kill,
}

/// Sends `signal` to `member` if it still runs.
fn signal(member: Member, signal: Signal) {
    if !alive(member) {
        return;
    }
    #[cfg(unix)]
    if let Ok(pid) = libc::pid_t::try_from(member.pid) {
        let signal = match signal {
            Signal::Terminate => libc::SIGTERM,
            Signal::Kill => libc::SIGKILL,
        };
        // SAFETY: `kill` takes plain integers; `pid` is positive.
        unsafe {
            libc::kill(pid, signal);
        }
    }
    #[cfg(not(unix))]
    let _ = signal;
}
