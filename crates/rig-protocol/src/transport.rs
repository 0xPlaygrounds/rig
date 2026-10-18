//! The local transport: an owner-only Unix socket at `run/rig.sock` (0600,
//! in a 0700 directory), or on Windows an owner-only named pipe derived
//! from that path, and the `run/token` every `rig/hello` carries.

use std::path::{Path, PathBuf};

/// Where the local server listens: `run/rig.sock` under the rig home.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub path: PathBuf,
}

impl Endpoint {
    pub fn for_home(rig_home: &Path) -> Endpoint {
        Endpoint {
            path: rig_home.join("run").join("rig.sock"),
        }
    }
}

/// The owner token in `run/token` (0600), without surrounding whitespace.
pub fn read_token(rig_home: &Path) -> std::io::Result<String> {
    let token = std::fs::read_to_string(rig_home.join("run").join("token"))?;
    let token = token.trim();
    if token.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "run/token is empty",
        ));
    }
    Ok(token.to_owned())
}

#[cfg(unix)]
pub use unix::{Conn, Listener, Peer};
#[cfg(windows)]
pub use windows::{Conn, Listener, Peer};

/// 64-bit FNV-1a: a stable hash (unlike `std`'s), so every rig of this
/// user and home derives the same pipe name.
#[cfg_attr(not(windows), allow(dead_code))]
fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.iter().chain(&[0u8]) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(unix)]
mod unix {
    use std::io::{BufRead, BufReader, ErrorKind, Write};
    use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};

    use super::Endpoint;

    /// A bound owner socket.
    pub struct Listener(UnixListener);
    /// One connection, either end.
    pub struct Conn(UnixStream);
    /// Who connected: whether its uid is ours, and its pid where known.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Peer {
        pub uid_ok: bool,
        pub pid: Option<u32>,
    }

    fn uid() -> u32 {
        // SAFETY: getuid has no preconditions and cannot fail.
        unsafe { libc::getuid() }
    }

    impl Listener {
        /// Binds the endpoint. Makes its directory 0700 when missing and
        /// refuses one that isn't a directory of ours with mode 0700. A
        /// socket nobody answers on is replaced; a live one is `AddrInUse`.
        /// The caller holds `run/server.lock`.
        pub fn bind(e: &Endpoint) -> std::io::Result<Listener> {
            let dir = e.path.parent().ok_or_else(|| {
                std::io::Error::new(ErrorKind::InvalidInput, "the socket path has no directory")
            })?;
            match std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)
            {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
            let meta = std::fs::symlink_metadata(dir)?;
            if !meta.is_dir() || meta.uid() != uid() || meta.mode() & 0o777 != 0o700 {
                return Err(std::io::Error::new(
                    ErrorKind::PermissionDenied,
                    format!(
                        "{} must be a directory of ours with mode 0700",
                        dir.display()
                    ),
                ));
            }
            match std::fs::symlink_metadata(&e.path) {
                Ok(meta) if meta.file_type().is_socket() && meta.uid() == uid() => {
                    if UnixStream::connect(&e.path).is_ok() {
                        return Err(ErrorKind::AddrInUse.into());
                    }
                    std::fs::remove_file(&e.path)?;
                }
                Ok(_) => {
                    return Err(std::io::Error::new(
                        ErrorKind::AlreadyExists,
                        format!("{} is not a socket of ours", e.path.display()),
                    ));
                }
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
            let listener = UnixListener::bind(&e.path)?;
            std::fs::set_permissions(&e.path, std::fs::Permissions::from_mode(0o600))?;
            Ok(Listener(listener))
        }

        /// The next connection and who made it.
        pub fn accept(&self) -> std::io::Result<(Conn, Peer)> {
            let (stream, _) = self.0.accept()?;
            let peer = peer(&stream);
            Ok((Conn(stream), peer))
        }
    }

    impl Conn {
        pub fn connect(e: &Endpoint) -> std::io::Result<Conn> {
            UnixStream::connect(&e.path).map(Conn)
        }

        /// A second handle on the same connection, such as one to
        /// [`shutdown`](Conn::shutdown) it while another thread reads.
        pub fn try_clone(&self) -> std::io::Result<Conn> {
            self.0.try_clone().map(Conn)
        }

        /// Shuts both directions down: a blocked read on any handle returns.
        pub fn shutdown(&self) -> std::io::Result<()> {
            self.0.shutdown(std::net::Shutdown::Both)
        }

        /// The reading and writing halves, for a thread each.
        pub fn split(
            self,
        ) -> std::io::Result<(impl BufRead + Send + use<>, impl Write + Send + use<>)> {
            let writer = self.0.try_clone()?;
            Ok((BufReader::new(self.0), writer))
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn peer(stream: &UnixStream) -> Peer {
        use std::os::fd::AsRawFd;
        let mut cred = libc::ucred {
            pid: 0,
            uid: 0,
            gid: 0,
        };
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: `cred` and `len` are valid for writes of a `ucred`, and
        // the descriptor is open for the call.
        let ok = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&raw mut cred).cast(),
                &mut len,
            )
        } == 0;
        Peer {
            uid_ok: ok && cred.uid == uid(),
            pid: ok.then(|| u32::try_from(cred.pid).ok()).flatten(),
        }
    }

    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    ))]
    fn peer(stream: &UnixStream) -> Peer {
        use std::os::fd::AsRawFd;
        let mut peer_uid: libc::uid_t = 0;
        let mut peer_gid: libc::gid_t = 0;
        // SAFETY: both out pointers are valid, and the descriptor is open.
        let ok = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut peer_uid, &mut peer_gid) } == 0;
        Peer {
            uid_ok: ok && peer_uid == uid(),
            pid: None,
        }
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    )))]
    fn peer(_stream: &UnixStream) -> Peer {
        Peer {
            uid_ok: false,
            pid: None,
        }
    }
}

#[cfg(windows)]
mod windows {
    //! `\\.\pipe\rig-<hash(user SID, socket path)>`, created with a DACL
    //! granting only this user's SID, `FILE_FLAG_FIRST_PIPE_INSTANCE` (a
    //! second server can't take the name) and `PIPE_REJECT_REMOTE_CLIENTS`.

    use std::io::{BufRead, BufReader, ErrorKind, Write};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

    use interprocess::TryClone;
    use interprocess::os::windows::ToWtf16;
    use interprocess::os::windows::named_pipe::{
        DuplexPipeStream, PipeListener, PipeListenerOptions, PipeMode, pipe_mode,
    };
    use interprocess::os::windows::security_descriptor::SecurityDescriptor;
    use windows_sys::Win32::Foundation::{ERROR_ACCESS_DENIED, ERROR_PIPE_BUSY, HANDLE, LocalFree};
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;
    use windows_sys::Win32::Security::{GetTokenInformation, TOKEN_QUERY, TOKEN_USER, TokenUser};
    use windows_sys::Win32::System::IO::CancelIoEx;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    use super::{Endpoint, fnv1a};

    type Stream = DuplexPipeStream<pipe_mode::Bytes>;

    /// A bound owner pipe.
    pub struct Listener(PipeListener<pipe_mode::Bytes, pipe_mode::Bytes>);
    /// One connection, either end.
    pub struct Conn(Stream);
    /// Who connected. Only this user's SID may open the pipe (its DACL)
    /// and remote clients are rejected, so `uid_ok` holds for every peer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Peer {
        pub uid_ok: bool,
        pub pid: Option<u32>,
    }

    /// This process's user SID as a string, such as `S-1-5-21-…-1001`.
    fn user_sid() -> std::io::Result<String> {
        let mut token: HANDLE = std::ptr::null_mut();
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no
        // closing; `token` is a valid out pointer for one HANDLE.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: OpenProcessToken succeeded, so `token` is an open handle
        // we own; OwnedHandle closes it once.
        let token = unsafe { OwnedHandle::from_raw_handle(token) };
        let mut len = 0u32;
        // SAFETY: a null buffer of length 0 asks only for the size, written
        // to the valid `len`.
        unsafe {
            GetTokenInformation(
                token.as_raw_handle(),
                TokenUser,
                std::ptr::null_mut(),
                0,
                &mut len,
            );
        }
        // u64 words keep the buffer aligned for TOKEN_USER (pointer-aligned).
        let words = (len as usize).div_ceil(8).max(1);
        let mut buf = vec![0u64; words];
        // SAFETY: `buf` is writable for `words * 8 >= len` bytes and `len`
        // is a valid out pointer.
        let ok = unsafe {
            GetTokenInformation(
                token.as_raw_handle(),
                TokenUser,
                buf.as_mut_ptr().cast(),
                u32::try_from(words * 8).unwrap_or(u32::MAX),
                &mut len,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: on success the buffer starts with a TOKEN_USER, aligned
        // (u64 buffer) and initialized by the call; its Sid points into
        // `buf`, which outlives this use.
        let sid = unsafe { (*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid };
        let mut text: *mut u16 = std::ptr::null_mut();
        // SAFETY: `sid` is the valid SID above; `text` is a valid out pointer.
        if unsafe { ConvertSidToStringSidW(sid, &mut text) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: on success `text` is a NUL-terminated wide string the
        // system allocated with LocalAlloc; we read up to its NUL, then free
        // it once with LocalFree.
        let sid = unsafe {
            let mut n = 0;
            while *text.add(n) != 0 {
                n += 1;
            }
            let sid = String::from_utf16_lossy(std::slice::from_raw_parts(text, n));
            LocalFree(text.cast());
            sid
        };
        Ok(sid)
    }

    fn pipe_name(e: &Endpoint, sid: &str) -> String {
        let path = e.path.to_string_lossy();
        format!(
            r"\\.\pipe\rig-{:016x}",
            fnv1a(&[sid.as_bytes(), path.as_bytes()])
        )
    }

    impl Listener {
        /// Creates the endpoint's pipe, owner-only. A live server holding
        /// the name is `AddrInUse`. A pipe disappears with its last handle,
        /// so there is never a stale one to replace.
        pub fn bind(e: &Endpoint) -> std::io::Result<Listener> {
            let sid = user_sid()?;
            // Protected DACL (no inherited entries) with one entry: generic
            // all for this user. Nobody else, SYSTEM and Administrators
            // included, is granted access.
            let sddl = format!("D:P(A;;GA;;;{sid})");
            let sddl = sddl
                .as_str()
                .to_wtf_16()
                .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "a NUL in the SID"))?;
            let descriptor = SecurityDescriptor::deserialize(&sddl)?;
            let name = pipe_name(e, &sid);
            let mut options = PipeListenerOptions::new().path(name.as_str());
            options.mode = PipeMode::Bytes;
            options.accept_remote = false;
            options.security_descriptor = Some(descriptor);
            options.inheritable = false;
            options.input_buffer_size_hint = 64 * 1024;
            options.output_buffer_size_hint = 64 * 1024;
            // interprocess creates the first instance with
            // FILE_FLAG_FIRST_PIPE_INSTANCE: it fails when another process
            // holds the name.
            match options.create_duplex::<pipe_mode::Bytes>() {
                Ok(listener) => Ok(Listener(listener)),
                Err(error) if error.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) => {
                    Err(ErrorKind::AddrInUse.into())
                }
                Err(error) => Err(error),
            }
        }

        /// The next connection and who made it.
        pub fn accept(&self) -> std::io::Result<(Conn, Peer)> {
            let stream = self.0.accept()?;
            let pid = stream.client_process_id().ok();
            Ok((Conn(stream), Peer { uid_ok: true, pid }))
        }
    }

    impl Conn {
        /// Connects to the endpoint's pipe. No server is `NotFound`; all
        /// instances busy (a server about to accept) is retried briefly.
        pub fn connect(e: &Endpoint) -> std::io::Result<Conn> {
            let name = pipe_name(e, &user_sid()?);
            let mut tries = 0;
            loop {
                match Stream::connect_by_path(name.as_str()) {
                    Ok(stream) => return Ok(Conn(stream)),
                    Err(error)
                        if error.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) && tries < 50 =>
                    {
                        tries += 1;
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        /// A second handle on the same connection.
        pub fn try_clone(&self) -> std::io::Result<Conn> {
            self.0.try_clone().map(Conn)
        }

        /// Cancels every read and write in flight on the connection, from
        /// any thread, so a blocked reader returns.
        pub fn shutdown(&self) -> std::io::Result<()> {
            // SAFETY: the handle is open for as long as `self.0` lives; a
            // null OVERLAPPED cancels all of this process's I/O on it.
            if unsafe { CancelIoEx(self.0.as_raw_handle(), std::ptr::null()) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        }

        /// The reading and writing halves, for a thread each.
        pub fn split(
            self,
        ) -> std::io::Result<(impl BufRead + Send + use<>, impl Write + Send + use<>)> {
            let (recv, send) = self.0.split();
            Ok((BufReader::new(recv), send))
        }
    }
}
