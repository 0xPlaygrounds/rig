//! The local transport: an owner-only Unix socket at `run/rig.sock` (0600,
//! in a 0700 directory) and the `run/token` every `rig/hello` carries.

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
