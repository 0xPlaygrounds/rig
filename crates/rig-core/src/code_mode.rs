//! Runtime-independent code-mode execution with a safe local reference backend.
//!
//! # Security
//!
//! Allow-lists, trusted working directories, protected paths, output limits,
//! deadlines, and mutation queues are defense-in-depth host policy. They are not
//! an OS sandbox: do not allow shells or interpreters when untrusted arguments
//! could encode paths or commands. Unix cancellation creates and terminates a
//! process group; Windows uses `taskkill /T /F` when available. SSH/container
//! adapters inherit the local transport process controls, while isolation inside
//! the remote/container environment remains that environment's responsibility.

use crate::{
    runtime::RunContext,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(not(target_family = "wasm"))]
use std::{collections::HashMap, sync::Arc};
use std::{collections::HashSet, path::PathBuf, time::Duration};
#[cfg(not(target_family = "wasm"))]
use tokio::io::AsyncReadExt;

/// Explicit execution limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionLimits {
    pub timeout: Duration,
    pub output_bytes: usize,
}
/// Request to an execution backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeRequest {
    pub program: String,
    pub arguments: Vec<String>,
    pub working_directory: PathBuf,
    pub resource: String,
    pub mutates: bool,
    pub limits: ExecutionLimits,
}
/// Reference to truncated output retained by the backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactReference {
    pub path: PathBuf,
    pub original_bytes: usize,
}
/// Safe execution result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeResult {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
    pub artifact: Option<ArtifactReference>,
}
/// Backend error.
#[derive(Debug, thiserror::Error)]
pub enum CodeError {
    #[error("permission denied: {0}")]
    Permission(String),
    #[error("execution cancelled")]
    Cancelled,
    #[error("execution timed out")]
    Timeout,
    #[error("backend error: {0}")]
    Backend(String),
}
/// Progress observer.
pub trait CodeProgress: WasmCompatSend + WasmCompatSync {
    fn report(&self, message: &str);
}
/// Runtime/language-agnostic backend.
pub trait CodeBackend: WasmCompatSend + WasmCompatSync {
    fn execute<'a>(
        &'a self,
        request: CodeRequest,
        context: &'a RunContext,
        progress: &'a dyn CodeProgress,
    ) -> WasmBoxedFuture<'a, Result<CodeResult, CodeError>>;
}

struct SilentProgress;
impl CodeProgress for SilentProgress {
    fn report(&self, _: &str) {}
}

/// Installable code-mode tool that delegates to any safe backend without
/// changing the agent drive loop.
#[derive(Clone)]
pub struct CodeModeTool<B> {
    backend: B,
}
impl<B> CodeModeTool<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}
impl<B> crate::tool::Tool for CodeModeTool<B>
where
    B: CodeBackend + Clone + 'static,
{
    const NAME: &'static str = "code_mode";
    type Error = CodeError;
    type Args = CodeRequest;
    type Output = CodeResult;
    fn description(&self) -> String {
        "Execute an allowlisted command in the configured safe backend".into()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","required":["program","arguments","working_directory","resource","mutates","limits"]})
    }
    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(CodeError::Permission("RunContext is required".into()))
    }
    async fn call_with_extensions(
        &self,
        args: Self::Args,
        extensions: &crate::tool::ToolCallExtensions,
    ) -> Result<Self::Output, Self::Error> {
        let context = extensions
            .get::<RunContext>()
            .ok_or_else(|| CodeError::Permission("RunContext is required".into()))?;
        self.backend.execute(args, context, &SilentProgress).await
    }
}

/// Hide wrapped executable tools and expose only the code-mode tool definition.
pub fn code_mode_catalog(
    definition: crate::completion::ToolDefinition,
) -> Vec<crate::completion::ToolDefinition> {
    vec![definition]
}

/// Permissions and project trust applied before execution.
#[derive(Clone, Debug)]
pub struct CodePermissions {
    pub programs: HashSet<String>,
    pub trusted_root: PathBuf,
}

/// Hermetic local process backend with allowlists, process termination,
/// output limits, artifacts, and per-resource mutation serialization.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
pub struct LocalCommandBackend {
    permissions: CodePermissions,
    artifact_directory: PathBuf,
    protected_paths: Arc<[PathBuf]>,
    mutations: Arc<tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

#[cfg(not(target_family = "wasm"))]
impl LocalCommandBackend {
    pub fn new(permissions: CodePermissions, artifact_directory: PathBuf) -> Self {
        Self {
            permissions,
            artifact_directory,
            protected_paths: Arc::from([]),
            mutations: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Deny command arguments that resolve to these protected paths.
    pub fn protect_paths(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.protected_paths = paths.into_iter().collect::<Vec<_>>().into();
        self
    }
}

#[cfg(not(target_family = "wasm"))]
async fn read_bounded<R>(mut reader: R, limit: usize) -> Result<(Vec<u8>, usize), CodeError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut captured = Vec::with_capacity(limit.min(8192));
    let mut total = 0usize;
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader
            .read(&mut buffer)
            .await
            .map_err(|error| CodeError::Backend(error.to_string()))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read);
        let remaining = limit.saturating_sub(captured.len());
        if let Some(bytes) = buffer.get(..read.min(remaining)) {
            captured.extend_from_slice(bytes);
        }
    }
    Ok((captured, total))
}

#[cfg(not(target_family = "wasm"))]
async fn terminate_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = tokio::process::Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .status()
            .await;
        futures_timer::Delay::new(Duration::from_millis(25)).await;
        let _ = tokio::process::Command::new("kill")
            .args(["-KILL", &format!("-{pid}")])
            .status()
            .await;
    }
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        // `taskkill /T` is the broadly available Windows process-tree fallback.
        let _ = tokio::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .await;
    }
    let _ = child.kill().await;
}

#[cfg(not(target_family = "wasm"))]
impl CodeBackend for LocalCommandBackend {
    fn execute<'a>(
        &'a self,
        request: CodeRequest,
        context: &'a RunContext,
        progress: &'a dyn CodeProgress,
    ) -> WasmBoxedFuture<'a, Result<CodeResult, CodeError>> {
        Box::pin(async move {
            if !self.permissions.programs.contains(&request.program) {
                return Err(CodeError::Permission(format!(
                    "program `{}` is not allowed",
                    request.program
                )));
            }
            let root = self
                .permissions
                .trusted_root
                .canonicalize()
                .map_err(|e| CodeError::Permission(e.to_string()))?;
            let cwd = request
                .working_directory
                .canonicalize()
                .map_err(|e| CodeError::Permission(e.to_string()))?;
            if !cwd.starts_with(&root) {
                return Err(CodeError::Permission(
                    "working directory escapes trusted project".into(),
                ));
            }
            for argument in &request.arguments {
                let values = std::iter::once(argument.as_str())
                    .chain(argument.split_once('=').map(|(_, value)| value));
                for value in values {
                    let candidate = cwd.join(value);
                    let protected = self.protected_paths.iter().any(|protected| {
                        let protected = protected
                            .canonicalize()
                            .unwrap_or_else(|_| protected.clone());
                        candidate.canonicalize().map_or_else(
                            |_| candidate.starts_with(&protected),
                            |candidate| candidate.starts_with(&protected),
                        )
                    });
                    if protected {
                        return Err(CodeError::Permission(format!(
                            "argument targets protected path `{argument}`"
                        )));
                    }
                }
            }
            let artifact_parent = self
                .artifact_directory
                .parent()
                .and_then(|parent| parent.canonicalize().ok())
                .ok_or_else(|| {
                    CodeError::Permission("artifact directory has no trusted parent".into())
                })?;
            if !artifact_parent.starts_with(&root) {
                return Err(CodeError::Permission(
                    "artifact directory escapes trusted project".into(),
                ));
            }
            let resource_lock = if request.mutates {
                let mut locks = self.mutations.lock().await;
                Some(
                    locks
                        .entry(request.resource.clone())
                        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                        .clone(),
                )
            } else {
                None
            };
            let _mutation_guard = match resource_lock {
                Some(lock) => {
                    let acquire = lock.lock_owned();
                    let stopped = context.stopped();
                    futures::pin_mut!(acquire, stopped);
                    match futures::future::select(acquire, stopped).await {
                        futures::future::Either::Left((guard, _)) => {
                            if context.should_stop() {
                                return Err(CodeError::Cancelled);
                            }
                            Some(guard)
                        }
                        futures::future::Either::Right(_) => return Err(CodeError::Cancelled),
                    }
                }
                None => None,
            };
            progress.report("starting process");
            let mut command = tokio::process::Command::new(&request.program);
            command
                .args(&request.arguments)
                .current_dir(cwd)
                .kill_on_drop(true);
            // A dedicated process group lets cancellation terminate shells and
            // all descendants rather than only the immediate child.
            #[cfg(unix)]
            command.as_std_mut().process_group(0);
            let mut child = command
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| CodeError::Backend(e.to_string()))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| CodeError::Backend("stdout pipe missing".into()))?;
            let stderr = child
                .stderr
                .take()
                .ok_or_else(|| CodeError::Backend("stderr pipe missing".into()))?;
            let started = std::time::Instant::now();
            let wait = async {
                loop {
                    if context.should_stop() {
                        terminate_process_tree(&mut child).await;
                        return Err(CodeError::Cancelled);
                    }
                    if started.elapsed() >= request.limits.timeout {
                        terminate_process_tree(&mut child).await;
                        return Err(CodeError::Timeout);
                    }
                    if let Some(status) = child
                        .try_wait()
                        .map_err(|e| CodeError::Backend(e.to_string()))?
                    {
                        return Ok(status);
                    }
                    futures_timer::Delay::new(Duration::from_millis(5)).await;
                }
            };
            let (status, stdout, stderr) = futures::join!(
                wait,
                read_bounded(stdout, request.limits.output_bytes),
                read_bounded(stderr, request.limits.output_bytes),
            );
            let status = status?;
            let (stdout, stdout_bytes) = stdout?;
            let (stderr, stderr_bytes) = stderr?;
            let stdout_visible = stdout
                .get(..stdout.len().min(request.limits.output_bytes))
                .unwrap_or(&stdout);
            let stderr_budget = request
                .limits
                .output_bytes
                .saturating_sub(stdout_visible.len());
            let stderr_visible = stderr
                .get(..stderr.len().min(stderr_budget))
                .unwrap_or(&stderr);
            let mut combined = stdout_visible.to_vec();
            combined.extend_from_slice(stderr_visible);
            let original_bytes = stdout_bytes.saturating_add(stderr_bytes);
            let truncated = original_bytes > request.limits.output_bytes;
            let artifact = if truncated {
                tokio::fs::create_dir_all(&self.artifact_directory)
                    .await
                    .map_err(|e| CodeError::Backend(e.to_string()))?;
                let path = self
                    .artifact_directory
                    .join(format!("rig-code-{}.log", crate::id::generate()));
                tokio::fs::write(&path, &combined)
                    .await
                    .map_err(|e| CodeError::Backend(e.to_string()))?;
                Some(ArtifactReference {
                    path,
                    original_bytes,
                })
            } else {
                None
            };
            progress.report("process finished");
            Ok(CodeResult {
                status: status.code(),
                stdout: String::from_utf8_lossy(stdout_visible).into(),
                stderr: String::from_utf8_lossy(stderr_visible).into(),
                truncated,
                artifact,
            })
        })
    }
}

/// Concrete command-configured remote/container/sandbox adapter. The configured
/// transport executable is itself subject to the local backend's allow-list,
/// trusted working directory, cancellation, output, and mutation controls.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
pub struct CommandAdapterBackend {
    local: LocalCommandBackend,
    executable: String,
    prefix: Vec<String>,
}

#[cfg(not(target_family = "wasm"))]
impl CommandAdapterBackend {
    /// Configure an SSH adapter (`ssh <destination> -- <program> ...`).
    pub fn ssh(
        local: LocalCommandBackend,
        executable: impl Into<String>,
        destination: impl Into<String>,
    ) -> Self {
        Self {
            local,
            executable: executable.into(),
            prefix: vec![destination.into(), "--".into()],
        }
    }
    /// Configure a container adapter (`runtime exec <container> <program> ...`).
    pub fn container(
        local: LocalCommandBackend,
        runtime: impl Into<String>,
        container: impl Into<String>,
    ) -> Self {
        Self {
            local,
            executable: runtime.into(),
            prefix: vec!["exec".into(), container.into()],
        }
    }
    /// Configure a sandbox adapter whose prefix precedes the requested command.
    pub fn sandbox(
        local: LocalCommandBackend,
        executable: impl Into<String>,
        prefix: Vec<String>,
    ) -> Self {
        Self {
            local,
            executable: executable.into(),
            prefix,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl CodeBackend for CommandAdapterBackend {
    fn execute<'a>(
        &'a self,
        request: CodeRequest,
        context: &'a RunContext,
        progress: &'a dyn CodeProgress,
    ) -> WasmBoxedFuture<'a, Result<CodeResult, CodeError>> {
        if !self.local.permissions.programs.contains(&request.program) {
            return Box::pin(async move {
                Err(CodeError::Permission(format!(
                    "inner program `{}` is not allowed",
                    request.program
                )))
            });
        }
        let mut arguments = self.prefix.clone();
        arguments.push(request.program);
        arguments.extend(request.arguments);
        self.local.execute(
            CodeRequest {
                program: self.executable.clone(),
                arguments,
                ..request
            },
            context,
            progress,
        )
    }
}

/// Backend location abstraction for hosts installing local, SSH, container, or sandbox adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendLocation {
    Local,
    Ssh(String),
    Container(String),
    Sandbox(String),
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    struct Progress;
    impl CodeProgress for Progress {
        fn report(&self, _: &str) {}
    }

    #[tokio::test]
    async fn local_backend_enforces_allowlist_and_truncates_to_artifact() {
        let root = std::env::temp_dir().join(format!("rig-code-test-{}", crate::id::generate()));
        std::fs::create_dir_all(&root).unwrap();
        let backend = LocalCommandBackend::new(
            CodePermissions {
                programs: ["sh".into()].into_iter().collect(),
                trusted_root: root.clone(),
            },
            root.join("artifacts"),
        );
        let (_, context) = crate::runtime::RunControlHandle::new(None, None);
        let result = backend
            .execute(
                CodeRequest {
                    program: "sh".into(),
                    arguments: vec!["-c".into(), "printf 123456789".into()],
                    working_directory: root.clone(),
                    resource: "project".into(),
                    mutates: true,
                    limits: ExecutionLimits {
                        timeout: Duration::from_secs(2),
                        output_bytes: 4,
                    },
                },
                &context,
                &Progress,
            )
            .await
            .unwrap();
        assert_eq!(result.stdout, "1234");
        assert!(result.truncated);
        assert!(result.artifact.unwrap().path.is_file());
        let denied = backend
            .execute(
                CodeRequest {
                    program: "false-not-allowed".into(),
                    arguments: vec![],
                    working_directory: root.clone(),
                    resource: "project".into(),
                    mutates: false,
                    limits: ExecutionLimits {
                        timeout: Duration::from_secs(1),
                        output_bytes: 10,
                    },
                },
                &context,
                &Progress,
            )
            .await;
        assert!(matches!(denied, Err(CodeError::Permission(_))));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn queued_mutation_cancels_before_resource_lock_acquisition() {
        let root = std::env::temp_dir().join(format!("rig-code-cancel-{}", crate::id::generate()));
        std::fs::create_dir_all(&root).unwrap();
        let backend = LocalCommandBackend::new(
            CodePermissions {
                programs: ["sh".into()].into_iter().collect(),
                trusted_root: root.clone(),
            },
            root.join("artifacts"),
        );
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        backend
            .mutations
            .lock()
            .await
            .insert("same".into(), lock.clone());
        let guard = lock.lock_owned().await;
        let (control, context) = crate::runtime::RunControlHandle::new(None, None);
        let request = CodeRequest {
            program: "sh".into(),
            arguments: vec!["-c".into(), "printf should-not-run".into()],
            working_directory: root.clone(),
            resource: "same".into(),
            mutates: true,
            limits: ExecutionLimits {
                timeout: Duration::from_secs(2),
                output_bytes: 100,
            },
        };
        let execution = backend.execute(request, &context, &Progress);
        futures::pin_mut!(execution);
        assert!(futures::poll!(&mut execution).is_pending());
        control.cancel();
        assert!(matches!(execution.await, Err(CodeError::Cancelled)));
        drop(guard);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn same_resource_mutations_serialize_and_command_adapter_executes() {
        let root = std::env::temp_dir().join(format!("rig-code-queue-{}", crate::id::generate()));
        std::fs::create_dir_all(&root).unwrap();
        let backend = LocalCommandBackend::new(
            CodePermissions {
                programs: ["sh".into(), "env".into()].into_iter().collect(),
                trusted_root: root.clone(),
            },
            root.join("artifacts"),
        );
        let request = || CodeRequest {
            program: "sh".into(),
            arguments: vec!["-c".into(), "mkdir lock && sleep 0.05 && rmdir lock".into()],
            working_directory: root.clone(),
            resource: "same".into(),
            mutates: true,
            limits: ExecutionLimits {
                timeout: Duration::from_secs(2),
                output_bytes: 100,
            },
        };
        let (_, first_context) = crate::runtime::RunControlHandle::new(None, None);
        let (_, second_context) = crate::runtime::RunControlHandle::new(None, None);
        let (first, second) = futures::join!(
            backend.execute(request(), &first_context, &Progress),
            backend.execute(request(), &second_context, &Progress),
        );
        assert_eq!(first.unwrap().status, Some(0));
        assert_eq!(second.unwrap().status, Some(0));
        let adapter = CommandAdapterBackend::sandbox(backend, "env", vec![]);
        let result = adapter
            .execute(
                CodeRequest {
                    program: "sh".into(),
                    arguments: vec!["-c".into(), "printf adapter".into()],
                    working_directory: root.clone(),
                    resource: "read".into(),
                    mutates: false,
                    limits: ExecutionLimits {
                        timeout: Duration::from_secs(2),
                        output_bytes: 100,
                    },
                },
                &first_context,
                &Progress,
            )
            .await
            .unwrap();
        assert_eq!(result.stdout, "adapter");
        let denied = CommandAdapterBackend::sandbox(
            LocalCommandBackend::new(
                CodePermissions {
                    programs: ["env".into()].into_iter().collect(),
                    trusted_root: root.clone(),
                },
                root.join("artifacts"),
            ),
            "env",
            vec![],
        )
        .execute(
            CodeRequest {
                program: "sh".into(),
                arguments: vec![],
                working_directory: root.clone(),
                resource: "read".into(),
                mutates: false,
                limits: ExecutionLimits {
                    timeout: Duration::from_secs(1),
                    output_bytes: 100,
                },
            },
            &first_context,
            &Progress,
        )
        .await;
        assert!(matches!(denied, Err(CodeError::Permission(_))));
        let large = CommandAdapterBackend::sandbox(
            LocalCommandBackend::new(
                CodePermissions {
                    programs: ["env".into(), "sh".into()].into_iter().collect(),
                    trusted_root: root.clone(),
                },
                root.join("artifacts"),
            ),
            "env",
            vec![],
        )
        .execute(
            CodeRequest {
                program: "sh".into(),
                arguments: vec!["-c".into(), "yes x | head -c 200000".into()],
                working_directory: root.clone(),
                resource: "read".into(),
                mutates: false,
                limits: ExecutionLimits {
                    timeout: Duration::from_secs(3),
                    output_bytes: 1024,
                },
            },
            &first_context,
            &Progress,
        )
        .await
        .unwrap();
        assert_eq!(large.status, Some(0));
        assert!(large.truncated);
        assert_eq!(large.stdout.len(), 1024);
        std::fs::remove_dir_all(root).unwrap();
    }
}
