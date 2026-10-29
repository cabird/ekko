//! Platform detection and filesystem discovery.
//!
//! Platform/WSL detection, the ekko data directory, the export-path sandbox,
//! and per-platform session-path candidates for Claude Code, VS Code Copilot,
//! GitHub Copilot CLI and OpenAI Codex.

use std::collections::VecDeque;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Name of the per-platform data directory.
pub const DATA_DIR_NAME: &str = "ekko";

const EXCLUDED_WINDOWS_USERS: [&str; 4] = ["Public", "Default", "Default User", "All Users"];

/// Detected platform. WSL is distinguished from plain Linux because the
/// session-path candidates differ (Windows-side cross-mounts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    Wsl,
    Linux,
    Macos,
}

impl Platform {
    pub fn as_str(self) -> &'static str {
        match self {
            Platform::Windows => "windows",
            Platform::Wsl => "wsl",
            Platform::Linux => "linux",
            Platform::Macos => "macos",
        }
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors from data-dir / session-path discovery.
#[derive(Debug, Error)]
pub enum PathsError {
    #[error("could not determine home directory")]
    NoHomeDir,
    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Detect the current platform: windows / wsl / linux / macos.
///
/// WSL is detected from the kernel release string in
/// `/proc/sys/kernel/osrelease` (the same string `uname -r` prints), which
/// contains "microsoft" or "wsl" there and nowhere else.
pub fn get_platform() -> Platform {
    #[cfg(target_os = "windows")]
    {
        Platform::Windows
    }
    #[cfg(target_os = "macos")]
    {
        Platform::Macos
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let release = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
        if is_wsl_release(&release) {
            Platform::Wsl
        } else {
            Platform::Linux
        }
    }
}

/// WSL check on the kernel release string. The production caller is the
/// Linux branch of `get_platform`, cfg'd out elsewhere; the unit test runs
/// on every platform.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), allow(dead_code))]
fn is_wsl_release(release: &str) -> bool {
    let release = release.to_lowercase();
    release.contains("microsoft") || release.contains("wsl")
}

/// Get the Windows username when running in WSL.
///
/// Order: `WIN_HOME` env var (its last path component), then a scan of
/// `/mnt/c/Users` — a single non-system user wins, otherwise a
/// case-insensitive match against `$USER`.
pub fn get_windows_username() -> Option<String> {
    windows_username_impl(
        env::var("WIN_HOME").ok().filter(|s| !s.is_empty()),
        Path::new("/mnt/c/Users"),
        &env::var("USER").unwrap_or_default(),
    )
}

fn windows_username_impl(
    win_home: Option<String>,
    users_root: &Path,
    linux_user: &str,
) -> Option<String> {
    if let Some(win_home) = win_home {
        return Path::new(&win_home)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
    }

    let entries = fs::read_dir(users_root).ok()?;
    let mut users: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name();
        if EXCLUDED_WINDOWS_USERS.iter().any(|excluded| name == *excluded) {
            continue;
        }
        users.push(name.to_string_lossy().into_owned());
    }

    if users.len() == 1 {
        return users.pop();
    }
    users
        .into_iter()
        .find(|u| u.to_lowercase() == linux_user.to_lowercase())
}

/// Discover WSL home directories accessible from Windows via `\\wsl$`.
///
/// Shells out to `wsl -l -q` (UTF-16LE output on Windows) and scans each
/// distro's `/home`. Only meaningful on Windows; returns empty when `wsl`
/// is missing or fails.
pub fn get_wsl_home_dirs() -> Vec<PathBuf> {
    let stdout = match run_wsl_list(Duration::from_secs(5)) {
        Some(bytes) => bytes,
        None => return Vec::new(),
    };
    let text = decode_utf16le_ignore_errors(&stdout);

    let mut homes = Vec::new();
    for distro in text.lines().map(str::trim).filter(|s| !s.is_empty()) {
        let wsl_home = PathBuf::from(format!(r"\\wsl$\{distro}\home"));
        let Ok(entries) = fs::read_dir(&wsl_home) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                homes.push(entry.path());
            }
        }
    }
    homes
}

/// Run `wsl -l -q` with a hard timeout, returning its stdout bytes, or `None`
/// on spawn failure or timeout.
///
/// `Command::output` has no timeout, so poll `try_wait` every ~50ms and
/// kill the child at the deadline. The output of `wsl -l -q` is tiny, so
/// reading stdout only after exit cannot deadlock on a full pipe.
fn run_wsl_list(timeout: Duration) -> Option<Vec<u8>> {
    let mut child = Command::new("wsl")
        .args(["-l", "-q"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }

    let mut buf = Vec::new();
    child.stdout.take()?.read_to_end(&mut buf).ok()?;
    Some(buf)
}

/// Decode UTF-16LE, dropping invalid units instead of erroring.
fn decode_utf16le_ignore_errors(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    char::decode_utf16(units)
        .filter_map(Result::ok)
        .collect()
}

fn env_path_nonempty(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Pure data-dir construction, testable without touching the real machine.
fn compute_data_dir(
    platform: Platform,
    home: &Path,
    localappdata: Option<PathBuf>,
    xdg_data_home: Option<PathBuf>,
) -> PathBuf {
    let base = match platform {
        Platform::Windows => {
            localappdata.unwrap_or_else(|| home.join("AppData").join("Local"))
        }
        Platform::Macos => home.join("Library").join("Application Support"),
        Platform::Linux | Platform::Wsl => {
            xdg_data_home.unwrap_or_else(|| home.join(".local").join("share"))
        }
    };
    base.join(DATA_DIR_NAME)
}

/// Get (and create) the standard ekko data directory.
///
/// - Windows: `%LOCALAPPDATA%\ekko`
/// - macOS: `~/Library/Application Support/ekko`
/// - Linux/WSL: `$XDG_DATA_HOME/ekko` or `~/.local/share/ekko`
pub fn get_data_dir() -> Result<PathBuf, PathsError> {
    let home = dirs::home_dir().ok_or(PathsError::NoHomeDir)?;
    let dir = compute_data_dir(
        get_platform(),
        &home,
        env_path_nonempty("LOCALAPPDATA"),
        env_path_nonempty("XDG_DATA_HOME"),
    );
    fs::create_dir_all(&dir).map_err(|source| PathsError::CreateDir {
        path: dir.clone(),
        source,
    })?;
    Ok(dir)
}

/// Path to the sessions database (`<data_dir>/sessions.db`).
pub fn get_database_path() -> Result<PathBuf, PathsError> {
    Ok(get_data_dir()?.join("sessions.db"))
}

/// Directory for user-initiated file exports; created if missing.
///
/// All file-write tools sandbox their writes under this directory.
pub fn get_exports_dir() -> Result<PathBuf, PathsError> {
    let dir = get_data_dir()?.join("exports");
    fs::create_dir_all(&dir).map_err(|source| PathsError::CreateDir {
        path: dir.clone(),
        source,
    })?;
    Ok(dir)
}

/// Raised when a requested export path escapes the exports sandbox.
///
/// `'{path}'` in the messages is a plain single-quote wrap of the (lossy)
/// string, not a fully escaped repr — the text is MCP-visible, so keep it
/// readable.
#[derive(Debug, Error)]
pub enum ExportPathError {
    #[error("output_path is required")]
    Missing,
    /// A null byte cannot appear in a filesystem path, so it is rejected up
    /// front rather than left to fail during resolution.
    #[error("output_path contains an embedded null byte")]
    EmbeddedNul,
    #[error(
        "output_path must be relative to the exports directory; got absolute path '{path}'. \
         Pass a relative name like 'my-export.txt' — it will be saved under {exports}."
    )]
    Absolute { path: String, exports: PathBuf },
    #[error(
        "output_path '{path}' escapes the exports sandbox ({exports}). \
         Pass a relative name without '..' segments."
    )]
    Escapes { path: String, exports: PathBuf },
    #[error("exports directory unavailable: {source}")]
    ExportsDir {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Symlink loop or I/O failure while resolving the candidate path.
    #[error("failed to resolve output_path '{path}'")]
    Resolve {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create export parent directory {path}: {source}")]
    CreateParent {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Resolve a user-supplied export path safely under the exports sandbox.
///
/// Rules:
/// - empty input is rejected;
/// - absolute paths and Windows drive paths (second character `:`) are
///   rejected on every platform;
/// - the candidate is resolved (following symlinks for components that
///   exist) and must remain inside the canonicalized exports dir — this
///   defeats `..` traversal and symlink escape via pre-existing symlinks.
///
/// Returns the absolute path to write to; parent directories are created.
pub fn resolve_export_path(user_path: &str) -> Result<PathBuf, ExportPathError> {
    let exports = get_exports_dir().map_err(|e| ExportPathError::ExportsDir {
        source: Box::new(e),
    })?;
    resolve_export_path_in(&exports, user_path)
}

/// Testable core of [`resolve_export_path`] against an explicit sandbox dir.
pub fn resolve_export_path_in(
    exports_dir: &Path,
    user_path: &str,
) -> Result<PathBuf, ExportPathError> {
    if user_path.is_empty() {
        return Err(ExportPathError::Missing);
    }
    if user_path.contains('\0') {
        return Err(ExportPathError::EmbeddedNul);
    }

    let p = Path::new(user_path);
    // Reject anything absolute, plus a drive-like second character `:` on
    // every platform. `has_root` additionally catches Windows drive-relative
    // paths like `\foo`, which the sandbox check would reject anyway.
    let drive_like = user_path.chars().nth(1) == Some(':');
    if p.has_root() || drive_like {
        return Err(ExportPathError::Absolute {
            path: user_path.to_string(),
            exports: exports_dir.to_path_buf(),
        });
    }

    let exports_canonical = fs::canonicalize(exports_dir).map_err(|e| {
        // Wrap with the path for context; keep the io::Error as the source.
        ExportPathError::ExportsDir {
            source: Box::new(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", exports_dir.display()),
            )),
        }
    })?;
    let candidate =
        resolve_nonstrict(&exports_canonical, p).map_err(|source| ExportPathError::Resolve {
            path: user_path.to_string(),
            source,
        })?;

    if !candidate.starts_with(&exports_canonical) {
        return Err(ExportPathError::Escapes {
            path: user_path.to_string(),
            exports: exports_canonical,
        });
    }

    if let Some(parent) = candidate.parent() {
        fs::create_dir_all(parent).map_err(|source| ExportPathError::CreateParent {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    Ok(candidate)
}

/// Maximum symlink traversals during resolution, like the Linux kernel's
/// limit (40); hitting it means a symlink loop.
const MAX_SYMLINK_HOPS: usize = 40;

/// One pending path component during non-strict resolution.
enum PendingComponent {
    /// Filesystem root (with drive prefix on Windows) from an absolute
    /// symlink target: resets the accumulated path.
    Root(PathBuf),
    Parent,
    Normal(OsString),
}

/// Prepend `path`'s components to the front of the work queue, preserving
/// their order. `CurDir` is dropped; a leading prefix/root collapses into a
/// single [`PendingComponent::Root`].
fn queue_components(path: &Path, queue: &mut VecDeque<PendingComponent>) {
    let mut comps: Vec<PendingComponent> = Vec::new();
    let mut root = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => comps.push(PendingComponent::Parent),
            Component::Normal(name) => comps.push(PendingComponent::Normal(name.to_os_string())),
            Component::Prefix(_) | Component::RootDir => root.push(component.as_os_str()),
        }
    }
    if !root.as_os_str().is_empty() {
        comps.insert(0, PendingComponent::Root(root));
    }
    for comp in comps.into_iter().rev() {
        queue.push_front(comp);
    }
}

/// Non-strict resolution of `base / rel`, i.e. resolution that tolerates a
/// path that does not fully exist: symlinks are followed for components that
/// exist,
/// dangling symlinks are resolved via `read_link` (their target is spliced
/// into the remaining work, so a link to a nonexistent path still resolves
/// to that target — this is what keeps dangling links from escaping the
/// export sandbox); `..` is applied to the (already symlink-free) resolved
/// prefix; verifiably nonexistent tails are appended lexically. More than
/// [`MAX_SYMLINK_HOPS`] link traversals is treated as a symlink loop and
/// returns an error. `base` must already be canonical.
fn resolve_nonstrict(base: &Path, rel: &Path) -> std::io::Result<PathBuf> {
    let mut cur = base.to_path_buf();
    let mut queue: VecDeque<PendingComponent> = VecDeque::new();
    queue_components(rel, &mut queue);

    let mut hops = 0usize;
    while let Some(component) = queue.pop_front() {
        match component {
            PendingComponent::Root(root) => cur = root,
            PendingComponent::Parent => {
                cur.pop();
            }
            PendingComponent::Normal(name) => {
                cur.push(&name);
                if let Ok(resolved) = fs::canonicalize(&cur) {
                    cur = resolved;
                    continue;
                }
                // canonicalize needs the target to exist; fall back to
                // symlink_metadata to distinguish dangling/cyclic symlinks
                // from plain nonexistent tails.
                match fs::symlink_metadata(&cur) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        hops += 1;
                        if hops > MAX_SYMLINK_HOPS {
                            return Err(std::io::Error::other(format!(
                                "symlink loop while resolving {}",
                                cur.display()
                            )));
                        }
                        let target = fs::read_link(&cur)?;
                        cur.pop();
                        // Absolute target replaces the prefix (via Root);
                        // relative target resolves against the parent.
                        queue_components(&target, &mut queue);
                    }
                    // Exists and is not a symlink: keep the lexical path.
                    Ok(_) => {}
                    // Verifiably nonexistent (not a symlink): lexical tail.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    // Cannot verify whether this is a symlink — refuse to
                    // guess rather than risk a sandbox escape.
                    Err(e) => return Err(e),
                }
            }
        }
    }
    Ok(cur)
}

/// Candidate session paths per CLI source. Field order is the discovery order
/// (`claude_code`, `vscode_copilot`, `github_copilot`, `codex`) and is pinned
/// by tests, because it decides which parser sees a path first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionPaths {
    pub claude_code: Vec<PathBuf>,
    pub vscode_copilot: Vec<PathBuf>,
    pub github_copilot: Vec<PathBuf>,
    pub codex: Vec<PathBuf>,
}

impl SessionPaths {
    /// Iterate `(cli_source, paths)` pairs in the pinned field order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &[PathBuf])> {
        [
            ("claude_code", self.claude_code.as_slice()),
            ("vscode_copilot", self.vscode_copilot.as_slice()),
            ("github_copilot", self.github_copilot.as_slice()),
            ("codex", self.codex.as_slice()),
        ]
        .into_iter()
    }
}

/// Codex keeps live threads under `sessions/` (dated subdirectories) and
/// archived ones under `archived_sessions/`; both are indexed.
const CODEX_SESSION_DIRS: [&str; 2] = ["sessions", "archived_sessions"];

/// The default Codex state directory, `~/.codex`.
pub fn default_codex_home() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".codex"))
}

fn push_codex_dirs(paths: &mut SessionPaths, codex_home: &Path) {
    for dir in CODEX_SESSION_DIRS {
        paths.codex.push(codex_home.join(dir));
    }
}

/// Pure session-path construction, testable with injected inputs.
/// `codex_home` is `$CODEX_HOME` when set: Codex's own override, which some
/// launchers point at a private copy (Orca does on macOS), so it is indexed
/// *in addition to* `~/.codex` — a person's threads may be in either.
fn compute_session_paths(
    platform: Platform,
    home: &Path,
    appdata: Option<PathBuf>,
    win_user: Option<&str>,
    wsl_homes: &[PathBuf],
    codex_home: Option<PathBuf>,
) -> SessionPaths {
    let mut paths = SessionPaths::default();

    // Codex is the same on every platform: `~/.codex`, `$CODEX_HOME` when it
    // points elsewhere, plus (below) the cross-mounted homes the other CLIs
    // also get.
    let default_codex = home.join(".codex");
    push_codex_dirs(&mut paths, &default_codex);
    if let Some(custom) = codex_home
        && custom != default_codex
    {
        push_codex_dirs(&mut paths, &custom);
    }

    match platform {
        Platform::Windows => {
            paths.claude_code.push(home.join(".claude").join("projects"));
            paths
                .github_copilot
                .push(home.join(".copilot").join("session-state"));

            let appdata =
                appdata.unwrap_or_else(|| home.join("AppData").join("Roaming"));
            paths.vscode_copilot.push(
                appdata
                    .join("Code")
                    .join("User")
                    .join("workspaceStorage"),
            );

            for wsl_home in wsl_homes {
                paths
                    .claude_code
                    .push(wsl_home.join(".claude").join("projects"));
                paths
                    .github_copilot
                    .push(wsl_home.join(".copilot").join("session-state"));
                push_codex_dirs(&mut paths, &wsl_home.join(".codex"));
            }
        }
        Platform::Wsl => {
            // Linux-side paths (if tools run in WSL)
            paths.claude_code.push(home.join(".claude").join("projects"));
            paths
                .github_copilot
                .push(home.join(".copilot").join("session-state"));

            // Windows-side paths via /mnt/c (more common — tools run on Windows)
            if let Some(win_user) = win_user {
                let win_home = PathBuf::from(format!("/mnt/c/Users/{win_user}"));
                paths
                    .claude_code
                    .push(win_home.join(".claude").join("projects"));
                paths
                    .github_copilot
                    .push(win_home.join(".copilot").join("session-state"));
                paths.vscode_copilot.push(
                    win_home
                        .join("AppData")
                        .join("Roaming")
                        .join("Code")
                        .join("User")
                        .join("workspaceStorage"),
                );
                push_codex_dirs(&mut paths, &win_home.join(".codex"));
            }
        }
        Platform::Macos => {
            paths.claude_code.push(home.join(".claude").join("projects"));
            paths
                .github_copilot
                .push(home.join(".copilot").join("session-state"));
            paths.vscode_copilot.push(
                home.join("Library")
                    .join("Application Support")
                    .join("Code")
                    .join("User")
                    .join("workspaceStorage"),
            );
        }
        Platform::Linux => {
            paths.claude_code.push(home.join(".claude").join("projects"));
            paths
                .github_copilot
                .push(home.join(".copilot").join("session-state"));
            paths.vscode_copilot.push(
                home.join(".config")
                    .join("Code")
                    .join("User")
                    .join("workspaceStorage"),
            );
        }
    }

    paths
}

/// Get candidate paths to search for AI coding sessions on this machine.
pub fn get_session_paths() -> Result<SessionPaths, PathsError> {
    let platform = get_platform();
    let home = dirs::home_dir().ok_or(PathsError::NoHomeDir)?;
    let appdata = env_path_nonempty("APPDATA");
    let win_user = if platform == Platform::Wsl {
        get_windows_username()
    } else {
        None
    };
    let wsl_homes = if platform == Platform::Windows {
        get_wsl_home_dirs()
    } else {
        Vec::new()
    };
    Ok(compute_session_paths(
        platform,
        &home,
        appdata,
        win_user.as_deref(),
        &wsl_homes,
        env_path_nonempty("CODEX_HOME"),
    ))
}

/// Apply config.json `session_paths` overrides: a source listed there
/// replaces that source's default candidates wholesale; sources not listed
/// keep their defaults. Unknown source names are ignored (there is no
/// parser to consume them).
pub fn apply_session_path_overrides(
    all: &mut SessionPaths,
    overrides: &[(String, Vec<PathBuf>)],
) {
    for (source, paths) in overrides {
        match source.as_str() {
            "claude_code" => all.claude_code = paths.clone(),
            "vscode_copilot" => all.vscode_copilot = paths.clone(),
            "github_copilot" => all.github_copilot = paths.clone(),
            "codex" => all.codex = paths.clone(),
            _ => {}
        }
    }
}

/// [`get_session_paths`] with the loaded config's `session_paths` overrides
/// applied. Discovery (scanning, `ekko check`, status) must resolve through
/// here so config.json's documented override actually takes effect.
pub fn get_configured_session_paths() -> Result<SessionPaths, PathsError> {
    let mut all = get_session_paths()?;
    if let Some(overrides) = &crate::config::get_config().session_paths {
        apply_session_path_overrides(&mut all, overrides);
    }
    Ok(all)
}

/// Shared helpers for tests that read or mutate process environment
/// variables: a crate-wide lock (env access is process-global, so tests in
/// different modules must serialize) and an RAII restore guard.
#[cfg(test)]
pub(crate) mod testenv {
    use std::env;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Serialize every test that reads or mutates process env vars.
    pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// RAII guard restoring an env var on drop.
    pub(crate) struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        pub(crate) fn set(key: &'static str, value: &str) -> Self {
            let previous = env::var_os(key);
            // SAFETY: all env-mutating tests hold ENV_LOCK.
            unsafe { env::set_var(key, value) };
            EnvVarGuard { key, previous }
        }

        pub(crate) fn unset(key: &'static str) -> Self {
            let previous = env::var_os(key);
            // SAFETY: all env-mutating tests hold ENV_LOCK.
            unsafe { env::remove_var(key) };
            EnvVarGuard { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: all env-mutating tests hold ENV_LOCK.
            unsafe {
                match &self.previous {
                    Some(v) => env::set_var(self.key, v),
                    None => env::remove_var(self.key),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testenv::{env_lock, EnvVarGuard};
    use super::*;
    use tempfile::TempDir;

    // ---- platform detection --------------------------------------------

    #[test]
    fn wsl_release_detection() {
        assert!(is_wsl_release("6.6.87.2-microsoft-standard-WSL2"));
        assert!(is_wsl_release("4.4.0-19041-Microsoft"));
        assert!(is_wsl_release("5.15.0-wsl2"));
        assert!(!is_wsl_release("6.8.0-45-generic"));
        assert!(!is_wsl_release(""));
    }

    #[test]
    fn utf16le_decoding_ignores_bad_units() {
        // "Ubuntu\r\n" in UTF-16LE
        let bytes: Vec<u8> = "Ubuntu\r\n".encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert_eq!(decode_utf16le_ignore_errors(&bytes), "Ubuntu\r\n");
        // Trailing odd byte is dropped, not an error
        let mut odd = bytes.clone();
        odd.push(0x41);
        assert_eq!(decode_utf16le_ignore_errors(&odd), "Ubuntu\r\n");
        // Lone surrogate is dropped
        let lone = [0x00u8, 0xD8, 0x41, 0x00]; // U+D800, 'A'
        assert_eq!(decode_utf16le_ignore_errors(&lone), "A");
    }

    // ---- data dir naming ------------------------------------------------

    #[test]
    fn data_dir_linux_default_and_xdg() {
        let home = Path::new("/home/alice");
        assert_eq!(
            compute_data_dir(Platform::Linux, home, None, None),
            PathBuf::from("/home/alice/.local/share/ekko")
        );
        assert_eq!(
            compute_data_dir(Platform::Wsl, home, None, Some(PathBuf::from("/xdg/data"))),
            PathBuf::from("/xdg/data/ekko")
        );
    }

    #[test]
    fn data_dir_windows_and_macos() {
        let home = Path::new("/home/alice");
        assert_eq!(
            compute_data_dir(
                Platform::Windows,
                Path::new(r"C:\Users\alice"),
                Some(PathBuf::from(r"C:\Users\alice\AppData\Local")),
                None,
            ),
            PathBuf::from(r"C:\Users\alice\AppData\Local").join("ekko")
        );
        assert_eq!(
            compute_data_dir(Platform::Windows, Path::new("/h"), None, None),
            Path::new("/h").join("AppData").join("Local").join("ekko")
        );
        assert_eq!(
            compute_data_dir(Platform::Macos, home, None, None),
            PathBuf::from("/home/alice/Library/Application Support/ekko")
        );
    }

    #[test]
    fn data_dir_is_named_ekko() {
        for plat in [Platform::Windows, Platform::Wsl, Platform::Linux, Platform::Macos] {
            let dir = compute_data_dir(plat, Path::new("/h"), None, None);
            assert_eq!(dir.file_name().unwrap(), "ekko");
        }
    }

    // ---- windows username discovery ------------------------------------

    #[test]
    fn win_home_env_takes_precedence() {
        let out = windows_username_impl(
            Some("/mnt/c/Users/Carol".into()),
            Path::new("/nonexistent"),
            "someone",
        );
        assert_eq!(out.as_deref(), Some("Carol"));
    }

    #[test]
    fn single_user_dir_wins() {
        let tmp = TempDir::new().unwrap();
        for d in ["Public", "Default", "Default User", "All Users", "Carol"] {
            fs::create_dir(tmp.path().join(d)).unwrap();
        }
        let out = windows_username_impl(None, tmp.path(), "unrelated");
        assert_eq!(out.as_deref(), Some("Carol"));
    }

    #[test]
    fn multiple_users_matched_case_insensitively_against_linux_user() {
        let tmp = TempDir::new().unwrap();
        for d in ["Carol", "Bob"] {
            fs::create_dir(tmp.path().join(d)).unwrap();
        }
        assert_eq!(
            windows_username_impl(None, tmp.path(), "carol").as_deref(),
            Some("Carol")
        );
        assert_eq!(windows_username_impl(None, tmp.path(), "mallory"), None);
    }

    #[test]
    fn missing_users_dir_returns_none() {
        assert_eq!(windows_username_impl(None, Path::new("/no/such/dir"), "x"), None);
    }

    // ---- export sandbox ------------------------------------------------

    fn sandbox() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let exports = tmp.path().join("exports");
        fs::create_dir(&exports).unwrap();
        (tmp, exports)
    }

    #[test]
    fn relative_name_resolves_under_exports() {
        let (_tmp, exports) = sandbox();
        let out = resolve_export_path_in(&exports, "hello.txt").unwrap();
        assert_eq!(out.parent().unwrap(), fs::canonicalize(&exports).unwrap());
        assert_eq!(out.file_name().unwrap(), "hello.txt");
    }

    #[test]
    fn relative_subdir_resolves_under_exports() {
        let (_tmp, exports) = sandbox();
        let out = resolve_export_path_in(&exports, "sub/dir/hello.txt").unwrap();
        assert!(out.parent().unwrap().exists()); // mkdir -p
        assert!(out.starts_with(fs::canonicalize(&exports).unwrap()));
    }

    #[test]
    fn empty_path_rejected() {
        let (_tmp, exports) = sandbox();
        assert!(matches!(
            resolve_export_path_in(&exports, ""),
            Err(ExportPathError::Missing)
        ));
    }

    #[test]
    fn absolute_posix_path_rejected() {
        let (_tmp, exports) = sandbox();
        let err = resolve_export_path_in(&exports, "/tmp/escape.txt").unwrap_err();
        assert!(matches!(err, ExportPathError::Absolute { .. }));
        assert!(err.to_string().to_lowercase().contains("absolute"));
    }

    #[test]
    fn windows_drive_path_rejected_on_all_platforms() {
        // The colon-at-index-1 check runs on every platform, so 'C:\\...' and
        // 'a:b' are both rejected even on Unix.
        let (_tmp, exports) = sandbox();
        for bad in ["C:\\Windows\\System32\\evil.txt", "C:/evil.txt", "a:b"] {
            assert!(
                matches!(
                    resolve_export_path_in(&exports, bad),
                    Err(ExportPathError::Absolute { .. })
                ),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn dotdot_traversal_rejected() {
        let (_tmp, exports) = sandbox();
        let err = resolve_export_path_in(&exports, "../escape.txt").unwrap_err();
        assert!(matches!(err, ExportPathError::Escapes { .. }));
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("escape") || msg.contains("sandbox"));
    }

    #[test]
    fn deep_traversal_rejected() {
        let (_tmp, exports) = sandbox();
        assert!(matches!(
            resolve_export_path_in(&exports, "a/b/../../../etc/passwd"),
            Err(ExportPathError::Escapes { .. })
        ));
    }

    #[test]
    fn internal_dotdot_that_stays_inside_is_allowed() {
        // 'sub/../hello.txt' normalizes to '<exports>/hello.txt', which is
        // inside the sandbox, so it is allowed.
        let (_tmp, exports) = sandbox();
        let out = resolve_export_path_in(&exports, "sub/../hello.txt").unwrap();
        assert_eq!(out, fs::canonicalize(&exports).unwrap().join("hello.txt"));
    }

    #[test]
    fn dot_resolves_to_exports_dir_itself() {
        // '.' resolves to the exports directory itself.
        let (_tmp, exports) = sandbox();
        let out = resolve_export_path_in(&exports, ".").unwrap();
        assert_eq!(out, fs::canonicalize(&exports).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_rejected() {
        let (tmp, exports) = sandbox();
        let outside = tmp.path().join("outside");
        fs::create_dir(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, exports.join("evil_link")).unwrap();

        assert!(matches!(
            resolve_export_path_in(&exports, "evil_link/file.txt"),
            Err(ExportPathError::Escapes { .. })
        ));
        // Even with a trailing '..' hop back through the link.
        assert!(matches!(
            resolve_export_path_in(&exports, "evil_link/../../whatever.txt"),
            Err(ExportPathError::Escapes { .. })
        ));
    }

    #[test]
    fn writes_actually_land_in_sandbox() {
        let (_tmp, exports) = sandbox();
        let out = resolve_export_path_in(&exports, "real-write.txt").unwrap();
        fs::write(&out, "hi").unwrap();
        assert_eq!(fs::read_to_string(&out).unwrap(), "hi");
    }

    #[test]
    fn nul_byte_rejected() {
        // A NUL anywhere in the path is rejected up front, whatever follows it.
        let (_tmp, exports) = sandbox();
        for bad in ["foo\0bar", "\0", "sub/\0.txt"] {
            assert!(
                matches!(
                    resolve_export_path_in(&exports, bad),
                    Err(ExportPathError::EmbeddedNul)
                ),
                "expected NUL rejection for {bad:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_outside_rejected() {
        // exports/dangling -> <tmp>/outside/pwned.txt with the target absent.
        // The link is followed via readlink, the containment check rejects the
        // target, and the write must never happen.
        let (tmp, exports) = sandbox();
        let target = tmp.path().join("outside").join("pwned.txt");
        std::os::unix::fs::symlink(&target, exports.join("dangling")).unwrap();

        assert!(matches!(
            resolve_export_path_in(&exports, "dangling"),
            Err(ExportPathError::Escapes { .. })
        ));
        assert!(!target.exists());
    }

    #[cfg(unix)]
    #[test]
    fn relative_dangling_symlink_outside_rejected() {
        // A relative dangling target ('../outside2/f.txt') resolves against the
        // link's parent → outside the sandbox.
        let (_tmp, exports) = sandbox();
        std::os::unix::fs::symlink("../outside2/f.txt", exports.join("reldangle")).unwrap();

        assert!(matches!(
            resolve_export_path_in(&exports, "reldangle"),
            Err(ExportPathError::Escapes { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_inside_exports_resolves_to_target() {
        // exports/din resolves to exports/sub/target.txt even though neither
        // the link target nor its parent exists yet.
        let (_tmp, exports) = sandbox();
        let target = exports.join("sub").join("target.txt");
        std::os::unix::fs::symlink(&target, exports.join("din")).unwrap();

        let out = resolve_export_path_in(&exports, "din").unwrap();
        assert_eq!(
            out,
            fs::canonicalize(&exports).unwrap().join("sub").join("target.txt")
        );
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_then_dotdot_resolves_link_first() {
        // 'dangling/../x' first resolves the link, THEN applies '..' →
        // <tmp>/outside/x (outside the sandbox → rejected). A lexical collapse
        // of '..' would wrongly allow it.
        let (tmp, exports) = sandbox();
        let target = tmp.path().join("outside").join("pwned.txt");
        std::os::unix::fs::symlink(&target, exports.join("dangling")).unwrap();

        assert!(matches!(
            resolve_export_path_in(&exports, "dangling/../x"),
            Err(ExportPathError::Escapes { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_cycle_rejected() {
        // A symlink cycle makes the request error out rather than resolve, for
        // both 'a' and 'a/file.txt'.
        let (_tmp, exports) = sandbox();
        std::os::unix::fs::symlink(exports.join("b"), exports.join("a")).unwrap();
        std::os::unix::fs::symlink(exports.join("a"), exports.join("b")).unwrap();

        assert!(matches!(
            resolve_export_path_in(&exports, "a"),
            Err(ExportPathError::Resolve { .. })
        ));
        assert!(matches!(
            resolve_export_path_in(&exports, "a/file.txt"),
            Err(ExportPathError::Resolve { .. })
        ));
    }

    #[test]
    fn prefix_similar_sibling_not_contained() {
        // starts_with is component-wise, so '/data/exports-evil' is NOT
        // under '/data/exports' (a naive string prefix check would pass).
        assert!(!Path::new("/data/exports-evil").starts_with("/data/exports"));

        // Behavioral pin: a symlink to an existing sibling whose name shares
        // the exports prefix is still an escape.
        #[cfg(unix)]
        {
            let (tmp, exports) = sandbox();
            let evil = tmp.path().join(format!(
                "{}-evil",
                exports.file_name().unwrap().to_str().unwrap()
            ));
            fs::create_dir(&evil).unwrap();
            std::os::unix::fs::symlink(&evil, exports.join("sib")).unwrap();
            assert!(matches!(
                resolve_export_path_in(&exports, "sib/file.txt"),
                Err(ExportPathError::Escapes { .. })
            ));
        }
    }

    // ---- env helpers and end-to-end wiring -------------------------------

    #[test]
    fn env_path_nonempty_empty_is_none() {
        let _lock = env_lock();
        {
            let _g = EnvVarGuard::set("EKKO_TEST_PATHVAR", "");
            assert_eq!(env_path_nonempty("EKKO_TEST_PATHVAR"), None);
        }
        {
            let _g = EnvVarGuard::unset("EKKO_TEST_PATHVAR");
            assert_eq!(env_path_nonempty("EKKO_TEST_PATHVAR"), None);
        }
        {
            let _g = EnvVarGuard::set("EKKO_TEST_PATHVAR", "/some/where");
            assert_eq!(
                env_path_nonempty("EKKO_TEST_PATHVAR"),
                Some(PathBuf::from("/some/where"))
            );
        }
    }

    /// End-to-end: get_data_dir → get_exports_dir → resolve_export_path,
    /// with the data dir redirected into a tempdir via XDG_DATA_HOME
    /// (Linux/WSL honor it; gated off macOS/Windows where it is unused).
    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn end_to_end_data_dir_to_export_resolution() {
        let _lock = env_lock();
        let tmp = TempDir::new().unwrap();
        let _g = EnvVarGuard::set("XDG_DATA_HOME", tmp.path().to_str().unwrap());

        let data_dir = get_data_dir().unwrap();
        assert_eq!(data_dir, tmp.path().join("ekko"));
        assert!(data_dir.is_dir());

        let exports = get_exports_dir().unwrap();
        assert_eq!(exports, data_dir.join("exports"));
        assert!(exports.is_dir());

        let out = resolve_export_path("e2e/report.txt").unwrap();
        assert!(out.starts_with(fs::canonicalize(&exports).unwrap()));
        fs::write(&out, "e2e").unwrap();
        assert_eq!(fs::read_to_string(&out).unwrap(), "e2e");

        assert!(matches!(
            resolve_export_path("../escape.txt"),
            Err(ExportPathError::Escapes { .. })
        ));
    }

    // ---- session path candidates ----------------------------------------

    #[test]
    fn linux_session_paths() {
        let home = Path::new("/home/alice");
        let paths = compute_session_paths(Platform::Linux, home, None, None, &[], None);
        assert_eq!(
            paths.claude_code,
            vec![PathBuf::from("/home/alice/.claude/projects")]
        );
        assert_eq!(
            paths.github_copilot,
            vec![PathBuf::from("/home/alice/.copilot/session-state")]
        );
        assert_eq!(
            paths.vscode_copilot,
            vec![PathBuf::from("/home/alice/.config/Code/User/workspaceStorage")]
        );
    }

    #[test]
    fn wsl_session_paths_with_windows_user() {
        let home = Path::new("/home/alice");
        let paths = compute_session_paths(Platform::Wsl, home, None, Some("Alice"), &[], None);
        assert_eq!(
            paths.claude_code,
            vec![
                PathBuf::from("/home/alice/.claude/projects"),
                PathBuf::from("/mnt/c/Users/Alice/.claude/projects"),
            ]
        );
        assert_eq!(
            paths.github_copilot,
            vec![
                PathBuf::from("/home/alice/.copilot/session-state"),
                PathBuf::from("/mnt/c/Users/Alice/.copilot/session-state"),
            ]
        );
        assert_eq!(
            paths.vscode_copilot,
            vec![PathBuf::from(
                "/mnt/c/Users/Alice/AppData/Roaming/Code/User/workspaceStorage"
            )]
        );
    }

    #[test]
    fn wsl_session_paths_without_windows_user() {
        let home = Path::new("/home/alice");
        let paths = compute_session_paths(Platform::Wsl, home, None, None, &[], None);
        assert_eq!(paths.claude_code.len(), 1);
        assert_eq!(paths.github_copilot.len(), 1);
        assert!(paths.vscode_copilot.is_empty());
    }

    #[test]
    fn macos_session_paths() {
        let home = Path::new("/Users/alice");
        let paths = compute_session_paths(Platform::Macos, home, None, None, &[], None);
        assert_eq!(
            paths.vscode_copilot,
            vec![PathBuf::from(
                "/Users/alice/Library/Application Support/Code/User/workspaceStorage"
            )]
        );
    }

    #[test]
    fn windows_session_paths_include_wsl_homes() {
        let home = Path::new(r"C:\Users\alice");
        let wsl_homes = vec![PathBuf::from(r"\\wsl$\Ubuntu\home\alice")];
        let paths = compute_session_paths(
            Platform::Windows,
            home,
            Some(PathBuf::from(r"C:\Users\alice\AppData\Roaming")),
            None,
            &wsl_homes,
            None,
        );
        assert_eq!(paths.claude_code.len(), 2);
        assert_eq!(paths.github_copilot.len(), 2);
        // Codex: own home (sessions + archived) plus the WSL home's pair.
        assert_eq!(paths.codex.len(), 4);
        assert_eq!(
            paths.codex[2],
            PathBuf::from(r"\\wsl$\Ubuntu\home\alice").join(".codex").join("sessions")
        );
        assert_eq!(
            paths.claude_code[1],
            PathBuf::from(r"\\wsl$\Ubuntu\home\alice").join(".claude").join("projects")
        );
        assert_eq!(
            paths.vscode_copilot,
            vec![PathBuf::from(r"C:\Users\alice\AppData\Roaming")
                .join("Code")
                .join("User")
                .join("workspaceStorage")]
        );
    }

    #[test]
    fn session_paths_iter_order_is_pinned() {
        let paths = compute_session_paths(Platform::Linux, Path::new("/h"), None, None, &[], None);
        let keys: Vec<&str> = paths.iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec!["claude_code", "vscode_copilot", "github_copilot", "codex"]
        );
    }

    #[test]
    fn codex_paths_default_home_on_every_platform() {
        for platform in [Platform::Linux, Platform::Macos, Platform::Windows, Platform::Wsl] {
            let home = Path::new("/home/alice");
            let paths = compute_session_paths(platform, home, None, None, &[], None);
            assert_eq!(
                &paths.codex[..2],
                &[
                    PathBuf::from("/home/alice/.codex/sessions"),
                    PathBuf::from("/home/alice/.codex/archived_sessions"),
                ],
                "{platform}"
            );
        }
    }

    #[test]
    fn config_session_path_overrides_replace_per_source() {
        let mut all = compute_session_paths(
            Platform::Linux,
            Path::new("/home/alice"),
            None,
            None,
            &[],
            None,
        );
        let default_codex = all.codex.clone();
        apply_session_path_overrides(
            &mut all,
            &[
                ("claude_code".to_string(), vec![PathBuf::from("/srv/cc")]),
                // Unknown sources are ignored, not an error.
                ("cursor".to_string(), vec![PathBuf::from("/srv/cursor")]),
            ],
        );
        assert_eq!(all.claude_code, vec![PathBuf::from("/srv/cc")]);
        assert_eq!(all.codex, default_codex);
        // An empty list disables a source entirely.
        apply_session_path_overrides(&mut all, &[("codex".to_string(), vec![])]);
        assert!(all.codex.is_empty());
    }

    #[test]
    fn codex_home_override_is_added_not_substituted() {
        let paths = compute_session_paths(
            Platform::Linux,
            Path::new("/home/alice"),
            None,
            None,
            &[],
            Some(PathBuf::from("/srv/codex-state")),
        );
        assert_eq!(
            paths.codex,
            vec![
                PathBuf::from("/home/alice/.codex/sessions"),
                PathBuf::from("/home/alice/.codex/archived_sessions"),
                PathBuf::from("/srv/codex-state/sessions"),
                PathBuf::from("/srv/codex-state/archived_sessions"),
            ]
        );
        // Pointing CODEX_HOME at the default adds nothing.
        let same = compute_session_paths(
            Platform::Linux,
            Path::new("/home/alice"),
            None,
            None,
            &[],
            Some(PathBuf::from("/home/alice/.codex")),
        );
        assert_eq!(same.codex.len(), 2);
    }

    #[test]
    fn wsl_codex_paths_include_windows_home() {
        let paths = compute_session_paths(
            Platform::Wsl,
            Path::new("/home/alice"),
            None,
            Some("Alice"),
            &[],
            None,
        );
        assert_eq!(paths.codex.len(), 4);
        assert_eq!(
            paths.codex[2],
            PathBuf::from("/mnt/c/Users/Alice/.codex/sessions")
        );
    }
}
