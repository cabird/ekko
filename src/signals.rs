//! Process shutdown signal, shared by every long-running subcommand.

/// Resolve on Ctrl-C (and SIGTERM on Unix). If the SIGTERM handler cannot
/// be installed, degrade to Ctrl-C only rather than failing to start.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_signal_resolves_on_sigterm() {
        // Poll once so tokio installs its SIGTERM handler (which also keeps
        // the raised signal from killing the test process), then raise it.
        let waiter = tokio::spawn(shutdown_signal());
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let status = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .expect("kill runs");
        assert!(status.success());
        tokio::time::timeout(std::time::Duration::from_secs(3), waiter)
            .await
            .expect("shutdown_signal did not resolve on SIGTERM")
            .unwrap();
    }
}
