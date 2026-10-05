//! Configuration types and loading.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::Result;

/// MCP transport type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    /// Stdio transport.
    Stdio,
    /// HTTP transport.
    Http,
}

/// MCP server configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    /// Server name.
    pub name: String,
    /// Transport type.
    pub transport: McpTransport,
    /// Command for stdio transport.
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments for stdio transport.
    #[serde(default)]
    pub args: Vec<String>,
    /// URL for HTTP transport.
    #[serde(default)]
    pub url: Option<String>,
    /// Environment variables.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// Lume configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LumeConfig {
    /// Ollama base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Small model name.
    #[serde(default = "default_small_model")]
    pub small_model: String,
    /// Large model name.
    #[serde(default = "default_large_model")]
    pub large_model: String,
    /// Maximum iterations.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Context window size.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// Workspace root.
    #[serde(default)]
    pub workspace_root: Option<PathBuf>,
    /// MCP servers.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".to_string()
}

fn default_small_model() -> String {
    "qwen2.5-coder:1.5b".to_string()
}

fn default_large_model() -> String {
    "qwen2.5:7b".to_string()
}

fn default_max_iterations() -> usize {
    24
}

fn default_context_window() -> usize {
    8192
}

impl Default for LumeConfig {
    fn default() -> Self {
        Self {
            ollama_url: default_ollama_url(),
            small_model: default_small_model(),
            large_model: default_large_model(),
            max_iterations: default_max_iterations(),
            context_window: default_context_window(),
            workspace_root: None,
            mcp_servers: Vec::new(),
        }
    }
}

impl LumeConfig {
    /// Overlay environment variables on top of the current values.
    pub fn apply_env(&mut self) {
        if let Ok(v) = env::var("LUME_OLLAMA_URL") {
            self.ollama_url = v;
        }
        if let Ok(v) = env::var("LUME_SMALL_MODEL") {
            self.small_model = v;
        }
        if let Ok(v) = env::var("LUME_LARGE_MODEL") {
            self.large_model = v;
        }
        if let Ok(v) = env::var("LUME_MAX_ITERATIONS") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_iterations = n;
            }
        }
        if let Ok(v) = env::var("LUME_CONTEXT_WINDOW") {
            if let Ok(n) = v.parse::<usize>() {
                self.context_window = n;
            }
        }
        if let Ok(v) = env::var("LUME_WORKSPACE_ROOT") {
            self.workspace_root = Some(PathBuf::from(v));
        }
    }

    /// Load configuration from environment variables (layered on top of defaults).
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.apply_env();
        cfg
    }

    /// Load configuration from a TOML file, then layer environment variables on
    /// top so that environment variables win over file values.
    ///
    /// Passing `None` skips the file, which makes this the single entry point
    /// every subcommand needs.
    pub fn resolve(path: Option<&Path>) -> Result<Self> {
        let mut cfg = match path {
            Some(path) => Self::load(path)?,
            None => Self::default(),
        };
        cfg.apply_env();
        Ok(cfg)
    }

    /// Load configuration from TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&content)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_config(name: &str, contents: &str) -> PathBuf {
        let path = env::temp_dir().join(format!("lume-config-test-{name}.toml"));
        fs::write(&path, contents).expect("write temp config");
        path
    }

    #[test]
    fn resolve_reads_a_toml_file() {
        let path = temp_config(
            "basic",
            "small_model = \"qwen2.5:3b\"\nmax_iterations = 7\n",
        );
        let cfg = LumeConfig::resolve(Some(&path)).expect("resolve");
        assert_eq!(cfg.small_model, "qwen2.5:3b");
        assert_eq!(cfg.max_iterations, 7);
        assert_eq!(cfg.large_model, default_large_model());
        fs::remove_file(&path).expect("remove temp config");
    }

    #[test]
    fn resolve_layers_env_over_file_values() {
        let path = temp_config("layered", "small_model = \"from-file\"\n");
        // SAFETY: `LUME_SMALL_MODEL` is read and written by no other test in this
        // binary, so no other thread can observe it mid-flight.
        unsafe { env::set_var("LUME_SMALL_MODEL", "from-env") };
        let cfg = LumeConfig::resolve(Some(&path)).expect("resolve");
        // SAFETY: as above.
        unsafe { env::remove_var("LUME_SMALL_MODEL") };
        assert_eq!(cfg.small_model, "from-env");
        fs::remove_file(&path).expect("remove temp config");
    }

    #[test]
    fn resolve_without_a_path_uses_defaults() {
        let cfg = LumeConfig::resolve(None).expect("resolve");
        assert_eq!(cfg.ollama_url, default_ollama_url());
        assert_eq!(cfg.max_iterations, default_max_iterations());
    }
}
