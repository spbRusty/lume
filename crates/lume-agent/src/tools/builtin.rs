//! Builtin tools with safety checks.

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::json;

use lume_core::error::{LumeError, Result};
use lume_core::tool::Tool;
use lume_core::types::ToolSpec;

/// Read file tool.
pub struct ReadFileTool {
    workspace_root: PathBuf,
}

impl ReadFileTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "read_file".to_string(),
                message: "path traversal detected".to_string(),
            });
        }
        Ok(canon)
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read a file from workspace".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let path = args["path"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "read_file".to_string(),
            message: "missing path".to_string(),
        })?;
        let p = self.resolve_path(path)?;
        let content = fs::read_to_string(&p)?;
        Ok(content)
    }
}

/// Write file tool.
pub struct WriteFileTool {
    workspace_root: PathBuf,
}

impl WriteFileTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "write_file".to_string(),
                message: "path traversal detected".to_string(),
            });
        }
        Ok(canon)
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".to_string(),
            description: "Write a file to workspace".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let path = args["path"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "write_file".to_string(),
            message: "missing path".to_string(),
        })?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| LumeError::ToolFailed {
                name: "write_file".to_string(),
                message: "missing content".to_string(),
            })?;
        let p = self.resolve_path(path)?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&p, content)?;
        Ok("ok".to_string())
    }
}

/// Edit file tool.
pub struct EditFileTool {
    workspace_root: PathBuf,
}

impl EditFileTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "edit_file".to_string(),
                message: "path traversal detected".to_string(),
            });
        }
        Ok(canon)
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".to_string(),
            description: "Edit a file in workspace".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"}
                },
                "required": ["path", "old", "new"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let path = args["path"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "edit_file".to_string(),
            message: "missing path".to_string(),
        })?;
        let old = args["old"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "edit_file".to_string(),
            message: "missing old".to_string(),
        })?;
        let newv = args["new"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "edit_file".to_string(),
            message: "missing new".to_string(),
        })?;
        let p = self.resolve_path(path)?;
        let content = fs::read_to_string(&p)?;
        if !content.contains(old) {
            return Err(LumeError::ToolFailed {
                name: "edit_file".to_string(),
                message: "old string not found".to_string(),
            });
        }
        let replaced = content.replace(old, newv);
        fs::write(&p, replaced)?;
        Ok("ok".to_string())
    }
}

/// List directory tool.
pub struct ListDirTool {
    workspace_root: PathBuf,
}

impl ListDirTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "list_dir".to_string(),
                message: "path traversal detected".to_string(),
            });
        }
        Ok(canon)
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".to_string(),
            description: "List directory contents".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let path = args["path"].as_str().ok_or_else(|| LumeError::ToolFailed {
            name: "list_dir".to_string(),
            message: "missing path".to_string(),
        })?;
        let p = self.resolve_path(path)?;
        let mut entries: Vec<String> = fs::read_dir(&p)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        entries.sort();
        Ok(entries.join("\n"))
    }
}

/// Grep tool.
pub struct GrepTool {
    workspace_root: PathBuf,
}

impl GrepTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "grep".to_string(),
                message: "path traversal detected".to_string(),
            });
        }
        Ok(canon)
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".to_string(),
            description: "Search for pattern in files".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| LumeError::ToolFailed {
                name: "grep".to_string(),
                message: "missing pattern".to_string(),
            })?;
        let path = args["path"].as_str().unwrap_or(".");
        let root = self.resolve_path(path)?;
        let mut results: Vec<String> = Vec::new();
        fn walk(dir: &Path, root: &Path, pattern: &str, results: &mut Vec<String>) -> Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str == ".git" || name_str == "target" || name_str == "node_modules" {
                    continue;
                }
                if path.is_dir() {
                    let _ = walk(&path, root, pattern, results);
                } else if path.is_file() {
                    if let Ok(content) = fs::read_to_string(&path) {
                        for (i, line) in content.lines().enumerate() {
                            if line.contains(pattern) {
                                let rel = path.strip_prefix(root).unwrap_or(&path);
                                results.push(format!("{}:{}: {}", rel.display(), i + 1, line));
                                if results.len() >= 200 {
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        walk(&root, &root, pattern, &mut results)?;
        Ok(results.join("\n"))
    }
}

/// Shell tool with safety checks.
pub struct ShellTool {
    workspace_root: PathBuf,
    max_output_bytes: usize,
}

/// Command fragments refused outright, matched against a lowercased and
/// whitespace-collapsed copy of the command. Some entries over-match on purpose —
/// `rm -rf /tmp/x` trips `rm -rf /` — because without a sandbox a false positive costs
/// one refused command while a false negative costs the machine.
const BLOCKED: &[&str] = &[
    "rm -rf /",
    "rm -fr /",
    "rm -rf --no-preserve-root",
    "rm -fr --no-preserve-root",
    "mkfs",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    ":(){",
    "dd if=/dev/zero of=/dev/",
    "chmod -r 777 /",
    "chown -r root /",
    "> /dev/sda",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
];

fn normalize_for_matching(command: &str) -> String {
    let mut out = String::with_capacity(command.len());
    let mut last_was_space = false;
    for ch in command.chars().flat_map(char::to_lowercase) {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out
}

impl ShellTool {
    /// Create new tool.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            max_output_bytes: 32 * 1024,
        }
    }

    /// Override the combined stdout+stderr budget.
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    fn resolve_path(&self, p: &str) -> Result<PathBuf> {
        let joined = self.workspace_root.join(p);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&self.workspace_root) {
            return Err(LumeError::ToolFailed {
                name: "shell".to_string(),
                message: format!("path escapes the workspace root: {p}"),
            });
        }
        Ok(canon)
    }

    /// Reject the command when an absolute path in it resolves outside the workspace.
    ///
    /// This is a heuristic, not a guarantee: it inspects whitespace-separated tokens, so
    /// an absolute path assembled at runtime (`p=/etc; cat "$p/passwd"`) is not seen here.
    /// The real boundary is the denylist plus the pinned working directory.
    fn check_paths(&self, command: &str) -> Result<()> {
        for token in command.split_whitespace() {
            let cleaned =
                token.trim_matches(|c| matches!(c, '"' | '\'' | '`' | ',' | ';' | '(' | ')'));
            if !cleaned.starts_with('/') || cleaned == "/" {
                continue;
            }
            self.resolve_path(cleaned)?;
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".to_string(),
            description: "Run shell command in workspace".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| LumeError::ToolFailed {
                name: "shell".to_string(),
                message: "missing command".to_string(),
            })?;

        let normalized = normalize_for_matching(command);
        for fragment in BLOCKED {
            if normalized.contains(fragment) {
                return Err(LumeError::ToolFailed {
                    name: "shell".to_string(),
                    message: format!("blocked pattern: {fragment}"),
                });
            }
        }
        self.check_paths(command)?;

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.workspace_root)
            .output()
            .await?;

        let mut out = String::new();
        out.push_str(&String::from_utf8_lossy(&output.stdout));
        if !output.stderr.is_empty() {
            out.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        Ok(crate::r#loop::truncate_to_bytes(
            &out,
            self.max_output_bytes,
        ))
    }
}
