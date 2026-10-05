//! Identity and capability declarations exchanged with MCP servers.
//!
//! Roots, sampling and logging are deprecated as of `2026-07-28` (earliest
//! removal `2027-07-28`) and are therefore not declared anywhere in this crate.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Name this harness advertises to servers.
pub const DEFAULT_CLIENT_NAME: &str = "lume";

/// Client identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    /// Name.
    pub name: String,
    /// Version.
    pub version: String,
}

impl ClientInfo {
    /// The identity this crate advertises: the lume harness at its own version.
    pub fn lume() -> Self {
        Self {
            name: DEFAULT_CLIENT_NAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Server identity, returned per result in `_meta` and by `server/discover`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    /// Name.
    pub name: String,
    /// Version.
    pub version: String,
}

/// Tools capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Whether the server emits `notifications/tools/list_changed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Capabilities a client declares.
///
/// Deliberately empty by default: the only client-side capabilities MCP ever
/// defined were roots, sampling and logging, and all three are deprecated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    /// Tools capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Free-form extension capabilities for out-of-spec features.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}

/// Capabilities a server advertises through `server/discover` or `initialize`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    /// Tools capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Free-form extension capabilities for out-of-spec features.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}
