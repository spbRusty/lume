//! Protocol revision negotiation.
//!
//! Revision `2026-07-28` made MCP stateless and added `server/discover`, which
//! every conforming server must implement. Almost every MCP server deployed
//! today predates that revision, so a client that only speaks the new one is
//! useless in practice: this module keeps the decision in one place so
//! [`NegotiatedProtocol`] can be either the stateless revision or the legacy
//! `2025-06-18` handshake.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::capabilities::{ClientCapabilities, ClientInfo, ServerCapabilities, ServerInfo};
use crate::results::default_result_type;

/// The revision this client speaks by default.
pub const CURRENT_PROTOCOL_VERSION: &str = "2026-07-28";

/// The revision removed from this client but still spoken by most servers.
pub const LEGACY_PROTOCOL_VERSION: &str = "2025-06-18";

/// Mandatory server method advertising supported revisions and capabilities.
pub const SERVER_DISCOVER_METHOD: &str = "server/discover";

/// Legacy handshake request, removed in `2026-07-28`.
pub const INITIALIZE_METHOD: &str = "initialize";

/// Legacy handshake notification, removed in `2026-07-28`.
pub const INITIALIZED_NOTIFICATION: &str = "notifications/initialized";

/// A protocol revision this client can speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolRevision {
    /// `2026-07-28`: stateless, version and capabilities carried per request in
    /// `_meta`.
    Stateless2026_07_28,
    /// `2025-06-18`: `initialize` plus `notifications/initialized` handshake,
    /// kept for servers that predate `server/discover`.
    LegacyHandshake2025_06_18,
}

impl ProtocolRevision {
    /// The revision string as it appears on the wire.
    pub const fn version(self) -> &'static str {
        match self {
            Self::Stateless2026_07_28 => CURRENT_PROTOCOL_VERSION,
            Self::LegacyHandshake2025_06_18 => LEGACY_PROTOCOL_VERSION,
        }
    }
}

impl fmt::Display for ProtocolRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.version())
    }
}

/// Parameters of the legacy `initialize` handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol revision requested by the client.
    pub protocol_version: String,
    /// Client identity.
    pub client_info: ClientInfo,
    /// Capabilities the client declares.
    #[serde(default)]
    pub capabilities: ClientCapabilities,
}

/// Result of the legacy `initialize` handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol revision the server settled on.
    pub protocol_version: String,
    /// Capabilities the server declares.
    #[serde(default)]
    pub capabilities: ServerCapabilities,
    /// Server identity.
    pub server_info: ServerInfo,
}

/// Result of the mandatory `server/discover` probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverResult {
    /// Result discriminant.
    #[serde(default = "default_result_type")]
    pub result_type: crate::results::ResultType,
    /// Every protocol revision the server supports, newest first.
    pub protocol_versions: Vec<String>,
    /// Capabilities the server declares.
    #[serde(default)]
    pub capabilities: ServerCapabilities,
    /// Server identity.
    pub server_info: ServerInfo,
    /// `_meta` object reserved by MCP.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

/// Outcome of probing a server with `server/discover`.
#[derive(Debug, Clone)]
pub enum DiscoverProbe {
    /// The server answered, so it speaks the stateless revision.
    Answered(DiscoverResult),
    /// The server rejected `server/discover` as an unknown method, so it
    /// predates the stateless revision.
    MethodNotFound,
}

/// The revision chosen for a connection, plus what the server said about itself.
#[derive(Debug, Clone)]
pub struct NegotiatedProtocol {
    revision: ProtocolRevision,
    advertised_versions: Vec<String>,
    capabilities: ServerCapabilities,
    server_info: Option<ServerInfo>,
    legacy_handshake: Option<InitializeResult>,
}

impl NegotiatedProtocol {
    /// Choose the revision from the `server/discover` outcome.
    ///
    /// `legacy_handshake` carries the handshake result when the legacy path was
    /// taken and already completed.
    pub fn from_probe(probe: DiscoverProbe, legacy_handshake: Option<InitializeResult>) -> Self {
        match probe {
            DiscoverProbe::Answered(discovered) => Self {
                revision: ProtocolRevision::Stateless2026_07_28,
                advertised_versions: discovered.protocol_versions,
                capabilities: discovered.capabilities,
                server_info: Some(discovered.server_info),
                legacy_handshake: None,
            },
            DiscoverProbe::MethodNotFound => match legacy_handshake {
                Some(handshake) => Self {
                    revision: ProtocolRevision::LegacyHandshake2025_06_18,
                    advertised_versions: vec![handshake.protocol_version.clone()],
                    capabilities: handshake.capabilities.clone(),
                    server_info: Some(handshake.server_info.clone()),
                    legacy_handshake: Some(handshake),
                },
                None => Self {
                    revision: ProtocolRevision::LegacyHandshake2025_06_18,
                    advertised_versions: Vec::new(),
                    capabilities: ServerCapabilities::default(),
                    server_info: None,
                    legacy_handshake: None,
                },
            },
        }
    }

    /// The revision in use.
    pub fn revision(&self) -> ProtocolRevision {
        self.revision
    }

    /// Protocol revisions the server advertised.
    pub fn advertised_versions(&self) -> &[String] {
        &self.advertised_versions
    }

    /// Capabilities the server declared.
    pub fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    /// Identity the server reported, if it has reported one yet.
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }

    /// The legacy handshake result, when the legacy path was taken.
    pub fn legacy_handshake(&self) -> Option<&InitializeResult> {
        self.legacy_handshake.as_ref()
    }

    /// An `InitializeResult` describing this session.
    ///
    /// On the legacy path this is the server's own handshake result. On the
    /// stateless path there is no `initialize` round trip at all, so the result
    /// is synthesised from the `server/discover` advertisement. `None` means the
    /// server has not identified itself yet, which happens only between
    /// rejecting `server/discover` and finishing the legacy handshake.
    pub fn initialize_result(&self) -> Option<InitializeResult> {
        match &self.legacy_handshake {
            Some(handshake) => Some(handshake.clone()),
            None => Some(InitializeResult {
                protocol_version: self.revision.version().to_string(),
                capabilities: self.capabilities.clone(),
                server_info: self.server_info.clone()?,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        CURRENT_PROTOCOL_VERSION, DiscoverProbe, DiscoverResult, LEGACY_PROTOCOL_VERSION,
        NegotiatedProtocol, ProtocolRevision,
    };
    use crate::capabilities::{ServerCapabilities, ServerInfo};
    use crate::results::ResultType;

    fn discover() -> DiscoverResult {
        DiscoverResult {
            result_type: ResultType::Complete,
            protocol_versions: vec![CURRENT_PROTOCOL_VERSION.to_string()],
            capabilities: ServerCapabilities {
                tools: None,
                extensions: Some(json!({"x-vendor": 1})),
            },
            server_info: ServerInfo {
                name: "files".to_string(),
                version: "2.1".to_string(),
            },
            meta: None,
        }
    }

    #[test]
    fn discover_success_selects_the_stateless_revision() {
        let negotiated = NegotiatedProtocol::from_probe(DiscoverProbe::Answered(discover()), None);
        assert_eq!(negotiated.revision(), ProtocolRevision::Stateless2026_07_28);
        assert_eq!(negotiated.advertised_versions(), [CURRENT_PROTOCOL_VERSION]);
        assert_eq!(
            negotiated.server_info().map(|info| info.name.as_str()),
            Some("files")
        );
        assert!(negotiated.legacy_handshake().is_none());
    }

    #[test]
    fn method_not_found_selects_the_legacy_revision() {
        let negotiated = NegotiatedProtocol::from_probe(DiscoverProbe::MethodNotFound, None);
        assert_eq!(
            negotiated.revision(),
            ProtocolRevision::LegacyHandshake2025_06_18
        );
        assert!(negotiated.advertised_versions().is_empty());
        assert!(negotiated.legacy_handshake().is_none());
        assert!(negotiated.server_info().is_none());
        assert!(negotiated.initialize_result().is_none());
    }

    #[test]
    fn legacy_path_reports_the_handshaked_revision() {
        let handshake = crate::negotiation::InitializeResult {
            protocol_version: LEGACY_PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities::default(),
            server_info: ServerInfo {
                name: "legacy".to_string(),
                version: "0.9".to_string(),
            },
        };
        let negotiated =
            NegotiatedProtocol::from_probe(DiscoverProbe::MethodNotFound, Some(handshake.clone()));
        assert_eq!(
            negotiated.revision(),
            ProtocolRevision::LegacyHandshake2025_06_18
        );
        assert_eq!(negotiated.advertised_versions(), [LEGACY_PROTOCOL_VERSION]);
        assert_eq!(
            negotiated
                .initialize_result()
                .map(|result| result.protocol_version),
            Some(LEGACY_PROTOCOL_VERSION.to_string())
        );
        assert_eq!(
            negotiated
                .legacy_handshake()
                .map(|result| result.server_info.name.as_str()),
            Some("legacy")
        );
    }

    #[test]
    fn stateless_initialize_result_is_synthesised_from_discovery() {
        let negotiated = NegotiatedProtocol::from_probe(DiscoverProbe::Answered(discover()), None);
        let synthesized = negotiated
            .initialize_result()
            .expect("stateless path knows the server identity");
        assert_eq!(synthesized.protocol_version, CURRENT_PROTOCOL_VERSION);
        assert_eq!(synthesized.server_info.name, "files");
        assert_eq!(
            synthesized.capabilities.extensions,
            Some(json!({"x-vendor": 1}))
        );
    }

    #[test]
    fn revision_strings_match_the_spec() {
        assert_eq!(
            ProtocolRevision::Stateless2026_07_28.version(),
            "2026-07-28"
        );
        assert_eq!(
            ProtocolRevision::LegacyHandshake2025_06_18.version(),
            "2025-06-18"
        );
    }

    #[test]
    fn discover_result_defaults_result_type_for_legacy_shaped_payloads() {
        let parsed: DiscoverResult = serde_json::from_value(json!({
            "protocolVersions": ["2026-07-28"],
            "serverInfo": {"name": "files", "version": "2.1"},
        }))
        .expect("payload without resultType must parse");
        assert_eq!(parsed.result_type, ResultType::Complete);
        assert_eq!(parsed.capabilities, ServerCapabilities::default());
    }
}
