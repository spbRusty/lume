//! `_meta` construction and parsing.
//!
//! The stateless revision has no handshake, so the protocol version and the
//! client's capabilities travel with every request. Clients also identify
//! themselves per request; servers answer with their own identity in the
//! `_meta` of each result.

use lume_core::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::capabilities::{ClientCapabilities, ClientInfo, ServerInfo};
use crate::negotiation::ProtocolRevision;

/// `_meta` key holding the protocol revision of a request.
pub const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";

/// `_meta` key holding the capabilities the client declares.
pub const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";

/// `_meta` key holding the client's identity.
pub const META_CLIENT_INFO: &str = "io.modelcontextprotocol/clientInfo";

/// `_meta` key under which a server returns its identity in a result.
pub const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

/// The `_meta` object the client attaches to every request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMeta {
    /// Protocol revision spoken by the client.
    #[serde(rename = "io.modelcontextprotocol/protocolVersion")]
    pub protocol_version: String,
    /// Capabilities the client declares.
    #[serde(rename = "io.modelcontextprotocol/clientCapabilities")]
    pub client_capabilities: ClientCapabilities,
    /// Identity the client advertises.
    #[serde(rename = "io.modelcontextprotocol/clientInfo")]
    pub client_info: ClientInfo,
}

impl RequestMeta {
    /// Build `_meta` for `revision`.
    pub fn new(
        revision: ProtocolRevision,
        client_info: &ClientInfo,
        client_capabilities: &ClientCapabilities,
    ) -> Self {
        Self {
            protocol_version: revision.version().to_string(),
            client_capabilities: client_capabilities.clone(),
            client_info: client_info.clone(),
        }
    }

    /// Serialize into the JSON object carried as the request's `_meta`.
    pub fn to_value(&self) -> Result<Value> {
        Ok(serde_json::to_value(self)?)
    }
}

/// Read a server's identity out of a result's `_meta`.
pub fn server_info_from_meta(meta: Option<&Value>) -> Option<ServerInfo> {
    meta?
        .get(META_SERVER_INFO)
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        META_CLIENT_CAPABILITIES, META_CLIENT_INFO, META_PROTOCOL_VERSION, META_SERVER_INFO,
        RequestMeta, server_info_from_meta,
    };
    use crate::capabilities::{ClientCapabilities, ClientInfo, ServerInfo};
    use crate::negotiation::{CURRENT_PROTOCOL_VERSION, ProtocolRevision};

    fn meta() -> RequestMeta {
        RequestMeta::new(
            ProtocolRevision::Stateless2026_07_28,
            &ClientInfo::lume(),
            &ClientCapabilities::default(),
        )
    }

    #[test]
    fn request_meta_carries_the_three_required_keys() {
        let value = meta().to_value().expect("serializable meta");
        let object = value.as_object().expect("meta is an object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = vec![
            META_PROTOCOL_VERSION,
            META_CLIENT_CAPABILITIES,
            META_CLIENT_INFO,
        ];
        expected.sort_unstable();
        assert_eq!(keys, expected);
        assert_eq!(
            object[META_PROTOCOL_VERSION],
            json!(CURRENT_PROTOCOL_VERSION)
        );
        assert_eq!(object[META_CLIENT_INFO]["name"], json!("lume"));
        assert!(object[META_CLIENT_CAPABILITIES].is_object());
    }

    #[test]
    fn keys_match_the_spec_strings() {
        assert_eq!(
            META_PROTOCOL_VERSION,
            "io.modelcontextprotocol/protocolVersion"
        );
        assert_eq!(
            META_CLIENT_CAPABILITIES,
            "io.modelcontextprotocol/clientCapabilities"
        );
        assert_eq!(META_CLIENT_INFO, "io.modelcontextprotocol/clientInfo");
        assert_eq!(META_SERVER_INFO, "io.modelcontextprotocol/serverInfo");
    }

    #[test]
    fn legacy_revision_is_reported_verbatim() {
        let value = RequestMeta::new(
            ProtocolRevision::LegacyHandshake2025_06_18,
            &ClientInfo::lume(),
            &ClientCapabilities::default(),
        )
        .to_value()
        .expect("serializable meta");
        assert_eq!(value[META_PROTOCOL_VERSION], json!("2025-06-18"));
    }

    #[test]
    fn server_identity_is_read_from_meta() {
        let meta = json!({META_SERVER_INFO: {"name": "files", "version": "2.1"}});
        let info = server_info_from_meta(Some(&meta)).expect("identity present");
        assert_eq!(
            info,
            ServerInfo {
                name: "files".to_string(),
                version: "2.1".to_string(),
            }
        );
    }

    #[test]
    fn server_identity_missing_from_meta_yields_none() {
        assert!(server_info_from_meta(None).is_none());
        assert!(server_info_from_meta(Some(&json!({}))).is_none());
        assert!(server_info_from_meta(Some(&json!({"io.example.org/serverInfo": {}}))).is_none());
    }
}
