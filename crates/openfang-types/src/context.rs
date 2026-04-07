//! Request-scoped identity context for multi-tenant, multi-user scoping.
//!
//! Bundles tenant/user/group identity into one struct that flows through
//! the system. All fields are optional for backward compatibility —
//! `None` means no scoping (single-tenant mode).

use serde::{Deserialize, Serialize};

/// Request-scoped identity context.
///
/// Created once at the API/channel boundary, then passed through the
/// kernel, runtime, and memory layers. Avoids parameter explosion —
/// one struct instead of 3 separate params on every function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestContext {
    /// Tenant/organization ID for multi-tenant isolation.
    pub tenant_id: Option<String>,
    /// User ID of the person interacting with the agent.
    pub user_id: Option<String>,
    /// Group/channel ID for shared conversations.
    pub group_id: Option<String>,
}

impl RequestContext {
    /// Create a context with all fields set.
    pub fn new(tenant_id: Option<String>, user_id: Option<String>, group_id: Option<String>) -> Self {
        Self { tenant_id, user_id, group_id }
    }

    /// Parse tenant_id as a UUID if present and valid.
    pub fn tenant_uuid(&self) -> Option<uuid::Uuid> {
        self.tenant_id.as_deref().and_then(|s| uuid::Uuid::parse_str(s).ok())
    }

    /// Parse user_id as a UUID if present and valid.
    pub fn user_uuid(&self) -> Option<uuid::Uuid> {
        self.user_id.as_deref().and_then(|s| uuid::Uuid::parse_str(s).ok())
    }

    /// Parse group_id as a UUID if present and valid.
    pub fn group_uuid(&self) -> Option<uuid::Uuid> {
        self.group_id.as_deref().and_then(|s| uuid::Uuid::parse_str(s).ok())
    }
}
