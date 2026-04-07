//! Multi-tenant scoping isolation tests.
//!
//! Verifies that tenant/user/group scoping correctly isolates data
//! across semantic memories, sessions, and the knowledge graph.
//!
//! Run:
//!   cargo test -p openfang-memory --test scoping_tests
//!   cargo test -p openfang-memory --features postgres --test scoping_tests

use openfang_types::agent::AgentId;
use openfang_types::context::RequestContext;
use openfang_types::memory::MemorySource;
use openfang_types::storage::{KnowledgeBackend, SemanticBackend};
use std::collections::HashMap;

// ─── SQLite scoping tests ─────────────────────────────────────────────

mod sqlite_scoping {
    use super::*;
    use openfang_memory::backends::SessionBackend;
    use openfang_memory::sqlite::{KnowledgeStore, SemanticStore, SessionStore};
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    fn setup() -> Arc<Mutex<Connection>> {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *const u8,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(sqlite_vec::sqlite3_vec_init as *const ())));
        }
        let conn = Connection::open_in_memory().unwrap();
        openfang_memory::sqlite::migration::run_migrations(&conn).unwrap();
        Arc::new(Mutex::new(conn))
    }

    #[test]
    fn semantic_tenant_isolation() {
        let conn = setup();
        let store = SemanticStore::new(conn);
        let agent = AgentId::new();

        let ctx_acme = RequestContext {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let ctx_startup = RequestContext {
            tenant_id: Some("startup".into()),
            ..Default::default()
        };

        // Store memory for each tenant
        SemanticBackend::remember(
            &store, agent, "Acme secret data", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Startup secret data", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_startup,
        ).unwrap();

        // Recall with acme filter -> only acme's memory
        let filter_acme = openfang_types::memory::MemoryFilter {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "secret", 10, Some(filter_acme), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Acme"));

        // Recall with startup filter -> only startup's memory
        let filter_startup = openfang_types::memory::MemoryFilter {
            tenant_id: Some("startup".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "secret", 10, Some(filter_startup), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Startup"));

        // No filter -> sees both (admin view)
        let results = SemanticBackend::recall(&store, "secret", 10, None, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn semantic_user_isolation() {
        let conn = setup();
        let store = SemanticStore::new(conn);
        let agent = AgentId::new();

        let ctx_alice = RequestContext {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let ctx_bob = RequestContext {
            user_id: Some("bob".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Alice private note", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Bob private note", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_bob,
        ).unwrap();

        // Alice filter -> only Alice's memory
        let filter_alice = openfang_types::memory::MemoryFilter {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "private note", 10, Some(filter_alice), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Alice"));

        // Bob filter -> only Bob's memory
        let filter_bob = openfang_types::memory::MemoryFilter {
            user_id: Some("bob".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "private note", 10, Some(filter_bob), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Bob"));

        // No filter -> sees both
        let results = SemanticBackend::recall(&store, "private note", 10, None, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn semantic_group_sharing() {
        let conn = setup();
        let store = SemanticStore::new(conn);
        let agent = AgentId::new();

        // Two users in the same group
        let ctx_alice = RequestContext {
            user_id: Some("alice".into()),
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        let ctx_bob = RequestContext {
            user_id: Some("bob".into()),
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        // A user in a different group
        let ctx_charlie = RequestContext {
            user_id: Some("charlie".into()),
            group_id: Some("team-beta".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Alice team-alpha insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Bob team-alpha insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_bob,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Charlie team-beta insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_charlie,
        ).unwrap();

        // Group filter for team-alpha -> sees both Alice's and Bob's memories
        let filter_alpha = openfang_types::memory::MemoryFilter {
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "insight", 10, Some(filter_alpha), None).unwrap();
        assert_eq!(results.len(), 2);
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("Alice")));
        assert!(contents.iter().any(|c| c.contains("Bob")));

        // Group filter for team-beta -> only Charlie
        let filter_beta = openfang_types::memory::MemoryFilter {
            group_id: Some("team-beta".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "insight", 10, Some(filter_beta), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Charlie"));

        // No filter -> sees all three
        let results = SemanticBackend::recall(&store, "insight", 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn session_carries_ctx_through_db() {
        let conn = setup();
        let store = SessionStore::new(conn);
        let agent = AgentId::new();

        let ctx = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            group_id: Some("engineering".into()),
        };

        // Create session with ctx
        let session = store.create_session(agent, &ctx).unwrap();
        let sid = session.id;

        // Reload from DB -> ctx should be preserved
        let loaded = store.get_session(sid).unwrap().expect("session should exist");
        assert_eq!(loaded.ctx.tenant_id.as_deref(), Some("acme"));
        assert_eq!(loaded.ctx.user_id.as_deref(), Some("alice"));
        assert_eq!(loaded.ctx.group_id.as_deref(), Some("engineering"));

        // Clean up
        store.delete_session(sid).unwrap();
    }

    #[test]
    fn knowledge_tenant_isolation() {
        let conn = setup();
        let store = KnowledgeStore::new(conn);

        let ctx_acme = RequestContext {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let ctx_startup = RequestContext {
            tenant_id: Some("startup".into()),
            ..Default::default()
        };

        // Add entities and a relation scoped to "acme"
        let alice_id = KnowledgeBackend::add_entity(&store, openfang_types::memory::Entity {
            id: String::new(),
            entity_type: openfang_types::memory::EntityType::Person,
            name: "Alice".to_string(),
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        let acme_corp_id = KnowledgeBackend::add_entity(&store, openfang_types::memory::Entity {
            id: String::new(),
            entity_type: openfang_types::memory::EntityType::Organization,
            name: "Acme Corp".to_string(),
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        KnowledgeBackend::add_relation(&store, openfang_types::memory::Relation {
            source: alice_id.clone(),
            relation: openfang_types::memory::RelationType::WorksAt,
            target: acme_corp_id.clone(),
            properties: HashMap::new(),
            confidence: 0.9,
            created_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        // Query with acme context -> should find the relation
        let matches = KnowledgeBackend::query_graph(&store, openfang_types::memory::GraphPattern {
            source: Some(alice_id.clone()),
            relation: None,
            target: None,
            max_depth: 1,
            tenant_id: None,
        }, &ctx_acme).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source.name, "Alice");
        assert_eq!(matches[0].target.name, "Acme Corp");

        // Query with startup context -> relation was stored under acme tenant,
        // so filtering by startup tenant should exclude it
        let matches = KnowledgeBackend::query_graph(&store, openfang_types::memory::GraphPattern {
            source: Some(alice_id),
            relation: None,
            target: None,
            max_depth: 1,
            tenant_id: None,
        }, &ctx_startup).unwrap();
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn semantic_combined_tenant_and_user_filter() {
        let conn = setup();
        let store = SemanticStore::new(conn);
        let agent = AgentId::new();

        // Two users in acme, one in startup
        let ctx_acme_alice = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let ctx_acme_bob = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("bob".into()),
            ..Default::default()
        };
        let ctx_startup_alice = RequestContext {
            tenant_id: Some("startup".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Acme Alice memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Acme Bob memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme_bob,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Startup Alice memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_startup_alice,
        ).unwrap();

        // Filter by tenant only -> 2 results for acme
        let filter_acme = openfang_types::memory::MemoryFilter {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_acme), None).unwrap();
        assert_eq!(results.len(), 2);

        // Filter by tenant + user -> 1 result for acme + alice
        let filter_acme_alice = openfang_types::memory::MemoryFilter {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_acme_alice), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Acme Alice"));

        // Filter by user "alice" across all tenants -> 2 results
        let filter_alice = openfang_types::memory::MemoryFilter {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_alice), None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn session_label_lookup_with_ctx() {
        let conn = setup();
        let store = SessionStore::new(conn);
        let agent = AgentId::new();

        let ctx = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };

        let labeled = store.create_session_with_label(agent, Some("scoping-test"), &ctx).unwrap();
        assert_eq!(labeled.label, Some("scoping-test".to_string()));

        // Find by label -> should preserve ctx
        let found = store.find_session_by_label(agent, "scoping-test").unwrap().unwrap();
        assert_eq!(found.id, labeled.id);
        assert_eq!(found.ctx.tenant_id.as_deref(), Some("acme"));
        assert_eq!(found.ctx.user_id.as_deref(), Some("alice"));

        // Clean up
        store.delete_agent_sessions(agent).unwrap();
    }
}

// ─── PostgreSQL scoping tests ─────────────────────────────────────────

#[cfg(feature = "postgres")]
mod postgres_scoping {
    use super::*;
    use openfang_memory::backends::SessionBackend;
    use openfang_memory::postgres::*;
    use openfang_types::storage::KnowledgeBackend;

    async fn setup() -> Option<deadpool_postgres::Pool> {
        let url = std::env::var("TEST_POSTGRES_URL")
            .unwrap_or_else(|_| "postgresql://openfang:openfang@localhost:5432/openfang_test".to_string());
        let pool = create_pool(&url, 2).ok()?;
        run_migrations(&pool).await.ok()?;
        Some(pool)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_semantic_tenant_isolation() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgSemanticStore::new(pool);
        let agent = AgentId::new();

        let ctx_acme = RequestContext {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let ctx_startup = RequestContext {
            tenant_id: Some("startup".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Acme secret data", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Startup secret data", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_startup,
        ).unwrap();

        // Recall with acme filter -> only acme's memory
        let filter_acme = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "secret", 10, Some(filter_acme), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Acme"));

        // Recall with startup filter -> only startup's memory
        let filter_startup = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            tenant_id: Some("startup".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "secret", 10, Some(filter_startup), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Startup"));

        // Agent filter only -> sees both
        let filter_agent = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "secret", 10, Some(filter_agent), None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_semantic_user_isolation() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgSemanticStore::new(pool);
        let agent = AgentId::new();

        let ctx_alice = RequestContext {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let ctx_bob = RequestContext {
            user_id: Some("bob".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Alice private note", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Bob private note", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_bob,
        ).unwrap();

        let filter_alice = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "private note", 10, Some(filter_alice), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Alice"));

        let filter_bob = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            user_id: Some("bob".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "private note", 10, Some(filter_bob), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Bob"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_semantic_group_sharing() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgSemanticStore::new(pool);
        let agent = AgentId::new();

        let ctx_alice = RequestContext {
            user_id: Some("alice".into()),
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        let ctx_bob = RequestContext {
            user_id: Some("bob".into()),
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        let ctx_charlie = RequestContext {
            user_id: Some("charlie".into()),
            group_id: Some("team-beta".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Alice team-alpha insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Bob team-alpha insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_bob,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Charlie team-beta insight", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_charlie,
        ).unwrap();

        // Group filter for team-alpha -> sees Alice and Bob
        let filter_alpha = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            group_id: Some("team-alpha".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "insight", 10, Some(filter_alpha), None).unwrap();
        assert_eq!(results.len(), 2);
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("Alice")));
        assert!(contents.iter().any(|c| c.contains("Bob")));

        // Group filter for team-beta -> only Charlie
        let filter_beta = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            group_id: Some("team-beta".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "insight", 10, Some(filter_beta), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Charlie"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_session_carries_ctx_through_db() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgSessionStore::new(pool);
        let agent = AgentId::new();

        let ctx = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            group_id: Some("engineering".into()),
        };

        let session = store.create_session(agent, &ctx).unwrap();
        let sid = session.id;

        let loaded = store.get_session(sid).unwrap().expect("session should exist");
        assert_eq!(loaded.ctx.tenant_id.as_deref(), Some("acme"));
        assert_eq!(loaded.ctx.user_id.as_deref(), Some("alice"));
        assert_eq!(loaded.ctx.group_id.as_deref(), Some("engineering"));

        store.delete_session(sid).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_knowledge_tenant_isolation() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgKnowledgeStore::new(pool);

        let ctx_acme = RequestContext {
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let ctx_startup = RequestContext {
            tenant_id: Some("startup".into()),
            ..Default::default()
        };

        let alice_id = KnowledgeBackend::add_entity(&store, openfang_types::memory::Entity {
            id: String::new(),
            entity_type: openfang_types::memory::EntityType::Person,
            name: "Alice".to_string(),
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        let acme_corp_id = KnowledgeBackend::add_entity(&store, openfang_types::memory::Entity {
            id: String::new(),
            entity_type: openfang_types::memory::EntityType::Organization,
            name: "Acme Corp".to_string(),
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        KnowledgeBackend::add_relation(&store, openfang_types::memory::Relation {
            source: alice_id.clone(),
            relation: openfang_types::memory::RelationType::WorksAt,
            target: acme_corp_id.clone(),
            properties: HashMap::new(),
            confidence: 0.9,
            created_at: chrono::Utc::now(),
            ctx: ctx_acme.clone(),
        }, &ctx_acme).unwrap();

        // Query with acme context -> visible
        let matches = KnowledgeBackend::query_graph(&store, openfang_types::memory::GraphPattern {
            source: Some(alice_id.clone()),
            relation: None,
            target: None,
            max_depth: 1,
            tenant_id: None,
        }, &ctx_acme).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source.name, "Alice");
        assert_eq!(matches[0].target.name, "Acme Corp");

        // Query with startup context -> not visible
        let matches = KnowledgeBackend::query_graph(&store, openfang_types::memory::GraphPattern {
            source: Some(alice_id),
            relation: None,
            target: None,
            max_depth: 1,
            tenant_id: None,
        }, &ctx_startup).unwrap();
        assert_eq!(matches.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pg_semantic_combined_tenant_and_user_filter() {
        let pool = match setup().await {
            Some(p) => p,
            None => { eprintln!("SKIP: PostgreSQL not available"); return; }
        };
        let store = PgSemanticStore::new(pool);
        let agent = AgentId::new();

        let ctx_acme_alice = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let ctx_acme_bob = RequestContext {
            tenant_id: Some("acme".into()),
            user_id: Some("bob".into()),
            ..Default::default()
        };
        let ctx_startup_alice = RequestContext {
            tenant_id: Some("startup".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };

        SemanticBackend::remember(
            &store, agent, "Acme Alice memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme_alice,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Acme Bob memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_acme_bob,
        ).unwrap();
        SemanticBackend::remember(
            &store, agent, "Startup Alice memory", MemorySource::Conversation,
            "episodic", HashMap::new(), None, &ctx_startup_alice,
        ).unwrap();

        // Filter by tenant only -> 2 results for acme
        let filter_acme = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            tenant_id: Some("acme".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_acme), None).unwrap();
        assert_eq!(results.len(), 2);

        // Filter by tenant + user -> 1 result
        let filter_acme_alice = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            tenant_id: Some("acme".into()),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_acme_alice), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Acme Alice"));

        // Filter by user "alice" across all tenants -> 2 results
        let filter_alice = openfang_types::memory::MemoryFilter {
            agent_id: Some(agent),
            user_id: Some("alice".into()),
            ..Default::default()
        };
        let results = SemanticBackend::recall(&store, "memory", 10, Some(filter_alice), None).unwrap();
        assert_eq!(results.len(), 2);
    }
}
