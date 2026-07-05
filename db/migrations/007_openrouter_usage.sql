-- OpenRouter spend metering (server-side only; never user-facing).
--
-- One row per OpenRouter call the product makes (vision-frame description,
-- mood fusion). Attributed to a durable billing key (user_ref) and the
-- session it happened in, with token counts and the charged cost (credits ==
-- USD, captured when the request asks OpenRouter for cost accounting). Lets an
-- operator aggregate spend per user and forward it to paid accounts as a meter
-- (see openrouter_metering.fetch_spend_summary). Written fire-and-forget and
-- only when OPENROUTER_METERING_ENABLED=true; never on the voice-turn path.

CREATE TABLE IF NOT EXISTS openrouter_usage_events (
    id BIGSERIAL PRIMARY KEY,
    -- Durable per-user billing key (memory identity key: clerk id / guest id /
    -- "api:<user_ref>"). NULL for sessions with no resolved identity — those
    -- rows are still attributable by session_id.
    user_ref TEXT,
    session_id TEXT,
    feature TEXT NOT NULL,          -- 'vision' | 'mood'
    model TEXT NOT NULL,            -- the OpenRouter model id actually billed
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    -- Charged cost in USD. NUMERIC (not float) so summed bills stay exact.
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_openrouter_usage_user
    ON openrouter_usage_events (user_ref, created_at);

CREATE INDEX IF NOT EXISTS idx_openrouter_usage_session
    ON openrouter_usage_events (session_id, created_at);
