CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migrations (
  filename text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clerk_user_id text UNIQUE NOT NULL,
  email text,
  first_name text,
  last_name text,

  session_count integer NOT NULL DEFAULT 0,
  total_session_duration_seconds integer NOT NULL DEFAULT 0,

  active_memory_count integer NOT NULL DEFAULT 0,
  memory_enabled boolean NOT NULL DEFAULT true,

  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Logged-in users use clerk_user_id.
  -- Logged-out guest sessions may use guest_id temporarily for 24-hour continuity.
  clerk_user_id text REFERENCES user_profiles(clerk_user_id) ON DELETE SET NULL,
  guest_id text,

  livekit_room_id text,
  livekit_room_name text,

  companion_id text,
  companion_slug text,

  memory_scope text NOT NULL DEFAULT 'guest' CHECK (memory_scope IN ('guest', 'account')),

  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  duration_seconds integer,

  -- For guest sessions, this allows temporary 24-hour retention of session-linked memory.
  ttl_expires_at timestamptz,

  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT guest_session_requires_guest_or_user CHECK (
    clerk_user_id IS NOT NULL OR guest_id IS NOT NULL
  )
);

CREATE TABLE IF NOT EXISTS memory_units (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  memory_scope text NOT NULL CHECK (memory_scope IN ('guest', 'account')),

  clerk_user_id text REFERENCES user_profiles(clerk_user_id) ON DELETE SET NULL,
  guest_id text,

  companion_id text,
  session_id uuid REFERENCES app_sessions(id) ON DELETE SET NULL,

  -- Compact memory only, not raw transcript.
  content text NOT NULL,
  content_hash text,

  is_persistent boolean NOT NULL DEFAULT false,

  ttl_expires_at timestamptz,
  deleted_at timestamptz,

  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT guest_memory_requires_guest_id CHECK (
    memory_scope != 'guest' OR guest_id IS NOT NULL
  ),

  CONSTRAINT account_memory_requires_user_id CHECK (
    memory_scope != 'account' OR clerk_user_id IS NOT NULL
  ),

  CONSTRAINT guest_memory_requires_ttl CHECK (
    memory_scope != 'guest' OR ttl_expires_at IS NOT NULL
  )
);

CREATE TABLE IF NOT EXISTS memory_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  memory_unit_id uuid REFERENCES memory_units(id) ON DELETE SET NULL,
  session_id uuid REFERENCES app_sessions(id) ON DELETE SET NULL,

  clerk_user_id text REFERENCES user_profiles(clerk_user_id) ON DELETE SET NULL,
  guest_id text,

  event_type text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS assistant_response_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  session_id uuid REFERENCES app_sessions(id) ON DELETE SET NULL,

  clerk_user_id text REFERENCES user_profiles(clerk_user_id) ON DELETE SET NULL,
  guest_id text,

  companion_id text,

  -- Primary response-speed metric:
  -- user stops speaking -> assistant first audio.
  user_stopped_to_first_audio text,

  -- Optional diagnostic breakdowns kept for backend debugging.
  user_stopped_to_final_stt text,
  final_stt_to_first_audio text,
  final_stt_to_llm_complete text,
  tts_request_to_first_audio text,

  text_length integer,
  sentence_end_count integer,
  hume_requests_during_speech integer,
  tts_path text,
  model_version text,
  description_applied boolean,
  interrupted boolean,

  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_rollups (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  rollup_date date UNIQUE NOT NULL,

  total_sessions integer NOT NULL DEFAULT 0,
  registered_sessions integer NOT NULL DEFAULT 0,
  guest_sessions integer NOT NULL DEFAULT 0,

  total_session_duration_seconds integer NOT NULL DEFAULT 0,
  registered_session_duration_seconds integer NOT NULL DEFAULT 0,
  guest_session_duration_seconds integer NOT NULL DEFAULT 0,

  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_clerk_user_id ON user_profiles(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);

CREATE INDEX IF NOT EXISTS idx_app_sessions_clerk_user_id ON app_sessions(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_app_sessions_guest_id ON app_sessions(guest_id);
CREATE INDEX IF NOT EXISTS idx_app_sessions_started_at ON app_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_app_sessions_duration_seconds ON app_sessions(duration_seconds);
CREATE INDEX IF NOT EXISTS idx_app_sessions_ttl_expires_at ON app_sessions(ttl_expires_at);

CREATE INDEX IF NOT EXISTS idx_memory_units_clerk_user_id ON memory_units(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_memory_units_guest_id ON memory_units(guest_id);
CREATE INDEX IF NOT EXISTS idx_memory_units_scope ON memory_units(memory_scope);
CREATE INDEX IF NOT EXISTS idx_memory_units_session_id ON memory_units(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_units_expires_at ON memory_units(ttl_expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_units_deleted_at ON memory_units(deleted_at);
CREATE INDEX IF NOT EXISTS idx_memory_units_content_hash ON memory_units(content_hash);

CREATE INDEX IF NOT EXISTS idx_memory_events_memory_unit_id ON memory_events(memory_unit_id);
CREATE INDEX IF NOT EXISTS idx_memory_events_session_id ON memory_events(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at);

CREATE INDEX IF NOT EXISTS idx_assistant_response_metrics_session_id ON assistant_response_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_assistant_response_metrics_clerk_user_id ON assistant_response_metrics(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_assistant_response_metrics_guest_id ON assistant_response_metrics(guest_id);
CREATE INDEX IF NOT EXISTS idx_assistant_response_metrics_created_at ON assistant_response_metrics(created_at);

CREATE INDEX IF NOT EXISTS idx_usage_rollups_rollup_date ON usage_rollups(rollup_date);
