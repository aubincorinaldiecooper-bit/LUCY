-- Emotional-analyzer dataset (server-side only; never user-facing).
--
-- voice_profile_events: one row per voiceProfile capture on the Inworld
-- Realtime path — raw label arrays (the analyzer's output) + the normalized
-- weak-signal dims + the transcript on the carrying event when present.
--
-- emotional_calibration_moments: ground truth — the subtle either/or
-- check-in Arche asked and how the user answered, paired with the inferred
-- pattern at the time. Together these let sessions accumulate a labeled
-- dataset for improving/evaluating the emotional analyzer.

CREATE TABLE IF NOT EXISTS voice_profile_events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_event TEXT NOT NULL,
    transcript TEXT,
    raw_profile JSONB NOT NULL,
    normalized_profile JSONB NOT NULL,
    emotion_confidence REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_voice_profile_events_session
    ON voice_profile_events (session_id, created_at);

CREATE TABLE IF NOT EXISTS emotional_calibration_moments (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    transcript TEXT,
    arche_question TEXT,
    user_answer TEXT,
    user_confirmed_or_corrected BOOLEAN,
    inferred_emotional_pattern TEXT,
    normalized_profile JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_emotional_calibration_moments_session
    ON emotional_calibration_moments (session_id, created_at);
