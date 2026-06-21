CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- End-of-session user feedback. Durable record for the team to review and, over
-- time, run inference over to learn what users actually want. Distinct from
-- memory_units (per-user personalization) — this is product/analytics data.
CREATE TABLE IF NOT EXISTS feedback (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Better Auth user id (verified) + email at submission time. Kept as plain
  -- text (no FK) since the auth tables are managed by Better Auth.
  user_id text,
  email text,

  -- The user's free-text feedback, and Arche's generated reply (filled in once
  -- the autonomous reply is sent; nullable until then / if generation fails).
  message text NOT NULL,
  reply text,
  replied_at timestamptz,

  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
