-- Better Auth core schema (user / session / account / verification).
--
-- Why this is a SQL migration instead of `npx @better-auth/cli migrate`:
-- the CLI must connect to Postgres to introspect, but our Postgres is only
-- reachable at the internal host (postgres.railway.internal) from inside
-- Railway — a local CLI run can't reach it. The app's migration runner
-- (scripts/apply_migrations.py) DOES run inside Railway, so shipping the schema
-- here creates the tables on deploy (requires RUN_DB_MIGRATIONS_ON_STARTUP=true).
--
-- Schema derived from @better-auth/core get-tables.ts + the Postgres type map
-- (string->text, date->timestamptz, boolean->boolean, id/fk->text). Column names
-- are camelCase and MUST stay quoted so they match Better Auth's quoted queries
-- (an unquoted emailVerified would fold to emailverified and never match).
-- The magic-link plugin adds no tables of its own (it uses `verification`).

CREATE TABLE IF NOT EXISTS "user" (
  "id" text NOT NULL PRIMARY KEY,
  "name" text NOT NULL,
  "email" text NOT NULL UNIQUE,
  "emailVerified" boolean NOT NULL,
  "image" text,
  "createdAt" timestamptz NOT NULL,
  "updatedAt" timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS "session" (
  "id" text NOT NULL PRIMARY KEY,
  "expiresAt" timestamptz NOT NULL,
  "token" text NOT NULL UNIQUE,
  "createdAt" timestamptz NOT NULL,
  "updatedAt" timestamptz NOT NULL,
  "ipAddress" text,
  "userAgent" text,
  "userId" text NOT NULL REFERENCES "user" ("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "account" (
  "id" text NOT NULL PRIMARY KEY,
  "accountId" text NOT NULL,
  "providerId" text NOT NULL,
  "userId" text NOT NULL REFERENCES "user" ("id") ON DELETE CASCADE,
  "accessToken" text,
  "refreshToken" text,
  "idToken" text,
  "accessTokenExpiresAt" timestamptz,
  "refreshTokenExpiresAt" timestamptz,
  "scope" text,
  "password" text,
  "createdAt" timestamptz NOT NULL,
  "updatedAt" timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS "verification" (
  "id" text NOT NULL PRIMARY KEY,
  "identifier" text NOT NULL,
  "value" text NOT NULL,
  "expiresAt" timestamptz NOT NULL,
  "createdAt" timestamptz NOT NULL,
  "updatedAt" timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS "session_userId_idx" ON "session" ("userId");
CREATE INDEX IF NOT EXISTS "account_userId_idx" ON "account" ("userId");
CREATE INDEX IF NOT EXISTS "verification_identifier_idx" ON "verification" ("identifier");
