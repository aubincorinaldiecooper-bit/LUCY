import os
from pathlib import Path


def main() -> int:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is required")
        return 1

    try:
        import psycopg
    except Exception:
        print("psycopg is required. Install with: pip install 'psycopg[binary]'")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    migrations_dir = repo_root / "db" / "migrations"
    if not migrations_dir.exists():
        print("No migrations directory found at db/migrations")
        return 0

    migration_files = sorted(p for p in migrations_dir.glob("*.sql") if p.is_file())
    if not migration_files:
        print("No migration files found")
        return 0

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  filename text PRIMARY KEY,
                  applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        conn.commit()

        for migration in migration_files:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM schema_migrations WHERE filename = %s",
                    (migration.name,),
                )
                if cur.fetchone():
                    print(f"Skipping already applied migration: {migration.name}")
                    continue

            sql = migration.read_text(encoding="utf-8")
            print(f"Applying migration: {migration.name}")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (migration.name,),
                )
            conn.commit()

    print("Migrations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
