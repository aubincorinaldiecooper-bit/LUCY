import os
import unittest

MIGRATION = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "db", "migrations", "004_better_auth_core.sql"
)


class BetterAuthMigrationTests(unittest.TestCase):
    def setUp(self):
        with open(MIGRATION, encoding="utf-8") as handle:
            self.sql = handle.read()

    def test_creates_all_four_core_tables(self):
        for table in ('"user"', '"session"', '"account"', '"verification"'):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", self.sql, table)

    def test_camelcase_columns_are_quoted(self):
        # Quoting preserves case so Better Auth's quoted queries match.
        for column in ('"emailVerified"', '"userId"', '"expiresAt"', '"accountId"', '"providerId"'):
            self.assertIn(column, self.sql, column)

    def test_foreign_keys_cascade_to_user(self):
        self.assertIn('REFERENCES "user" ("id") ON DELETE CASCADE', self.sql)

    def test_is_idempotent(self):
        # Re-running the migration must be safe (the runner applies once, but
        # IF NOT EXISTS guards a manual re-run too).
        self.assertNotIn("CREATE TABLE \"", self.sql)  # never an unguarded CREATE TABLE
        self.assertEqual(self.sql.count("CREATE TABLE IF NOT EXISTS"), 4)


if __name__ == "__main__":
    unittest.main()
