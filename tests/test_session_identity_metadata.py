"""Regression coverage for the session-identity metadata extraction bug.

_safe_attr() is a logging helper that str()-converts whatever it fetches.
Chaining it — _safe_attr(_safe_attr(ctx, "job"), "metadata") — calls getattr
on the *stringified* job object, which always fails and silently falls back
to the "n/a" default. _metadata_debug_entries_from_context (which feeds
identity_from_metadata for BOTH voice engines) and the room-name derivation
used to route through exactly this pattern, so every session's metadata
resolved to the literal string "n/a" instead of real job/room/participant
metadata — collapsing every guest session into one shared identity and
silently dropping the signed-in user id carried via SESSION_IDENTITY_SHARED_SECRET.

These tests pin the fixed (plain-getattr) behavior directly against
JobContext-shaped fakes, and confirm identity_from_metadata resolves a real
signed-in id end-to-end through the fixed extraction.
"""

import json
import types
import unittest

import agent
from memory_layer import identity_from_metadata


def _ctx(*, job_metadata=None, room_metadata=None, room_name=None,
         participant_metadata=None, remote_participants=None):
    return types.SimpleNamespace(
        job=types.SimpleNamespace(
            metadata=job_metadata,
            participant=types.SimpleNamespace(metadata=participant_metadata) if participant_metadata else None,
            participant_info=None,
        ),
        room=types.SimpleNamespace(
            metadata=room_metadata,
            name=room_name,
            remote_participants=remote_participants or {},
        ),
    )


class MetadataDebugEntriesTests(unittest.TestCase):
    def test_job_metadata_is_extracted_verbatim(self):
        entries = agent._metadata_debug_entries_from_context(
            _ctx(job_metadata=json.dumps({"clerk_user_id": "user_1"}))
        )
        by_label = dict(entries)
        self.assertEqual(by_label["job.metadata"], json.dumps({"clerk_user_id": "user_1"}))
        self.assertNotEqual(by_label["job.metadata"], "n/a")

    def test_room_metadata_is_extracted_verbatim(self):
        entries = agent._metadata_debug_entries_from_context(
            _ctx(room_metadata={"guest_id": "g-1"})
        )
        by_label = dict(entries)
        self.assertEqual(by_label["room.metadata"], {"guest_id": "g-1"})

    def test_absent_metadata_is_none_not_the_string_n_a(self):
        entries = agent._metadata_debug_entries_from_context(_ctx())
        for _, value in entries:
            self.assertIsNone(value)
            self.assertNotEqual(value, "n/a")

    def test_participant_metadata_is_extracted(self):
        entries = agent._metadata_debug_entries_from_context(
            _ctx(participant_metadata=json.dumps({"clerk_user_id": "user_2"}))
        )
        by_label = dict(entries)
        self.assertEqual(by_label["job.participant.metadata"], json.dumps({"clerk_user_id": "user_2"}))

    def test_remote_participant_metadata_is_extracted(self):
        participant = types.SimpleNamespace(metadata=json.dumps({"guest_id": "g-2"}))
        entries = agent._metadata_debug_entries_from_context(
            _ctx(remote_participants={"p1": participant})
        )
        by_label = dict(entries)
        self.assertEqual(by_label["room.remote_participants.p1.metadata"], json.dumps({"guest_id": "g-2"}))

    def test_missing_job_and_room_does_not_raise(self):
        entries = agent._metadata_debug_entries_from_context(types.SimpleNamespace())  # must not raise
        self.assertTrue(len(entries) >= 2)
        for _, value in entries:
            self.assertIsNone(value)


class RoomNameFromContextTests(unittest.TestCase):
    def test_returns_real_room_name(self):
        self.assertEqual(agent._room_name_from_context(_ctx(room_name="lucy-8c3a851498")), "lucy-8c3a851498")

    def test_none_when_room_missing(self):
        self.assertIsNone(agent._room_name_from_context(types.SimpleNamespace()))

    def test_none_when_name_blank(self):
        self.assertIsNone(agent._room_name_from_context(_ctx(room_name="   ")))

    def test_no_longer_returns_the_literal_n_a_placeholder(self):
        # The bug this regresses: _safe_attr(_safe_attr(ctx, "room"), "name")
        # always evaluated to "n/a", so every guest session collapsed into
        # the same identity. A present, real room name must round-trip.
        self.assertNotEqual(agent._room_name_from_context(_ctx(room_name="room-42")), "n/a")


class EndToEndIdentityResolutionTests(unittest.TestCase):
    """The full chain: ctx -> debug entries -> candidates -> identity_from_metadata."""

    def _candidates(self, ctx):
        entries = agent._metadata_debug_entries_from_context(ctx)
        return [value for _, value in entries if value]

    def test_signed_in_user_id_resolves_from_job_metadata(self):
        ctx = _ctx(job_metadata=json.dumps({"clerk_user_id": "user_42"}), room_name="room-1")
        identity = identity_from_metadata(self._candidates(ctx), fallback_guest_id=agent._room_name_from_context(ctx))
        self.assertEqual(identity.scope, "account")
        self.assertEqual(identity.clerk_user_id, "user_42")

    def test_two_different_guest_rooms_get_different_identities(self):
        # Before the fix, room name always resolved to "n/a" for every
        # session, so any two anonymous sessions collapsed into one shared
        # guest identity regardless of which room they were in.
        ctx_a = _ctx(room_name="room-a")
        ctx_b = _ctx(room_name="room-b")
        identity_a = identity_from_metadata(self._candidates(ctx_a), fallback_guest_id=agent._room_name_from_context(ctx_a))
        identity_b = identity_from_metadata(self._candidates(ctx_b), fallback_guest_id=agent._room_name_from_context(ctx_b))
        self.assertEqual(identity_a.guest_id, "room-a")
        self.assertEqual(identity_b.guest_id, "room-b")
        self.assertNotEqual(identity_a.key, identity_b.key)


if __name__ == "__main__":
    unittest.main()
