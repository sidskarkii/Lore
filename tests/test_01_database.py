"""Test SQLite chat persistence — sessions, messages, search, settings."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from tutorialvault.core.database import Database


def test_database():
    # Use a temp file so we don't pollute the real DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    print("1. Database created")

    # Create session
    session = db.create_session(title="Test Chat", provider="kilo", model="kilo-auto/free")
    sid = session["id"]
    print(f"2. Session created: id={sid}, title={session['title']}")
    assert session["title"] == "Test Chat"
    assert session["provider"] == "kilo"

    # Add messages
    m1 = db.add_message(sid, "user", "How do I rig a character in Blender?")
    print(f"3. User message added: id={m1['id']}")

    m2 = db.add_message(sid, "assistant", "To rig a character, first create an armature...",
                        sources=[{"collection_display": "Blender 101", "timestamp": "05:30"}])
    print(f"4. Assistant message added: id={m2['id']}, sources={len(m2['sources'])}")

    m3 = db.add_message(sid, "user", "What about weight painting?")
    m4 = db.add_message(sid, "assistant", "Weight painting controls bone influence...")
    print(f"5. Two more messages added")

    # Get session with messages
    full = db.get_session(sid)
    assert len(full["messages"]) == 4
    assert full["messages"][0]["role"] == "user"
    assert full["messages"][1]["sources"][0]["timestamp"] == "05:30"
    print(f"6. Session retrieved: {len(full['messages'])} messages, sources intact")

    # List sessions
    sessions = db.list_sessions()
    assert len(sessions) >= 1
    assert sessions[0]["message_count"] == 4
    print(f"7. List sessions: {len(sessions)} session(s), message_count={sessions[0]['message_count']}")

    # Rename
    db.update_session_title(sid, "Rigging Tutorial Chat")
    renamed = db.get_session(sid)
    assert renamed["title"] == "Rigging Tutorial Chat"
    print(f"8. Renamed to: {renamed['title']}")

    # Search messages
    results = db.search_messages("weight painting")
    assert len(results) >= 1
    print(f"9. Search 'weight painting': {len(results)} result(s)")

    # Settings
    db.set_setting("active_provider", "kilo")
    db.set_setting("theme", {"mode": "dark", "accent": "blue"})
    assert db.get_setting("active_provider") == "kilo"
    assert db.get_setting("theme")["mode"] == "dark"
    assert db.get_setting("nonexistent", "fallback") == "fallback"
    print(f"10. Settings: get/set/default all work")

    # Create second session
    s2 = db.create_session(title="Another Chat")
    db.add_message(s2["id"], "user", "test message")
    sessions = db.list_sessions()
    assert len(sessions) == 2
    print(f"11. Second session created, total: {len(sessions)}")

    # Delete first session (should cascade delete messages)
    db.delete_session(sid)
    assert db.get_session(sid) is None
    sessions = db.list_sessions()
    assert len(sessions) == 1
    print(f"12. Deleted session, cascade removed messages, remaining: {len(sessions)}")

    db.close()
    os.unlink(db_path)
    print("\n=== ALL DATABASE TESTS PASSED ===")


if __name__ == "__main__":
    test_database()
