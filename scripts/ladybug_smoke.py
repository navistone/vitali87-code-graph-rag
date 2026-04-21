"""CI-2: LadybugDB round-trip smoke test (DEV-1170)."""
import tempfile
import os
import sys

import real_ladybug as lb

def run_smoke_test(db_path: str) -> None:
    print(f"Opening LadybugDB at: {db_path}")
    db = lb.Database(db_path)
    conn = lb.Connection(db)

    # Schema
    conn.execute("CREATE NODE TABLE IF NOT EXISTS SmokeTest(id INT64, name STRING, PRIMARY KEY (id))")
    print("✓ Node table created")

    # Insert
    conn.execute("CREATE (:SmokeTest {id: 1, name: 'hello'})")
    conn.execute("CREATE (:SmokeTest {id: 2, name: 'world'})")
    print("✓ Nodes inserted")

    # Query
    result = conn.execute("MATCH (n:SmokeTest) RETURN n.id, n.name ORDER BY n.id")
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    print(f"✓ Query returned {len(rows)} rows:")
    for row in rows:
        print(f"   id={row[0]}, name={row[1]}")

    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert rows[0][1] == "hello"
    assert rows[1][1] == "world"

    # Vector index support check
    try:
        conn.execute("CREATE NODE TABLE IF NOT EXISTS VecTest(id INT64, emb FLOAT[4], PRIMARY KEY (id))")
        conn.execute("CREATE (:VecTest {id: 1, emb: [0.1, 0.2, 0.3, 0.4]})")
        conn.execute("CALL CREATE_VECTOR_INDEX('vec_idx', 'VecTest', 'emb')")
        print("✓ Vector index created")
        result2 = conn.execute(
            "CALL QUERY_VECTOR_INDEX('vec_idx', 2, [0.1, 0.2, 0.3, 0.4]) RETURN node.id, distance"
        )
        hits = []
        while result2.has_next():
            hits.append(result2.get_next())
        print(f"✓ Vector search returned {len(hits)} hits: {hits}")
    except Exception as e:
        print(f"⚠ Vector index: {e} (may need LadybugDB >= 0.15 with vector support)")

    print("\n✅ LadybugDB smoke test PASSED")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        run_smoke_test(os.path.join(tmpdir, "smoke.db"))
