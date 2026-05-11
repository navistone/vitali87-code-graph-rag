"""BUC-1617: wildcard consumer-side re-export resolution.

BUC-1610 (PR #16) shipped explicit re-export chain walking — given
``export { X } from './m'`` at every hop, a consumer reaches the
original definition.  It also captured the *wildcard* form
(``export * from './m'``) at the schema level via the ``*<target>``
sentinel in ``re_export_mapping``, but the chain walker did not fan
out across those sentinels — so a consumer doing
``import { helper } from './barrel'`` where the barrel only does
``export * from './utils'`` dead-ended at the barrel.

These tests pin the BUC-1617 behaviour:

* Single wildcard hop resolves to the leaf definition
* Two wildcard hops resolve
* Cyclic wildcards terminate without infinite-looping
* Mixed barrels (explicit re-export + ``export *``) resolve both names
* Python equivalent (``from .x import *`` in ``__init__.py``) resolves
* The 16-hop budget is shared with the named-chain ceiling
* Provenance: wildcard-traversed chains are tagged ``"wildcard"`` so
  downstream confidence filters can deprioritize them relative to a
  fully-named chain

Like ``test_reexports_resolution.py``, we drive ``ImportProcessor`` +
``CallResolver`` directly with a minimal ``function_registry`` so the
assertions are sharp and independent of the full ``GraphUpdater``
end-to-end pipeline.
"""

from __future__ import annotations

from pathlib import Path

from codebase_rag.parsers.call_resolver import (
    CONFIDENCE_EXACT,
    CONFIDENCE_WILDCARD,
    RESOLVED_VIA_EXACT,
    RESOLVED_VIA_WILDCARD,
    CallResolver,
)
from codebase_rag.parsers.import_processor import ImportProcessor


class _FakeRegistry:
    """Minimal stand-in for FunctionRegistryTrie.

    Mirrors the helper used in ``test_reexports_resolution.py`` — the
    chain walker only needs ``__contains__`` / ``__getitem__`` plus a
    no-op ``find_ending_with`` so the broader CallResolver doesn't trip
    if any unrelated codepath touches the trie.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def __contains__(self, qn: str) -> bool:
        return qn in self._mapping

    def __getitem__(self, qn: str) -> str:
        return self._mapping[qn]

    def get(self, qn: str, default: str | None = None) -> str | None:
        return self._mapping.get(qn, default)

    def find_ending_with(self, _suffix: str) -> list[str]:
        return []


def _make_resolver(
    project_name: str,
    repo_path: Path,
    import_mapping: dict[str, dict[str, str]],
    re_export_mapping: dict[str, dict[str, str]],
    registry: _FakeRegistry,
) -> CallResolver:
    proc = ImportProcessor(repo_path=repo_path, project_name=project_name)
    proc.import_mapping = import_mapping
    proc.re_export_mapping = re_export_mapping
    return CallResolver(
        function_registry=registry,  # type: ignore[arg-type]
        import_processor=proc,
        type_inference=None,  # type: ignore[arg-type]
        class_inheritance={},
    )


# --------------------------------------------------------------------------
# Single wildcard hop — the headline BUC-1617 case
# --------------------------------------------------------------------------


def test_should_resolve_through_wildcard_barrel_when_consumer_imports_named_symbol(
    tmp_path: Path,
) -> None:
    """``consumer -> barrel (export * from utils) -> utils::helper``.

    The barrel has no explicit re-export of ``helper`` — only the
    wildcard sentinel ``*proj.utils``.  Pre-BUC-1617 this dead-ended at
    ``proj.barrel.helper`` because the chain walker only consulted
    named entries; we now fan out across wildcard sentinels at each hop.
    """

    registry = _FakeRegistry({"proj.utils.helper": "Function"})
    import_mapping = {
        # `import { helper } from './barrel'`
        "proj.consumer": {"helper": "proj.barrel.helper"},
        # `export * from './utils'` registers the wildcard sentinel in
        # both maps (see ImportProcessor._parse_js_reexport).
        "proj.barrel": {"*proj.utils": "proj.utils"},
    }
    re_export_mapping = {
        "proj.barrel": {"*proj.utils": "proj.utils"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("helper", "proj.consumer", None)

    assert result is not None
    _node_type, qn = result
    assert qn == "proj.utils.helper", (
        f"Expected wildcard chain to resolve to the leaf, got {qn}"
    )


def test_should_tag_wildcard_chain_resolution_with_wildcard_provenance(
    tmp_path: Path,
) -> None:
    """The provenance dispatcher must downgrade a wildcard-traversed
    chain to ``("wildcard", 0.5)``.

    Named-chain resolutions stay ``("exact", 1.0)``; only chains that
    actually crossed a ``*`` sentinel are deprioritized, so a consumer
    of the CALLS graph that filters by ``min_confidence`` can tell the
    two cases apart.
    """

    registry = _FakeRegistry({"proj.utils.helper": "Function"})
    import_mapping = {
        "proj.consumer": {"helper": "proj.barrel.helper"},
        "proj.barrel": {"*proj.utils": "proj.utils"},
    }
    re_export_mapping = {
        "proj.barrel": {"*proj.utils": "proj.utils"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    tagged = resolver.resolve_function_call_with_provenance(
        "helper", "proj.consumer"
    )

    assert tagged is not None
    assert tagged.callee_qn == "proj.utils.helper"
    assert tagged.resolved_via == RESOLVED_VIA_WILDCARD
    assert tagged.confidence == CONFIDENCE_WILDCARD


# --------------------------------------------------------------------------
# Two wildcard hops
# --------------------------------------------------------------------------


def test_should_resolve_through_two_wildcard_barrels_when_chain_depth_is_two(
    tmp_path: Path,
) -> None:
    """``consumer -> barrel1 (export * from barrel2) -> barrel2 (export * from utils) -> utils::helper``.

    Two wildcard hops: nothing along the way has an explicit re-export
    for ``helper``, so the only path is wildcard-fanout at each hop.
    """

    registry = _FakeRegistry({"proj.utils.helper": "Function"})
    import_mapping = {
        "proj.consumer": {"helper": "proj.barrel1.helper"},
        "proj.barrel1": {"*proj.barrel2": "proj.barrel2"},
        "proj.barrel2": {"*proj.utils": "proj.utils"},
    }
    re_export_mapping = {
        "proj.barrel1": {"*proj.barrel2": "proj.barrel2"},
        "proj.barrel2": {"*proj.utils": "proj.utils"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("helper", "proj.consumer", None)

    assert result is not None
    _node_type, qn = result
    assert qn == "proj.utils.helper"


# --------------------------------------------------------------------------
# Cyclic wildcards must terminate
# --------------------------------------------------------------------------


def test_should_terminate_when_wildcard_chain_cycles_back_on_itself(
    tmp_path: Path,
) -> None:
    """``barrelA: export * from './barrelB'``, ``barrelB: export * from './barrelA'``.

    The cycle must be detected and the walk must return ``None``
    instead of looping.  The shared visited-set keyed on full qn is
    what guarantees termination — every candidate qn the walker
    constructs (``barrelB.thing``, ``barrelA.thing``) is recorded
    before recursing, so the second visit short-circuits cleanly.
    """

    registry = _FakeRegistry({})  # nothing concrete to land on
    import_mapping = {
        "proj.consumer": {"thing": "proj.barrelA.thing"},
        "proj.barrelA": {"*proj.barrelB": "proj.barrelB"},
        "proj.barrelB": {"*proj.barrelA": "proj.barrelA"},
    }
    re_export_mapping = {
        "proj.barrelA": {"*proj.barrelB": "proj.barrelB"},
        "proj.barrelB": {"*proj.barrelA": "proj.barrelA"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("thing", "proj.consumer", None)

    assert result is None, "Cyclic wildcard chain should return None, not loop"


def test_should_terminate_when_wildcard_chain_self_loops(
    tmp_path: Path,
) -> None:
    """A barrel that does ``export * from './self'`` (degenerate) must
    not infinite-loop.  This is the tightest cycle the resolver can
    encounter and is a useful guard against off-by-one errors in the
    visited-set bookkeeping.
    """

    registry = _FakeRegistry({})
    import_mapping = {
        "proj.consumer": {"thing": "proj.barrel.thing"},
        "proj.barrel": {"*proj.barrel": "proj.barrel"},
    }
    re_export_mapping = {
        "proj.barrel": {"*proj.barrel": "proj.barrel"},
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("thing", "proj.consumer", None)
    assert result is None


# --------------------------------------------------------------------------
# Mixed barrel — explicit re-export AND ``export *``
# --------------------------------------------------------------------------


def test_should_resolve_both_named_and_wildcard_exports_when_barrel_mixes_them(
    tmp_path: Path,
) -> None:
    """Realistic barrel pattern:

    .. code-block:: ts

        // barrel.ts
        export { Foo } from './foo';   // explicit
        export * from './utils';        // wildcard

        // consumer.ts
        import { Foo, helper } from './barrel';

    Both names must resolve.  ``Foo`` goes through the named chain
    (``("exact", 1.0)``) and ``helper`` goes through the wildcard
    sentinel (``("wildcard", 0.5)``).
    """

    registry = _FakeRegistry(
        {
            "proj.foo.Foo": "Class",
            "proj.utils.helper": "Function",
        }
    )
    import_mapping = {
        "proj.consumer": {
            "Foo": "proj.barrel.Foo",
            "helper": "proj.barrel.helper",
        },
        "proj.barrel": {
            "Foo": "proj.foo.Foo",
            "*proj.utils": "proj.utils",
        },
    }
    re_export_mapping = {
        "proj.barrel": {
            "Foo": "proj.foo.Foo",
            "*proj.utils": "proj.utils",
        },
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    # Foo: explicit re-export → "exact"
    foo = resolver.resolve_function_call_with_provenance("Foo", "proj.consumer")
    assert foo is not None
    assert foo.callee_qn == "proj.foo.Foo"
    assert foo.resolved_via == RESOLVED_VIA_EXACT
    assert foo.confidence == CONFIDENCE_EXACT

    # helper: wildcard re-export → "wildcard"
    helper = resolver.resolve_function_call_with_provenance(
        "helper", "proj.consumer"
    )
    assert helper is not None
    assert helper.callee_qn == "proj.utils.helper"
    assert helper.resolved_via == RESOLVED_VIA_WILDCARD
    assert helper.confidence == CONFIDENCE_WILDCARD


def test_should_prefer_named_export_when_barrel_also_has_wildcard_for_same_name(
    tmp_path: Path,
) -> None:
    """Tie-break: when a barrel both names a symbol and wildcards a
    module that happens to contain the same name, the named hop wins
    (and stays ``"exact"``) — the wildcard path is only consulted when
    the named hop dead-ends.
    """

    registry = _FakeRegistry(
        {
            "proj.preferred.Foo": "Function",
            "proj.utils.Foo": "Function",
        }
    )
    import_mapping = {
        "proj.consumer": {"Foo": "proj.barrel.Foo"},
        "proj.barrel": {
            "Foo": "proj.preferred.Foo",  # named, wins
            "*proj.utils": "proj.utils",  # would also have Foo
        },
    }
    re_export_mapping = {
        "proj.barrel": {
            "Foo": "proj.preferred.Foo",
            "*proj.utils": "proj.utils",
        },
    }

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )
    tagged = resolver.resolve_function_call_with_provenance(
        "Foo", "proj.consumer"
    )
    assert tagged is not None
    assert tagged.callee_qn == "proj.preferred.Foo"
    assert tagged.resolved_via == RESOLVED_VIA_EXACT


# --------------------------------------------------------------------------
# Python __init__.py with ``from .x import *``
# --------------------------------------------------------------------------


def test_should_resolve_python_init_wildcard_when_consumer_imports_named_symbol(
    tmp_path: Path,
) -> None:
    """Python equivalent of the TS wildcard barrel:

    .. code-block:: python

        # mypkg/__init__.py
        from .submod import *

        # consumer.py
        from mypkg import helper

    ImportProcessor registers ``*<base_module>`` for the wildcard
    side in both ``import_mapping`` and ``re_export_mapping``
    (see ``_register_python_import``).  The chain walker must follow
    that sentinel to resolve ``mypkg.helper`` to ``mypkg.submod.helper``.
    """

    registry = _FakeRegistry({"pyproj.mypkg.submod.helper": "Function"})
    import_mapping = {
        # `from mypkg import helper` -> import_mapping['consumer']['helper'] = 'pyproj.mypkg.helper'
        "pyproj.consumer": {"helper": "pyproj.mypkg.helper"},
        # `from .submod import *` -> import_mapping['mypkg']['*pyproj.mypkg.submod'] = 'pyproj.mypkg.submod'
        "pyproj.mypkg": {
            "*pyproj.mypkg.submod": "pyproj.mypkg.submod",
        },
    }
    re_export_mapping = {
        "pyproj.mypkg": {
            "*pyproj.mypkg.submod": "pyproj.mypkg.submod",
        },
    }

    resolver = _make_resolver(
        "pyproj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("helper", "pyproj.consumer", None)
    assert result is not None
    _node_type, qn = result
    assert qn == "pyproj.mypkg.submod.helper"


# --------------------------------------------------------------------------
# Hop budget shared with the named-chain ceiling
# --------------------------------------------------------------------------


def test_should_respect_16_hop_ceiling_across_wildcard_fanout(
    tmp_path: Path,
) -> None:
    """A wildcard chain longer than the 16-hop budget must terminate.

    We build a linear chain of 20 barrels, each one re-exporting
    ``*<next>`` and nothing else, then a leaf module.  The walker
    should bail at the 16th hop and return ``None`` — burning hops on
    the fan-out branches counts against the same budget that named
    chains use.
    """

    chain_depth = 20
    registry = _FakeRegistry({"proj.leaf.helper": "Function"})

    import_mapping: dict[str, dict[str, str]] = {
        "proj.consumer": {"helper": "proj.barrel0.helper"},
    }
    re_export_mapping: dict[str, dict[str, str]] = {}
    for i in range(chain_depth):
        nxt = f"proj.barrel{i + 1}" if i + 1 < chain_depth else "proj.leaf"
        import_mapping[f"proj.barrel{i}"] = {f"*{nxt}": nxt}
        re_export_mapping[f"proj.barrel{i}"] = {f"*{nxt}": nxt}

    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )

    result = resolver._try_resolve_via_imports("helper", "proj.consumer", None)
    assert result is None, (
        "20-deep wildcard chain should exceed the 16-hop ceiling and bail"
    )


def test_should_resolve_when_wildcard_chain_is_exactly_at_ceiling(
    tmp_path: Path,
) -> None:
    """A chain of depth 3 (well under 16) resolves cleanly.  Sanity
    check that the budget bookkeeping doesn't off-by-one on short
    chains.
    """

    registry = _FakeRegistry({"proj.leaf.helper": "Function"})
    import_mapping = {
        "proj.consumer": {"helper": "proj.b0.helper"},
        "proj.b0": {"*proj.b1": "proj.b1"},
        "proj.b1": {"*proj.b2": "proj.b2"},
        "proj.b2": {"*proj.leaf": "proj.leaf"},
    }
    re_export_mapping = {
        "proj.b0": {"*proj.b1": "proj.b1"},
        "proj.b1": {"*proj.b2": "proj.b2"},
        "proj.b2": {"*proj.leaf": "proj.leaf"},
    }
    resolver = _make_resolver(
        "proj", tmp_path, import_mapping, re_export_mapping, registry
    )
    result = resolver._try_resolve_via_imports("helper", "proj.consumer", None)
    assert result is not None
    _node_type, qn = result
    assert qn == "proj.leaf.helper"
