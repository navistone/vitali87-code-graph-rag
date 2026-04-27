import cgr


class TestCgrShimExports:
    def test_all_symbols_importable(self) -> None:
        for name in cgr.__all__:
            assert hasattr(cgr, name), f"{name!r} listed in __all__ but not importable"

    def test_all_matches_module_exports(self) -> None:
        public_attrs = {k for k in vars(cgr) if not k.startswith("_")}
        assert set(cgr.__all__) == public_attrs

    def test_settings_is_canonical_instance(self) -> None:
        from codebase_rag.config import settings

        assert cgr.settings is settings

    def test_embed_code_is_canonical_function(self) -> None:
        from codebase_rag.embedder import embed_code

        assert cgr.embed_code is embed_code

    def test_embed_query_is_canonical_function(self) -> None:
        from codebase_rag.embedder import embed_query

        assert cgr.embed_query is embed_query

    def test_graph_loader_is_canonical_class(self) -> None:
        from codebase_rag.graph_loader import GraphLoader

        assert cgr.GraphLoader is GraphLoader

    def test_load_graph_is_canonical_function(self) -> None:
        from codebase_rag.graph_loader import load_graph

        assert cgr.load_graph is load_graph

    def test_memgraph_ingestor_is_canonical_class(self) -> None:
        from codebase_rag.services.ladybug_ingestor import LadybugIngestor

        # graph_service re-exports LadybugIngestor as MemgraphIngestor for
        # backward compatibility; cgr exposes it under the legacy name.
        assert cgr.MemgraphIngestor is LadybugIngestor

    def test_cypher_generator_is_canonical_class(self) -> None:
        from codebase_rag.services.llm import CypherGenerator

        assert cgr.CypherGenerator is CypherGenerator
