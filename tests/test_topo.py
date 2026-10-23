"""Kahn's algorithm against a hand order and against NetworkX on the partial order."""

from __future__ import annotations

import networkx as nx

from pravah.ingest.models import PrecedenceEdge
from pravah.ingest.topo import kahn_topological_sort, resolve_precedence_order


def test_diamond_with_a_disconnected_node_has_one_stable_order() -> None:
    # A before B and C; B and C before D; E is free.
    # Ready nodes are always taken in the order they were given.
    nodes = ["A", "B", "C", "D", "E"]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    order, cyclic = kahn_topological_sort(nodes, edges)
    assert cyclic == []
    assert order == ["A", "B", "C", "D", "E"]


def test_self_loop_is_left_in_the_cyclic_remainder() -> None:
    # B never depends on A, so it is ordered. A's self-loop keeps its in-degree
    # at 1, so Kahn's algorithm cannot remove A. An extra A -> B edge would
    # trap B as well, because A is never removed to release it.
    order, cyclic = kahn_topological_sort(["A", "B"], [("A", "A")])
    assert order == ["B"]
    assert cyclic == ["A"]


def test_networkx_agrees_that_every_edge_points_forward() -> None:
    nodes = ["apply", "review", "offer", "join"]
    edges = [("apply", "review"), ("apply", "offer"), ("review", "join"), ("offer", "join")]
    order, cyclic = kahn_topological_sort(nodes, edges)
    assert cyclic == []
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(graph)
    position = {name: index for index, name in enumerate(order)}
    for source, target in edges:
        assert position[source] < position[target]
    # NetworkX's own order is a valid topological order, not necessarily ours.
    nx_order = list(nx.topological_sort(graph))
    nx_position = {name: index for index, name in enumerate(nx_order)}
    for source, target in edges:
        assert nx_position[source] < nx_position[target]


def test_weakest_cycle_edge_is_dropped_and_the_rest_orders() -> None:
    nodes = ["A", "B", "C", "D"]
    edges = [
        PrecedenceEdge("A", "B", 0.95, 20, 0.80, 0.99),
        PrecedenceEdge("B", "C", 0.90, 20, 0.75, 0.97),
        PrecedenceEdge("C", "A", 0.60, 20, 0.40, 0.77),
        PrecedenceEdge("C", "D", 0.85, 20, 0.70, 0.94),
    ]
    order, dropped = resolve_precedence_order(nodes, edges)
    assert [(edge.source, edge.target) for edge in dropped] == [("C", "A")]
    assert order == ["A", "B", "C", "D"]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    assert nx.is_directed_acyclic_graph(graph)


def test_equal_probability_drops_the_lower_support_edge() -> None:
    nodes = ["A", "B"]
    edges = [
        PrecedenceEdge("A", "B", 0.80, 30, 0.62, 0.91),
        PrecedenceEdge("B", "A", 0.80, 10, 0.49, 0.94),
    ]
    _order, dropped = resolve_precedence_order(nodes, edges)
    assert [(edge.source, edge.target) for edge in dropped] == [("B", "A")]
