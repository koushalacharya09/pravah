"""Kahn's algorithm and greedy cycle breaking. No NetworkX in the core."""

from __future__ import annotations

from collections.abc import Sequence

from pravah.ingest.models import DroppedEdge, PrecedenceEdge


def kahn_topological_sort(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Stable topological order, plus the nodes left inside a cycle.

    WHAT: A topological order is a list where for every edge ``u -> v``, ``u``
    appears before ``v``. Kahn's algorithm repeatedly removes a node whose
    remaining in-degree is 0. This implementation removes, each step, the
    zero-in-degree node that appears earliest in ``nodes``. Nodes still
    blocked when no such node remains are the cyclic remainder, in input order.

    WHY: Wide-sheet stage columns need one total order before they can be
    melted into an event log. The earliest-column tie-break is deterministic:
    the same sheet always yields the same order, and a founder can see that
    ties follow the sheet from left to right.

    WHY NOT: A depth-first topological sort is the usual alternative. It also
    runs in linear time, but the order depends on recursion order and is
    harder to explain when two stages have no direct link. A binary heap would
    give the same order in ``O((V+E) log V)``; stage graphs have fewer than
    about 30 columns, so a linear scan is simpler and just as stable.

    ASSUMPTIONS: ``nodes`` contains every endpoint that should appear in the
    order. Duplicate edges count once. A self-loop is a cycle.

    COMPLEXITY: ``O(V^2 + E)`` time because each of the ``V`` removals scans
    the node list, and ``O(V + E)`` space. ``V`` is the number of stage columns.

    FAILS WHEN: The graph has a cycle. The cyclic nodes are then returned in
    the second list and are absent from the order. Callers that need a total
    order must break the cycle first. Two valid orders that differ only by a
    tie are collapsed to one; the collapsed order is not more true, only stable.
    """
    ordered_nodes = list(dict.fromkeys(nodes))
    for source, target in edges:
        if source not in ordered_nodes:
            ordered_nodes.append(source)
        if target not in ordered_nodes:
            ordered_nodes.append(target)

    indegree = {node: 0 for node in ordered_nodes}
    adjacency: dict[str, list[str]] = {node: [] for node in ordered_nodes}
    seen_edges: set[tuple[str, str]] = set()
    for source, target in edges:
        if (source, target) in seen_edges:
            continue
        seen_edges.add((source, target))
        adjacency[source].append(target)
        indegree[target] += 1

    remaining = set(ordered_nodes)
    order: list[str] = []
    while True:
        ready = [node for node in ordered_nodes if node in remaining and indegree[node] == 0]
        if not ready:
            break
        chosen = ready[0]
        remaining.remove(chosen)
        order.append(chosen)
        for target in adjacency[chosen]:
            indegree[target] -= 1

    cyclic = [node for node in ordered_nodes if node in remaining]
    return order, cyclic


def resolve_precedence_order(
    nodes: Sequence[str],
    edges: Sequence[PrecedenceEdge],
) -> tuple[list[str], list[DroppedEdge]]:
    """Drop the weakest edge in a cycle until Kahn's algorithm succeeds.

    WHAT: While the graph has a cyclic remainder, delete one edge whose both
    ends sit in that remainder. The deleted edge is the minimum of
    ``(probability, support, source, target)``. Repeat, then return Kahn's
    order on the surviving directed acyclic graph.

    WHY: A handful of backdated cells can put three stage columns into a
    cycle. Refusing the whole file would throw away a usable stage order. The
    weakest edge is the claim the dates support least, which is the one a
    person would cross out first.

    WHY NOT: The minimum feedback arc set removes the smallest set of edges
    that leaves a DAG. It is NP-hard, and it optimises the count of removals
    rather than the quality of the claim removed: it can delete one strong
    edge instead of one weak edge. Failing the upload is the other
    alternative; it is rejected because the dropped edge is reported and the
    rest of the order is still evidence.

    ASSUMPTIONS: Edge probability is the right measure of weakness, and a
    lower support is weaker when probabilities tie. The true stage relation
    is nearly a chain; the cycle is noise, not a real loop in the business.

    COMPLEXITY: Each pass is ``O(V^2 + E)``, and each pass deletes one edge,
    so the total is ``O(E (V^2 + E))`` time and ``O(V + E)`` space.

    FAILS WHEN: The true process is cyclic (a stage genuinely repeats) or the
    true edge is slightly weaker than the false ones. The true edge is then
    the one removed. The removal is always listed in the result so a person
    can put it back.
    """
    current: list[PrecedenceEdge] = list(edges)
    dropped: list[DroppedEdge] = []
    for _ in range(len(edges) + 1):
        order, cyclic = kahn_topological_sort(
            nodes, [(edge.source, edge.target) for edge in current]
        )
        if not cyclic:
            return order, dropped
        cyclic_nodes = set(cyclic)
        inside = [
            edge
            for edge in current
            if edge.source in cyclic_nodes and edge.target in cyclic_nodes
        ]
        if not inside:
            return order, dropped
        weakest = min(inside, key=lambda edge: (edge.probability, edge.support, edge.source, edge.target))
        current = [edge for edge in current if edge is not weakest]
        dropped.append(
            DroppedEdge(
                source=weakest.source,
                target=weakest.target,
                probability=weakest.probability,
                support=weakest.support,
                lower=weakest.lower,
                upper=weakest.upper,
                reason=(
                    "Removed to break a cycle. This was the lowest precedence "
                    "inside the cyclic part of the stage graph."
                ),
            )
        )
    order, _cyclic = kahn_topological_sort(
        nodes, [(edge.source, edge.target) for edge in current]
    )
    return order, dropped
