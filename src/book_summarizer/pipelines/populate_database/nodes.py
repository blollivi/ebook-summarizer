import networkx as nx

from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from requests import post


def build_payload(
    hierarchical_summary: dict, summary_tree: SummaryTree, global_summary: dict
) -> dict:
    tree_nodes = []
    tree_edges = []
    global_head = [0, 0]
    list_heads = []
    for head_str, nodes_loader in hierarchical_summary.items():
        nodes = nodes_loader()
        # Refactor in a dict with (start, end) as key

        head = list(map(int, head_str.split("-")))
        list_heads.append(head)
        global_head = [min(global_head[0], head[0]), max(global_head[1], head[1])]

        bfs_edges = list(nx.bfs_edges(summary_tree.graph, tuple(head)))

        # Convert all tuples to list
        bfs_edges = [
            [[edge[0][0], edge[0][1]], [edge[1][0], edge[1][1]]] for edge in bfs_edges
        ]

        tree_nodes += nodes
        tree_edges += list(bfs_edges)

    global_edges = [[global_head, head] for head in list_heads]

    tree_nodes.append(
        dict(
            start=global_head[0],
            end=global_head[1],
            title="Global Summary",
            summary=global_summary["summary"],
        )
    )

    tree_edges += global_edges

    return dict(
        tree_nodes=tree_nodes,
        tree_edges=tree_edges,
    )


def make_post_request(url: str, payload: dict):
    # Make a POST request to the API
    response = post(url, json=payload)
    print(response.status_code)
    print(response.json())
    return response.json()
