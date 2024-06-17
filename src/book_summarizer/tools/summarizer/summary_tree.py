import networkx as nx
import numpy as np


class SummaryTree(nx.DiGraph):

    def __init__(self, change_point_detection_graph: nx.DiGraph):

        for attr, value in vars(change_point_detection_graph).items():
            setattr(self, attr, value)

        self.connected_trees = list(nx.weakly_connected_components(self))

        self.summary_path = self.compute_summary_path()

    @property
    def heads(self):
        heads = []

        for nodes in self.connected_trees:
            # Find the node with the highest level
            head = max(nodes, key=lambda node: self.nodes[node]["pen"])
            heads.append(head)

        return heads

    def compute_summary_path(self):

        summary_order = 0
        summary_path = []

        for head in self.heads:
            self.nodes[head]["level"] = 0
            summary_order = self._get_node_order(head, summary_order, summary_path, level=0)
            summary_order += 1
            self.nodes[head]["summary_order"] = summary_order
            summary_path.append(head)

        # Return the summary path sorted by summary order
        summary_path = sorted(
            summary_path, key=lambda node: self.nodes[node]["summary_order"]
        )

        return summary_path

    def _get_node_order(self, node, order, summary_path, level):
        child_nodes = list(self.predecessors(node))

        # Sort child nodes by order
        orders = [self.nodes[child]["order"] for child in child_nodes]
        child_nodes = [child for _, child in sorted(zip(orders, child_nodes))]

        level += 1
        for child in child_nodes:
            self.nodes[child]["level"] = level
            order = self._get_node_order(child, order, summary_path, level) + 1
            self.nodes[child]["summary_order"] = order
            summary_path.append(child)

        return order

    def compute_context_graph(self):
        pass
