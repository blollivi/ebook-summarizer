import numpy as np
import ruptures as rpt
import networkx as nx
from plotly import graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


def get_segments_from_breakpoints(bkpts: np.array):
    """
    Convert a list of breakpoints into a list of segments.

    Args:
        bkpts (np.array): An array of breakpoint indices.

    Returns:
        list: A list of tuples, where each tuple represents a segment defined by its start and end indices.
    """
    segments = [(0, bkpts[0] - 1)]
    for i in range(1, len(bkpts)):
        segments.append((int(bkpts[i - 1]), int(bkpts[i] - 1)))
    return segments


def medoid(arr: np.array):
    """
    Calculate the medoid of a set of vectors.

    The medoid is the point in the set with the smallest average distance to all other points.

    Args:
        arr (np.array): A 2D array of vectors.

    Returns:
        np.array: The medoid vector.
    """
    dist_matrix = cosine_similarity(arr)
    idx_medoid = np.argmin(np.sum(dist_matrix, axis=1))
    return arr[idx_medoid]


def compute_centered_signal(signal: np.array, bkpts: np.array):
    """
    Center a signal based on the medoid of each segment defined by the breakpoints.

    Args:
        signal (np.array): The input signal.
        bkpts (np.array): An array of breakpoint indices.

    Returns:
        np.array: The centered signal.
    """
    segments = get_segments_from_breakpoints(bkpts)
    centered_signal = np.zeros_like(signal)

    for seg in segments:
        _signal = signal[seg[0] : seg[1] + 1]
        center = medoid(_signal)
        centered_signal[seg[0] : seg[1] + 1] = center
    return centered_signal


class SummaryTree:
    """
    A class to represent the hierarchical structure of a text using change point detection.

    This class uses a change point detection algorithm to identify segments in a signal
    (e.g., embeddings of text chunks) and builds a tree structure where each node represents
    a segment and the edges represent the hierarchical relationships between segments.
    """

    def __init__(
        self, penalties: np.array = np.arange(2, 50, 0.25), denoise: bool = True
    ):
        """
        Initialize the SummaryTree.

        Args:
            penalties (np.array, optional): An array of penalty values to use for change point detection.
                Defaults to np.arange(2, 50, 0.25).
            denoise (bool, optional): Whether to denoise the signal before change point detection.
                Defaults to True.
        """
        self.penalties = penalties
        self.denoise = denoise

    def fit(self, signal: np.array):
        """
        Fit the SummaryTree to the given signal.

        Args:
            signal (np.array): The input signal (e.g., embeddings of text chunks).

        Returns:
            SummaryTree: The fitted SummaryTree object.
        """
        self._find_change_points(signal)
        self.graph = self._build_tree()
        return self

    @property
    def heads(self):
        """
        Get the heads of the summary tree.

        Heads are nodes with no incoming edges, representing the top-level segments of the text.

        Returns:
            list: A list of head node IDs.
        """
        heads = [
            node_id
            for node_id in self.graph.nodes
            if self.graph.in_degree(node_id) == 0
        ]
        # Sort heads by order
        heads = sorted(heads, key=lambda node: self.graph.nodes[node]["order"])
        return heads

    def _sort_neighbors(self, neighbors):
        """
        Sort a list of neighbor nodes by their order attribute.

        Args:
            neighbors: A list of neighbor node IDs.

        Returns:
            list: The sorted list of neighbor node IDs.
        """
        neighbors = list(neighbors)
        orders = [self.graph.nodes[neighbor]["order"] for neighbor in neighbors]
        neighbors = [neighbor for _, neighbor in sorted(zip(orders, neighbors))]
        return neighbors

    def get_node_penalty_range(self, node_id):
        """
        Get the penalty range for a given node.

        The penalty range is defined by the penalty value of the node and the penalty value of its parent node.

        Args:
            node_id: The ID of the node.

        Returns:
            tuple: A tuple containing the minimum and maximum penalty values for the node.
        """
        node = self.graph.nodes[node_id]
        start = node["pen"]
        # Find parent node
        parent = list(self.graph.predecessors(node_id))
        assert len(parent) <= 1
        if len(parent) == 0:
            return start, np.inf
        parent = parent[0]
        end = self.graph.nodes[parent]["pen"]
        return start, end

    @property
    def summary_path(self):
        """
        Get the path through the summary tree for summarization.

        This path starts from the heads of the tree and traverses the tree in a depth-first
        post-order, ensuring that child nodes are summarized before their parents.

        Returns:
            list: A list of node IDs representing the summary path.
        """
        summary_path = []
        for head in self.heads:
            summary_path += list(
                nx.dfs_postorder_nodes(
                    self.graph,
                    source=head,
                    sort_neighbors=self._sort_neighbors,
                )
            )
        return summary_path

    @property
    def outline_path(self):
        """
        Get the path through the summary tree for outline generation.

        This path starts from the heads of the tree and traverses the tree in a depth-first
        pre-order, ensuring that parent nodes are visited before their children.

        Returns:
            list: A list of node IDs representing the outline path.
        """
        outline_path = []
        for head in self.heads:
            outline_path += list(
                nx.dfs_preorder_nodes(
                    self.graph,
                    source=head,
                    sort_neighbors=self._sort_neighbors,
                )
            )
        return outline_path

    def get_penalty_level_cut(self, penalty_level):
        """
        Get the nodes in the summary tree at a specific penalty level.

        Args:
            penalty_level (float): The penalty level to cut the tree at.

        Returns:
            list: A list of node IDs representing the nodes at the specified penalty level.
        """
        summary_path = self.summary_path
        penalty_ranges = [self.get_node_penalty_range(node) for node in summary_path]
        return [
            node
            for node, (min_penalty, max_penalty) in zip(summary_path, penalty_ranges)
            if min_penalty <= penalty_level < max_penalty
        ]

    def compute_node_level(self):
        """
        Compute the level of each node in the summary tree.

        The level of a node represents its depth in the tree, with the heads having a level of 0.
        """
        for head in self.heads:
            shortest_path = nx.shortest_path(self.graph, source=head)
            node_levels = {node: len(shortest_path[node]) - 1 for node in shortest_path}
            nx.set_node_attributes(self.graph, node_levels, "level")

    def _find_change_points(self, signal: np.array) -> None:
        """
        Find change points in the given signal using the specified penalty values.

        This method iterates through the penalty values, performing change point detection
        at each level and optionally denoising the signal between levels.

        Args:
            signal (np.array): The input signal.
        """
        bkps = []
        centered_signal = signal
        for pen in self.penalties:
            cost = rpt.costs.CostCosine()
            model = rpt.Binseg(model="cosine").fit(centered_signal)
            # model = rpt.KernelCPD(kernel="cosine", min_size=3).fit(centered_signal)
            bkps.append(model.predict(pen=pen))

            if self.denoise:
                centered_signal = compute_centered_signal(centered_signal, bkps[-1])

        bkpt_matrix = np.zeros((len(self.penalties), len(signal)))
        for i, _bkps in enumerate(bkps):
            for j in _bkps[:-1]:
                bkpt_matrix[i, j - 1] = 1

        self.bkpts = bkps
        self.bkpts_matrix = bkpt_matrix

    def _build_tree(self) -> None:
        """
        Build the summary tree from the detected change points.

        This method constructs a directed graph where each node represents a segment of the text
        and the edges represent the hierarchical relationships between segments.

        Returns:
            nx.DiGraph: The summary tree as a directed graph.
        """
        bkpts = self.bkpts

        penalties = self.penalties

        # Filter out duplicate penalty levels  that have the same breakpoints
        diff_matrix = np.abs(np.diff(self.bkpts_matrix, axis=0))
        transition_mask = np.sum(diff_matrix, axis=1) != 0
        transition_mask = np.insert(transition_mask, 0, True)

        bkpts = [bkpts[i] for i in range(len(bkpts)) if transition_mask[i]]
        penalties = penalties[transition_mask]

        # Init first level segments
        first_level_segments = get_segments_from_breakpoints(bkpts[0])

        G = nx.DiGraph()
        for segment in first_level_segments:
            G.add_node(segment, pen=penalties[0], start=segment[0], end=segment[1])

        # Loop over all penalty levels
        for penalty_idx in range(1, len(penalties)):

            previous_segments = get_segments_from_breakpoints(bkpts[penalty_idx - 1])
            current_segments = get_segments_from_breakpoints(bkpts[penalty_idx])

            for segment in current_segments:
                start, end = segment

                child_order = 0
                for prev_segment in previous_segments:
                    prev_start, prev_end = prev_segment

                    if (prev_start > end or prev_end < start) or (
                        prev_start == start and prev_end == end
                    ):
                        continue

                    elif prev_start < end or prev_end < end:
                        print(f"Segment {segment} intersects with {prev_segment}")
                        # Add new node
                        G.add_node(
                            segment, pen=penalties[penalty_idx], start=start, end=end
                        )
                        # Update order of child
                        nx.set_node_attributes(
                            G, {prev_segment: {"order": child_order}}
                        )
                        child_order += 1
                        # Add edge, with weight equal to the size of intersection
                        G.add_edge(
                            segment,
                            prev_segment,
                            label="summarized_from",
                            weight=(min(end, prev_end) - max(start, prev_start))
                            / (end - start),
                        )

        # Add order atrtibute to the last level
        for order, segment in enumerate(current_segments):
            nx.set_node_attributes(G, {segment: {"order": order}})

        # Remove all edges with weight = 1
        edges_to_remove = []
        for edge in G.edges:
            if G.edges[edge]["weight"] == 1:
                edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)

        return G

    def plot_graph(self, chunks_df, node_size=50):
        """
        Plot the summary tree using Plotly.

        Args:
            chunks_df (pd.DataFrame): Dataframe containing the text chunks.
            node_size (int, optional): The size of the nodes in the plot. Defaults to 50.
        """
        G = self.graph
        pos = {
            node: [
                (G.nodes[node]["start"] + G.nodes[node]["end"]) / 2,
                G.nodes[node]["pen"],
            ]
            for node in G.nodes
        }

        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in G.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}")
            node_colors.append(G.nodes[node]["pen"])
            node_sizes.append(count_words_in_segment(node, chunks_df))

        node_sizes = np.sqrt(node_sizes / np.max(node_sizes)) * node_size

        edge_traces = []
        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_width = G.edges[edge]["weight"] * 3

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge_width, color="rgba(0, 0, 0, 0.8)"),
                hoverinfo="none",
                mode="lines",
            )
            edge_traces.append(edge_trace)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale="Cividis",
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title="Penalty",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
                opacity=1,
            ),
            textposition="top center",
        )

        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                height=600,
                width=1000,
            ),
        )
        fig.show()


def count_words_in_segment(segment, chunks_df):
    """
    Count the total number of words in a segment of text chunks.

    Args:
        segment (tuple): A tuple representing the segment by its start and end indices.
        chunks_df (pd.DataFrame): Dataframe containing the text chunks.

    Returns:
        int: The total number of words in the segment.
    """
    start, end = segment
    return chunks_df.iloc[start:end]["Length"].sum()
