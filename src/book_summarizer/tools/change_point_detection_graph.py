import numpy as np
import ruptures as rpt
import networkx as nx
from plotly import graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


def get_segments_from_breakpoints(bkpts: np.array):
    segments = [(0, bkpts[0] - 1)]
    for i in range(1, len(bkpts)):
        segments.append((bkpts[i - 1], bkpts[i] - 1))
    return segments


def medoid(arr: np.array):
    dist_matrix = cosine_similarity(arr)
    idx_medoid = np.argmin(np.sum(dist_matrix, axis=1))
    return arr[idx_medoid]


def compute_centered_signal(signal: np.array, bkpts: np.array):
    segments = get_segments_from_breakpoints(bkpts)
    centered_signal = np.zeros_like(signal)

    for seg in segments:
        _signal = signal[seg[0] : seg[1] + 1]
        center = medoid(_signal)
        centered_signal[seg[0] : seg[1] + 1] = center
    return centered_signal


class ChangePointDetectionGraph(nx.DiGraph):

    def __init__(
        self, penalties: np.array = np.arange(2, 50, 0.25), denoise: bool = True
    ):
        self.penalties = penalties
        self.denoise = denoise

    def fit(self, signal: np.array):
        bkps = []
        centered_signal = signal
        for pen in self.penalties:
            model = rpt.KernelCPD(kernel="cosine", min_size=3).fit(centered_signal)
            bkps.append(model.predict(pen=pen))

            if self.denoise:
                centered_signal = compute_centered_signal(centered_signal, bkps[-1])

        bkpt_matrix = np.zeros((len(self.penalties), len(signal)))
        for i, _bkps in enumerate(bkps):
            for j in _bkps[:-1]:
                bkpt_matrix[i, j - 1] = 1

        self.bkpts = bkps
        self.bkpts_matrix = bkpt_matrix

        return self

    def compute_graph(self) -> nx.DiGraph:
        bkpts = self.bkpts

        penalties = self.penalties

        # Filter out duplicate penalty levels  that have the same breakpoints
        diff_matrix = np.abs(np.diff(self.bkpts_matrix, axis=0))
        transition_mask = np.sum(diff_matrix, axis=1) != 0
        transition_mask = np.insert(transition_mask, 0, True)

        bkpts = [bkpts[i] for i in range(len(bkpts)) if transition_mask[i]]
        penalties = penalties[transition_mask]

        G = nx.DiGraph()

        # Init first level segments
        first_level_segments = get_segments_from_breakpoints(bkpts[0])

        for segment in first_level_segments:
            G.add_node(segment, pen=penalties[0], start=segment[0], end=segment[1])

        # Loop over all penalty levels
        for penalty_idx in range(1, len(penalties)):

            previous_segments = get_segments_from_breakpoints(bkpts[penalty_idx - 1])
            current_segments = get_segments_from_breakpoints(bkpts[penalty_idx])

            for segment in current_segments:
                start, end = segment
                G.add_node(segment, pen=penalties[penalty_idx], start=start, end=end)

                child_order = 0
                for prev_segment in previous_segments:
                    prev_start, prev_end = prev_segment

                    if prev_start > end or prev_end < start:
                        continue

                    elif prev_start < end or prev_end < end:
                        # Update order of child
                        nx.set_node_attributes(
                            G, {prev_segment: {"order": child_order}}
                        )
                        child_order += 1
                        # Add edge, with weight equal to the size of intersection
                        G.add_edge(
                            prev_segment,
                            segment,
                            label="child_of",
                            weight=(min(end, prev_end) - max(start, prev_start))
                            / (end - start),
                        )

        # Add order atrtibute to the last level
        for order, segment in enumerate(current_segments):
            nx.set_node_attributes(G, {segment: {"order": order}})

        self.graph = G

        # Remove all edges with weight = 1
        edges_to_remove = []
        for edge in G.edges:
            if G.edges[edge]["weight"] == 1:
                edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)

        return G

    def plot_graph(self, chunks_df, node_size=50):
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
    start, end = segment
    return chunks_df.iloc[start:end]["Length"].sum()
