import networkx as nx
from ebooklib import epub
from typing import Optional, Union, List, Tuple
from pyvis.network import Network

import networkx as nx
from ebooklib import epub
from typing import Optional, Union, List, Tuple
from pyvis.network import Network


class BookGraph(nx.DiGraph):
    """A directed graph representing book structure and metadata from EPUB files."""

    def __init__(self, toc: Union[List[epub.Link], List[Tuple[epub.Link, list]]]):
        super().__init__()
        self.toc = toc

    @classmethod
    def from_epub(cls, filepath: str) -> "BookGraph":
        """Create a BookGraph from an EPUB file."""
        book = epub.read_epub(filepath)
        graph = cls(book.toc)

        # Add root node with metadata
        root_id = "root"
        graph.add_node(
            root_id,
            title=cls._get_metadata(book, "title", "Unknown Title"),
            author=cls._get_metadata(book, "creator", "Unknown Author"),
            label=cls._get_metadata(book, "title", "Unknown Title"),  # for pyvis
            level=0,  # for hierarchical layout
            group=0,  # for color grouping if needed
        )

        # Build graph from table of contents
        if book.toc:
            cls._add_toc_items(graph, book.toc, parent=root_id, level=1)

        return graph

    @staticmethod
    def _get_metadata(book: epub.EpubBook, field: str, default: str) -> str:
        """Helper to safely extract metadata from EPUB."""
        metadata = book.get_metadata("DC", field)
        return metadata[0][0] if metadata else default

    @staticmethod
    def _add_toc_items(
        graph: nx.DiGraph,
        items: List[Union[epub.Link, Tuple[epub.Link, list]]],
        parent: Optional[str] = None,
        level: int = 1,
    ) -> None:
        """Recursively add TOC items to the graph with parent-child and followed-by relationships."""
        prev_sibling_id = None
        group_counter = 1  # for color grouping in pyvis, root is 0
        for item in items:
            if isinstance(item, tuple):
                section, children = item
                current_id = id(section)
                graph.add_node(
                    current_id,
                    title=section.title,
                    href=getattr(section, "href", None),
                    label=section.title,  # for pyvis
                    level=level,  # for hierarchical layout
                    group=group_counter,  # for color grouping if needed
                )
                if parent:
                    graph.add_edge(
                        parent, current_id, type="parent_of", label="parent_of"
                    )  # edge label for pyvis
                if prev_sibling_id is not None:
                    graph.add_edge(
                        prev_sibling_id,
                        current_id,
                        type="followed_by",
                        label="followed_by",
                    )  # edge label for pyvis
                prev_sibling_id = current_id
                BookGraph._add_toc_items(
                    graph, children, parent=current_id, level=level + 1
                )
                group_counter += 1  # increment group for next section
            else:
                current_id = id(item)
                graph.add_node(
                    current_id,
                    title=item.title,
                    href=item.href,
                    label=item.title,  # for pyvis
                    level=level,  # for hierarchical layout
                    group=group_counter,  # for color grouping if needed
                )
                if parent:
                    graph.add_edge(
                        parent, current_id, type="parent_of", label="parent_of"
                    )  # edge label for pyvis
                if prev_sibling_id is not None:
                    graph.add_edge(
                        prev_sibling_id,
                        current_id,
                        type="followed_by",
                        label="followed_by",
                    )  # edge label for pyvis
                prev_sibling_id = current_id
                group_counter += 1  # increment group for next section

    def plot_interactive_tree_html(self, output_filepath: str = "book_graph.html"):
        """
        Plots the BookGraph as an interactive tree in an HTML file.

        Args:
            output_filepath: The filepath to save the HTML file.
        """
        nt = Network(
            notebook=False, directed=True, layout="hierarchical"
        )  # Initialize pyvis network

        # Customize hierarchical layout
        nt.options.layout.hierarchical.direction = "UD"  # Up-Down direction for tree
        nt.options.layout.hierarchical.sortMethod = (
            "directed"  # Sort nodes in directed order
        )
        nt.options.layout.hierarchical.levelSeparation = 150  # Adjust vertical spacing
        nt.options.layout.hierarchical.nodeSpacing = 100  # Adjust horizontal spacing

        # Customize nodes and edges appearance and interactions
        nt.options.nodes = {  # Correct way to set node options
            "shape": "box",
            "widthConstraint": 200,
            "margin": 10,
        }
        nt.options.edges = {  # Correct way to set edge options
            "smooth": {"type": "cubicBezier", "forceDirection": "vertical"},
            "arrows": {"to": {"enabled": True}},
        }

        # Add nodes and edges from BookGraph to pyvis Network
        for node_id, node_data in self.nodes(data=True):
            nt.add_node(
                node_id,
                label=node_data.get("label", str(node_id)),  # Use label for display
                title=f"Title: {node_data.get('title', 'N/A')}<br>Href: {node_data.get('href', 'N/A')}",  # Tooltip
                level=node_data.get("level", None),  # Hierarchical level
                group=node_data.get("group", 0),  # Group for coloring
            )

        for u, v, edge_data in self.edges(data=True):
            nt.add_edge(
                u,
                v,
                title=f"Type: {edge_data.get('type', 'N/A')}",  # Edge tooltip
                label=edge_data.get("label", ""),  # Edge label
            )

        nt.save_graph(output_filepath)
        print(f"Interactive graph saved to {output_filepath}")
        print(
            f"Open {output_filepath} in your web browser to view the interactive book graph."
        )

    def get_leaf_nodes(self) -> List[str]:
        """Return leaf nodes from the subgraph of nodes connected by 'parent_of' edges."""
        parent_edges = [
            (u, v) for u, v, d in self.edges(data=True) if d["type"] == "parent_of"
        ]
        subgraph = self.edge_subgraph(parent_edges)

        leaf_nodes = [
            node for node in subgraph.nodes() if not list(subgraph.successors(node))
        ]

        return leaf_nodes

    def load_node_content(self, node_id: int) -> str:
        """Load the content of a node from the EPUB file."""
        node = self.nodes[node_id]

        if "href" not in node:
            return "No content found."
        else:
            pass
