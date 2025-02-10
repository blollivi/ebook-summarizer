import networkx as nx
from ebooklib import epub
from typing import Optional, Union, List, Tuple
from pyvis.network import Network

import networkx as nx
from ebooklib import epub
from typing import Optional, Union, List, Tuple
from pyvis.network import Network
from bs4 import BeautifulSoup


class BookGraph(nx.DiGraph):
    """A directed graph representing book structure and metadata from EPUB files."""

    def __init__(self, book: epub.EpubBook):
        super().__init__()
        self.book = book
        self.items = self._get_items_from_book()

    def _get_items_from_book(self) -> List[epub.EpubHtml]:
        """Get all items from the book."""
        items = self.book.get_items()
        return [item for item in items if isinstance(item, epub.EpubHtml)]

    @property
    def item_names(self) -> List[str]:
        """Return a list of names of the items using get_name()."""
        return [item.get_name() for item in self.items]

    @classmethod
    def from_epub(cls, filepath: str) -> "BookGraph":
        """Create a BookGraph from an EPUB file."""
        book = epub.read_epub(filepath)
        graph = cls(book)

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

    def get_items_html_content(items: List[epub.EpubHtml]) -> str:
        """Loop over all items, get their html content and concatenate all the bodies together."""
        # Init beautifulsoup body tag
        soup = BeautifulSoup(
            "<html><head><meta charset='utf-8'/></head><body></body></html>",
            "html.parser",
        )
        body_tag = soup.body

        # Loop over each document item and append its body contents
        for item in items:
            # Only process items that have HTML content
            try:
                content_html = item.get_body_content()
            except Exception:
                continue

            item_soup = BeautifulSoup(content_html, "html.parser")
            if item_soup.body:
                for element in item_soup.body.contents:
                    body_tag.append(element)

        return str(soup)

    def find_next_leaf_node(self, node_id):
        is_leaf = self.is_leaf_tag(node_id)

        # Check for a direct "followed_by" successor.
        for _, target, edge_data in self.out_edges(node_id, data=True):
            if edge_data.get("type") == "followed_by":
                return target
        # If none, recursively check the parent's "followed_by" edge.
        parents = [
            src
            for src, _, edge_data in self.in_edges(node_id, data=True)
            if edge_data.get("type") == "parent_of"
        ]
        if parents:
            return self.ind_next_node(parents[0])
        return None

    def subsections(self, node_id: int) -> List[int]:
        """Return successor nodes that are linked by 'parent_of' edges."""
        return [
            v
            for u, v, d in self.out_edges(node_id, data=True)
            if d["type"] == "parent_of"
        ]

    def get_node_text(self, node_id: int) -> str:
        """Get the text content of a node from the EPUB file."""
        node = self.nodes[node_id]

        is_leaf = self.is_leaf_node(node_id)
        if not is_leaf:
            # Find the first child node that is a leaf node
            first_child = self.subsections(node_id)[0]
            if self.is_leaf_node(first_child):
                next_node = first_child
            else:
                return self.get_node_text(first_child)

        else:
            next_node = self.find_next_node(node_id)

        href = node.get("href")
        next_href = self.nodes[next_node].get("href") if next_node else None

        if "#" in href:
            href, anchor = href.split("#")
        if "#" in next_href:
            next_href, next_anchor = next_href.split("#")

        item = self.book.get_item_with_href(href)
        body = item.get_body_content()
        soup = BeautifulSoup(body, "html.parser")
        anchor_tag = soup.find(id=anchor)

        if href == next_href:
            next_anchor_tag = soup.find(id=next_anchor)
        else:
            next_anchor_tag = None

        between_tags = get_tags_between(anchor_tag, next_anchor_tag)

        # Extract all html tags between the two anchors respectively included and excluded
        text = "".join(str(tag) for tag in between_tags)
        
        return text

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
                title=f"Node_id {node_id}",  # Tooltip
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

    def is_leaf_node(self, node_id) -> bool:
        """Return True if the node is a leaf node, ignoring the "followed_by" edges."""
        return not self.subsections(node_id)

    def load_node_content(self, node_id: int) -> str:
        """Load the content of a node from the EPUB file."""
        node = self.nodes[node_id]

        if "href" not in node:
            return "No content found."
        else:
            pass


def get_tags_between(start_tag, end_tag):
    """
    Given two BeautifulSoup tag objects (start_tag and end_tag), this function:
      1. Finds their lowest common ancestor.
      2. Identifies which direct children of that ancestor contain the start_tag and end_tag.
      3. Returns the list of direct children (relative to that common ancestor) that lie between
         the two children that enclose the start and end tags.

    If the two tags are siblings (i.e. they share the same parent), this function will return
    the siblings that lie between them.

    Parameters:
        start_tag (bs4.element.Tag): The BeautifulSoup tag representing the start point.
        end_tag (bs4.element.Tag):   The BeautifulSoup tag representing the end point.

    Returns:
        tuple: (common_ancestor, between_tags)
            - common_ancestor (bs4.element.Tag): The lowest common ancestor of start_tag and end_tag.
            - between_tags (list): A list of BeautifulSoup tag objects that are direct children of the common
              ancestor and that fall strictly between the direct children containing start_tag and end_tag.
              If none exist, an empty list is returned.

    Raises:
        ValueError: If no common ancestor is found or if the necessary direct children cannot be determined.
    """
    # 1. Get the list of all ancestors for each tag.
    start_ancestors = list(start_tag.parents)
    end_ancestors = list(end_tag.parents)

    # 2. Find the lowest common ancestor by iterating through start_tag's ancestors.
    common_ancestor = None
    for ancestor in start_ancestors:
        if ancestor in end_ancestors:
            common_ancestor = ancestor
            break
    if common_ancestor is None:
        raise ValueError("No common ancestor found for the provided tags.")

    # 3. Helper: Find which direct child of 'parent' contains 'tag'.
    def find_direct_child(tag, parent):
        for child in parent.children:
            # Only consider Tag objects (skip strings, comments, etc.)
            if not hasattr(child, "descendants"):
                continue
            # If the child is the tag or if the tag is anywhere inside the child's descendants, return it.
            if child == tag or tag in child.descendants:
                return child
        return None

    child_start = find_direct_child(start_tag, common_ancestor)
    child_end = find_direct_child(end_tag, common_ancestor)

    if child_start is None or child_end is None:
        raise ValueError(
            "Could not locate the direct children of the common ancestor that contain the start or end tag."
        )

    # 4. Get a list of direct children of the common ancestor that are Tag objects.
    direct_children = [
        child for child in common_ancestor.children if getattr(child, "name", None)
    ]

    try:
        index_start = direct_children.index(child_start)
        index_end = direct_children.index(child_end)
    except ValueError:
        raise ValueError(
            "The expected direct children were not found among the common ancestor's children."
        )

    # 5. Ensure the start tag appears before the end tag at this level.
    if index_start > index_end:
        # Option: swap if out of order, or raise an error.
        index_start, index_end = index_end, index_start

    # 6. The tags between are the ones strictly between these indices.
    between_tags = direct_children[index_start + 1 : index_end]

    return between_tags
