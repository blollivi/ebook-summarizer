from bs4 import BeautifulSoup
import numpy as np

class HTMLAnalyser:
    def __init__(self, html):
        self.soup = BeautifulSoup(html, "html.parser")
        self.structure = self.analyse_structure()

    def analyse_structure(self):
        def traverse(node):
            if not hasattr(node, "name") or node.name is None:
                return None

            children = [
                child
                for child in node.find_all()
                if hasattr(child, "name") and child.name is not None
            ]
            children_structure = [
                traverse(child) for child in children if traverse(child) is not None
            ]

            content_length = len(node.get_text(strip=True))
            num_children = len(children_structure)

            return {
                "tag": node.name,
                "num_children": num_children,
                "content_length": content_length,
                "children": children_structure,
            }

        return traverse(self.soup)

    def compute_lengths_sequence(self, level: int):
        lengths = []

        def traverse(node, current_level):
            children = node.get("children", [])
            if children and current_level < level:
                for child in children:
                    traverse(child, current_level + 1)
            else:
                lengths.append(node.get("content_length", 0))

        traverse(self.structure, 0)
        return lengths

