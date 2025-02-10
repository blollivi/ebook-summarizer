from typing import List, Union, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from ebooklib import epub
from pathlib import Path
import urllib.parse
from tqdm import tqdm
from sklearn.covariance import EllipticEnvelope
import ebooklib
import concurrent.futures

from .tools.chain import build_llm_chain


# Define the main tags we care about.
MAIN_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "p",
    "span",
    "a",
}


def list_epub_files_from_directory(directory: str) -> List[Path]:
    return list(Path(directory).rglob("*.epub"))


def _get_hrefs_from_toc(toc_item, hrefs=None):
    """
    Recursively extract all hrefs from a TOC item.

    Args:
        toc_item: A TOC item or list of TOC items
        hrefs: List of all hrefs found in the TOC

    Returns:
        List of all items found in the TOC
    """
    if hrefs is None:
        hrefs = []

    if isinstance(toc_item, (list, tuple)):
        for sub_item in toc_item:
            _get_hrefs_from_toc(sub_item, hrefs)
    else:
        if isinstance(toc_item, epub.Link):
            href = toc_item.href.split("#")[0]
            href = urllib.parse.unquote(href)
            if href not in hrefs:
                hrefs.append(href)

    return hrefs


def parse_toc(toc: Union[List[epub.Link], List[Tuple[epub.Link, list]]]):
    pass


def get_items_from_book(book: epub.EpubBook) -> List[str]:
    hrefs = _get_hrefs_from_toc(book.toc)
    items = [book.get_item_with_href(href) for href in hrefs]
    return items


def concatenate_items_body(items: List[epub.EpubHtml]) -> str:
    """Parse items body with beautifulsoup and concatenate all child tags to a single body"""
    # Init beautifulsoup body tag
    global_body_tag = BeautifulSoup("<body></body>", "html.parser").body
    for item in items:
        content = item.content
        soup = BeautifulSoup(content, "html.parser")
        body = soup.find("body")
        if body is not None:
            # Add tags from body to the global body
            for child_tag in list(body.children):
                global_body_tag.append(child_tag)

    # Return the html string of the global body
    return str(global_body_tag)


def is_leaf_tag(tag) -> bool:
    """
    Returns True if the given tag does not contain any descendant that is one of the main tags.
    """
    # If any descendant (at any level) has a tag name in MAIN_TAGS, then tag is not a leaf.
    return tag.find(lambda child: child is not tag and child.name in MAIN_TAGS and len(child.content) > 0) is None


def extract_main_tags(html_str) -> pd.DataFrame:
    soup = BeautifulSoup(html_str, "html.parser")

    # Initialize stats using a defaultdict for automatic key creation.
    tags = []

    # Process each tag in MAIN_TAGS with a class attribute that is also a leaf tag.
    for tag in soup.find_all(lambda t: t.name in MAIN_TAGS):
        if not is_leaf_tag(tag):
            continue

        t = {}
        t["tag"] = tag.name
        classes = tag.get("class", [])
        if classes:
            t["class"] = classes[0]
        else:
            style = tag.get("style", "")
            if style:
                t["class"] = style
            else:
                t["class"] = "NA"

        t["text"] = tag.get_text(strip=True)
        t["text_length"] = len(t["text"])

        tags.append(t)

    return pd.DataFrame(tags)


def compute_tags_stats(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute various tag statistics aggregated over the book items.
    Also compute two new features:
      - avg_nb_section: average number of occurrences of a given tag/class combination per item.
      - section_proportion: proportion of items where the tag/class combination is found.
    """

    stats = tags_df.groupby(["tag", "class"]).agg(
        count=("text", "count"),
        total_length=("text_length", "sum"),
        avg_length=("text_length", "mean"),
    )

    stats["proportion"] = stats["count"] / tags_df.shape[0]
    stats["coverage"] = stats["total_length"] / tags_df["text_length"].sum()

    return stats

def extract_toc(book: epub.EpubBook) -> str:
    toc = book.toc

    def process_toc_item(item, indent=0):
        toc_str = ""
        if isinstance(item, tuple):
            toc_str += f"{'  ' * indent}{item[0].title} - {item[0].href}\n"
            for sub_item in item[1]:
                toc_str += process_toc_item(sub_item, indent + 1)
        else:
            toc_str += f"{'  ' * indent}{item.title} - {item.href}\n"
        return toc_str

    toc_string = ""
    for item in toc:
        toc_string += process_toc_item(item)
    
    return toc_string




def compute_tags_stats_all(directory: str) -> pd.DataFrame:
    list_epub_files = list_epub_files_from_directory(directory)
    tags_stats = []
    for epub_file in tqdm(list_epub_files):
        try:
            book = epub.read_epub(epub_file, options={"ignore_ncx": True})
# items = get_items_from_book(book)
            items = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
            body = concatenate_items_body(items)
            tags_df = extract_main_tags(body)
            # Concat the string content of "tag" and "class" columns to get a unique identifier for each tag.
            tags_df["tag_class"] = tags_df["tag"] + "_" + tags_df["class"]            
            ohe = pd.get_dummies(tags_df["tag_class"])

            # Multiply the one-hot-encoded tags by the text length to get the total length of each tag.
            ohe = ohe.mul(tags_df["text_length"].clip(0, 1000), axis=0)

            stats = compute_tags_stats(tags_df)
            stats["book"] = epub_file.stem
            tags_stats.append(stats)
        except Exception as e:
            print(f"Error processing {epub_file}: {e}")

    return pd.concat(tags_stats, ignore_index=True)

def plot_stats(stats):
    from plotly import express as px

    fig = px.scatter_matrix(
        stats,
        dimensions=["proportion", "coverage", "count", "avg_length"],
        color="tag",
        hover_data=["class", "book"],
    )
    fig.show()

    fig = px.strip(
        stats,
        y="total_length",
        x="count",
        hover_data=["class"],
    )

    fig.show()


def extract_chunks_from_items(items: List[epub.EpubHtml]):
    chunks = []

    for item in items:
        body = item.get_body_content().decode("utf-8")

        # Extract all lines from the body
        lines = body.split("\n")

        chunks.append(lines)

    return chunks


def build_book_from_chunks(chunks: List[List[str]]) -> str:
    book = "\n\n".join(
        ["\n".join([c.split(".")[0] for c in chunk]) for chunk in chunks]
    )
    # Save the book to a file
    with open("book.txt", "w") as f:
        f.write(book)
    return book
