from typing import List, Union, Tuple
from bs4 import BeautifulSoup
import pandas as pd
from ebooklib import epub
from pathlib import Path
import urllib.parse
from tqdm import tqdm
from sklearn.covariance import EllipticEnvelope
import ebooklib

from .tools.chain import build_llm_chain


# Define the main tags we care about.
MAIN_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "div", "p"}


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
    return tag.find(lambda child: child is not tag and child.name in MAIN_TAGS) is None


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


def compute_tags_stats(items: List[epub.EpubHtml]) -> pd.DataFrame:
    """
    Compute various tag statistics aggregated over the book items.
    Also compute two new features:
      - avg_nb_section: average number of occurrences of a given tag/class combination per item.
      - section_proportion: proportion of items where the tag/class combination is found.
    """

    # To accumulate tag information per item.
    per_item_stats = []
    # To accumulate raw tag rows from every item to compute overall stats.
    all_tags = []

    for section_idx, item in enumerate(items):
        # Extract the HTML content from the item.
        content = item.content
        # Use the shared helper to extract main tags from the HTML.
        tags_df = extract_main_tags(str(content))
        if tags_df.empty:
            continue

        # Annotate with section id.
        tags_df = tags_df.copy()
        tags_df["section"] = section_idx
        all_tags.append(tags_df)

        # For this section, compute raw group stats (used later for consecutive groups if needed).
        # Group consecutive tags within the section so as to calculate group lengths.
        df = tags_df.copy()
        df["group_change"] = (df["tag"].ne(df["tag"].shift())) | (df["class"].ne(df["class"].shift()))
        df["group_id"] = df["group_change"].cumsum()
        # Compute average consecutive group length for each tag/class in this section.
        group_stats = df.groupby("group_id").agg(
            tag=("tag", "first"),
            clas=("class", "first"),
            group_length=("tag", "count")
        ).reset_index(drop=True)
        group_stats = group_stats.rename(columns={"clas": "class"})
        avg_consecutive = (
            group_stats.groupby(["tag", "class"])["group_length"]
            .mean()
            .reset_index()
            .rename(columns={"group_length": "av_consecutive_length"})
        )
        # Compute per section counts and text stats.
        section_counts = (
            tags_df.groupby(["tag", "class"])
            .agg(
                count=("tag", "count"),
                total_length=("text_length", "sum"),
                avg_length=("text_length", "mean"),
                last_decile_length=("text_length", lambda x: x.quantile(0.9)),
            )
            .reset_index()
        )
        section_counts["section"] = section_idx
        # Merge the average consecutive group length computed for this section.
        section_counts = section_counts.merge(avg_consecutive, on=["tag", "class"], how="left")
        per_item_stats.append(section_counts)

    if not all_tags:
        return pd.DataFrame()

    # Combine all raw tag rows from all items.
    overall_tags_df = pd.concat(all_tags, ignore_index=True)
    overall_stats = (
        overall_tags_df.groupby(["tag", "class"])
        .agg(
            count=("tag", "count"),
            total_length=("text_length", "sum"),
            avg_length=("text_length", "mean"),
            last_decile_length=("text_length", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Compute coverage and proportion based on overall counts.
    overall_stats["coverage"] = overall_stats["total_length"] / overall_stats["total_length"].sum()
    overall_stats["proportion"] = overall_stats["count"] / overall_stats["count"].sum()

    # Combine per-section statistics to compute avg_nb_section and section_proportion.
    per_item_df = pd.concat(per_item_stats, ignore_index=True)
    # Calculate average occurrences per item (only over sections where the tag appears).
    avg_nb_section = (
        per_item_df.groupby(["tag", "class"])["count"].mean().reset_index().rename(columns={"count": "avg_nb_section"})
    )
    # Calculate the number of sections where each tag/class combination appears.
    sections_presence = (
        per_item_df.groupby(["tag", "class"])["section"].nunique().reset_index().rename(columns={"section": "sections_found"})
    )
    av_consecutive_length = (
        per_item_df.groupby(["tag", "class"])["av_consecutive_length"]
        .mean()
        .reset_index()
    )
    # Total number of items.
    total_sections = len(items)
    sections_presence["section_proportion"] = sections_presence["sections_found"] / total_sections
    sections_presence = sections_presence.drop(columns="sections_found")

    # Merge new features into the overall stats.
    overall_stats = overall_stats.merge(avg_nb_section, on=["tag", "class"], how="left")
    overall_stats = overall_stats.merge(sections_presence, on=["tag", "class"], how="left")
    overall_stats = overall_stats.merge(av_consecutive_length, on=["tag", "class"], how="left")

    return overall_stats


def compute_tags_stats_all(directory: str) -> pd.DataFrame:
    list_epub_files = list_epub_files_from_directory(directory)
    tags_stats = []
    for epub_file in tqdm(list_epub_files):
        try:
            book = epub.read_epub(epub_file, options={"ignore_ncx": True})
            # items = get_items_from_book(book)
            items = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
            # body = concatenate_items_body(items)
            # tags_df = extract_main_tags(body)
            stats = compute_tags_stats(items)
            stats["book"] = epub_file.stem
            tags_stats.append(stats)
        except Exception as e:
            print(f"Error processing {epub_file}: {e}")

    return pd.concat(tags_stats, ignore_index=True)


def plot_stats(stats):
    from plotly import express as px

    fig = px.scatter_matrix(
        stats,
        dimensions=["proportion", "coverage",  "avg_nb_section", "count", "section_proportion"],
        color="tag",
        hover_data=["book", "class"],
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


def fit_content_classifier(stats: pd.DataFrame):
    header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    stats["is_header"] = stats["tag"].isin(header_tags)
    
    feature_cols = ["avg_length", "coverage", "last_decile_length", "count", "proportion", "av_consecutive_length"]
    X = stats.loc[stats["is_header"], feature_cols].dropna().values
    
    clf = EllipticEnvelope(contamination=0.5)
    clf.fit(X)
    
    stats["is_header_pred"] = clf.predict(stats[feature_cols].fillna(0).values).astype(str)