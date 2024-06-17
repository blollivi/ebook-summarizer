from typing import List
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd


def _explore_tree(tag, stats=None, level=0):
    """
    Recursively explore the tree structure of the HTML content.
    Compute the number of tags and the number of words per tree level.
    Keep a list of unique tags at each level.
    """
    if stats is None:
        stats = {}

    # Number of words in the current tag's direct text (excluding children)
    nb_words = len(tag.get_text(strip=True, separator=" ").split())

    if level not in stats:
        stats[level] = {
            "nb_words": nb_words,
            "nb_tags": 1,
            "unique_tags": {tag.name} if tag.name else set(),
        }
    else:
        stats[level]["nb_words"] += nb_words
        stats[level]["nb_tags"] += 1
        if tag.name:
            stats[level]["unique_tags"].add(tag.name)

    for child in tag.children:
        if child.name:
            _explore_tree(child, stats, level + 1)

    return stats


def _infer_paragraph_level(nb_words: List[int]) -> int:
    """
    Infer the paragraph level based on the number of words per level.
    The paragraph level is the deepest level that cover more than 95%
    of the number of words in the top level.
    """
    top_level_words = nb_words[0]
    for i in range(1, len(nb_words)):
        if nb_words[i] < 0.95 * top_level_words:
            return i - 1
    return 1


def infer_paragraph_levels(ebook_content: List[str]) -> pd.DataFrame:
    """
    Loop over all the xhtml content. For each xhtml file, explore the tree structure
    in order to infer which tag levels are likely to represent paragraphs.
    """

    paragraph_levels = []

    for content in ebook_content:
        soup = BeautifulSoup(content, "html.parser")
        body_tag = soup.find("body")
        level_stats = _explore_tree(body_tag)
        nb_words = [level_stats[i]["nb_words"] for i in level_stats.keys()]
        paragraph_levels.append(_infer_paragraph_level(nb_words))

    return paragraph_levels


def _extract_chunks(tag, paragraph_level, chunks=None, level=0):
    """
    Recursively extract chunks of text from the xhtml content based on the inferred paragraph levels.
    Each chunk corresponds to the text contained in tags at the inferred paragraph level.
    """
    if chunks is None:
        chunks = []

    if level == paragraph_level:
        chunks.append(tag.get_text(strip=True))
        return chunks

    for child in tag.children:
        if child.name:
            _extract_chunks(child, paragraph_level, chunks, level + 1)

    return chunks


def extract_chunks(ebook_content: List[str], paragraph_levels: List[int]) -> List[str]:
    """
    Extract chunks of text from the xhtml content based on the inferred paragraph levels.
    Each chunk corresponds to the text contained in tags at the inferred paragraph level.

    Parameters:
    ----------
    ebook_content: List[str]
        List of xhtml content.
    paragraph_levels: List[int]
        List of inferred paragraph levels.

    Example:
    ----------
    For the following xhtml content:
    <body>
        <div>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
        </div>
        <p>Paragraph 3</p>
    </body>

    If the inferred paragraph level is 1, the output for the given html content would be:
    ["Paragraph 1 Paragraph 2", "Paragraph 3"]
    """
    chapters = []
    for content, level in zip(ebook_content, paragraph_levels):
        soup = BeautifulSoup(content, "html.parser")
        body_tag = soup.find("body")
        # Find all tags at the given level in the tree
        chapters.append(_extract_chunks(body_tag, paragraph_level=level))

    # Format as dataframe, one row per chunk, one column for the text, one for the chapter index,
    # and one for the chunk index within the chapter
    chunks_df = pd.DataFrame(
        [
            {"text": chunk, "chapter": i, "chunk": j}
            for i, chapter in enumerate(chapters)
            for j, chunk in enumerate(chapter)
        ]
    )

    return chunks_df


def filter_chunks(chunks_df: pd.DataFrame, min_words: int = 20) -> pd.DataFrame:
    """
    Filter out chunks with less than `min_words` words.
    """
    return chunks_df[chunks_df["text"].apply(lambda x: len(x.split()) >= min_words)]