import pytest
from kedro.config import OmegaConfigLoader
from pathlib import Path

from book_summarizer.tools.book_scraping.libgen_scraper import LibgenScraper


@pytest.fixture
def config_loader():
    return OmegaConfigLoader(conf_source=str(Path.cwd()))


@pytest.fixture
def sample_query():
    return "The Great Gatsby"


@pytest.fixture
def libgen_scrapper(config_loader, sample_query):
    config = config_loader.get("parameters").get("libgen_scrapper")
    libgen_scrapper = LibgenScraper(**config)
    libgen_scrapper.language = "eng"
    libgen_scrapper.extension = "epub"
    libgen_scrapper.query = sample_query
    return libgen_scrapper


def test_libgen_scrapper(libgen_scrapper, sample_query):
    soup = libgen_scrapper.fetch_webpage(sample_query)
    # Make sure that soup is not empty
    assert soup is not None, "Fetch search page failed"

    rows = libgen_scrapper.parse_table(soup)
    assert len(rows) > 0, "No rows found in the table"

    row = rows[0]
    row_data = libgen_scrapper.extract_row_data(row)

    assert (
        sample_query.lower() in row_data["title"].lower()
    ), "Title does not match the query"

    results = libgen_scrapper.parse_results_webpage(soup)
    results = libgen_scrapper.filter_results(results)

    is_downloaded = libgen_scrapper.download_book(
        results["mirror_links"].iloc[0][0], results["title"].iloc[0]
    )

    assert is_downloaded, "Book download failed"
