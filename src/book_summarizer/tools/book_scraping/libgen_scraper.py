from bs4 import BeautifulSoup
import Levenshtein
import requests
import pandas as pd


class LibgenScraper:
    """
    A class to scrape books from the Libgen library.

    Attributes:
    -----------
    base_url : str
        The base URL for the Libgen website.
    levenshtein_thd : int
        The threshold for the Levenshtein distance to filter search results.
    columns_idx : dict
        A dictionary mapping the columns of the results table to their indices.
    results : list
        A list to store the results from the search.
    """

    def __init__(self, base_url: str, levenshtein_thd: int, columns_idx: dict, output_folder: str):
        """
        Initialize the LibgenScraper with the given parameters.

        Parameters:
        -----------
        base_url : str
            The base URL for the Libgen website.
        levenshtein_thd : int
            The threshold for the Levenshtein distance to filter search results.
        columns_idx : dict
            A dictionary mapping the columns of the results table to their indices.
        """
        self.base_url = base_url
        self.levenshtein_thd = levenshtein_thd
        self.columns_idx = columns_idx
        self.results = []
        self.output_folder = output_folder

    def get_book_by_title(self, title: str, language="eng", extension="epub") -> bool:
        """
        Search for a book by title and attempt to download it.

        Parameters:
        -----------
        title : str
            The title of the book to search for.
        language : str, optional
            The language of the book (default is "eng").
        extension : str, optional
            The file extension of the book (default is "epub").

        Returns:
        --------
        bool
            True if the book is successfully downloaded, False otherwise.
        """
        self.query = title
        self.language = language
        self.extension = extension
        webpage = self.fetch_webpage(title)
        if webpage:
            results = self.parse_results_webpage(webpage)
            results = self.filter_results(results)

            for _, row in results.iterrows():
                for link in row["mirror_links"]:
                    if self.download_book(link, row["title"], self.output_folder):
                        return True
        return False

    def _search_url(self, title: str) -> str:
        """
        Construct the search URL for the given title.

        Parameters:
        -----------
        title : str
            The title of the book to search for.

        Returns:
        --------
        str
            The constructed search URL.
        """
        research_field = title.replace(" ", "+")
        return f"{self.base_url}/index.php?req={research_field}+lang%3A{self.language}+ext%3A{self.extension}&res=100&gmode=on"

    def fetch_webpage(self, title: str) -> BeautifulSoup:
        """
        Fetch the search results webpage for the given title.

        Parameters:
        -----------
        title : str
            The title of the book to search for.

        Returns:
        --------
        BeautifulSoup
            The parsed HTML of the search results page.
        """
        try:
            response = requests.get(self._search_url(title))
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Failed to fetch webpage: {e}")
            return None

    def extract_row_data(self, row) -> dict:
        """
        Extract data from a table row.

        Parameters:
        -----------
        row : bs4.element.Tag
            A table row containing book data.

        Returns:
        --------
        dict
            A dictionary with the extracted book data.
        """
        columns = row.find_all("td")
        if not columns:
            raise ValueError("No columns with tag td found in the row")
        
        title = columns[self.columns_idx["title"]].find_all("a")[0].text
        author = columns[self.columns_idx["author"]].text
        size = columns[self.columns_idx["size"]].text
        size, size_unit = size.split(" ")
        size = float(size)
        if size_unit == "kB":
            size *= 1e-3
        mirrors = columns[self.columns_idx["mirrors"]]
        mirror_links = [m.get("href") for m in mirrors.find_all("a")]

        return {
            "title": title,
            "author": author,
            "size": size,
            "mirror_links": mirror_links,
        }

    def parse_table(self, webpage: BeautifulSoup) -> list:
        """
        Parse the results table from the webpage.

        Parameters:
        -----------
        webpage : BeautifulSoup
            The parsed HTML of the search results page.

        Returns:
        --------
        list
            A list of table rows.
        """
        tbody = webpage.find("tbody")
        if tbody is None:
            raise ValueError("No table with tag tbody found in the webpage")
        
        return tbody.find_all("tr")

    def parse_results_webpage(self, webpage: BeautifulSoup) -> pd.DataFrame:
        """
        Parse the search results webpage and extract book data.

        Parameters:
        -----------
        webpage : BeautifulSoup
            The parsed HTML of the search results page.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the search results.
        """
        rows = self.parse_table(webpage)
        for row in rows:
            try:
                row_data = self.extract_row_data(row)
                self.results.append(row_data)
            except ValueError as e:
                print(f"Error extracting row data: {e}")

        return pd.DataFrame(self.results)

    def filter_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the search results based on Levenshtein distance and file size.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the search results.

        Returns:
        --------
        pd.DataFrame
            The filtered search results.
        """
        df["levenshtein"] = df["title"].apply(
            lambda x: Levenshtein.distance(
                x.lower().strip(), self.query.lower().strip()
            )
        )
        df = df[df["levenshtein"] <= self.levenshtein_thd]
        df = df[df["size"].between(df["size"].quantile(0.4), df["size"].quantile(0.6))]
        return df

    def download_book(self, link: str, title: str, output_folder: str = None) -> bool:
        """
        Attempt to download a book from the given link.

        Parameters:
        -----------
        link : str
            The download link for the book.
        title : str
            The title of the book.
        output_folder : str, optional
            The folder to save the downloaded file (default is None).

        Returns:
        --------
        bool
            True if the book is successfully downloaded, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/{link}")
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("body").find("table")
            tags_with_href = table.find_all(lambda tag: tag.has_attr("href"))
            href_with_get = [tag for tag in tags_with_href if "get" in tag["href"].lower()]

            if len(href_with_get) != 1:
                raise ValueError("There should be only one get link")

            download_link = href_with_get[0]["href"]
            response = requests.get(f"{self.base_url}/{download_link}", stream=True)
            response.raise_for_status()

            if output_folder is not None:
                output_file = f"{output_folder}/{title.strip()}.{self.extension}"
                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"File downloaded successfully and saved as {output_file}")
            return True

        except requests.RequestException as e:
            print(f"Failed to download file: {e}")
            return False
        except ValueError as e:
            print(f"Error in download process: {e}")
            return False
