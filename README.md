## Book Summarizer: Introduction

The Book Summarizer project aims to provide concise and insightful summaries of books using the power of Large Language Models (LLM) and change point detection algorithms.  This project goes beyond simple text summarization by identifying key segments within the book and leveraging the context of previous sections to generate a hierarchical summary that reflects the book's structure and main arguments.

Key features of this project include:

* **EPUB Parsing:** Extracts textual content from EPUB files, handling different formatting and structures.
* **Text Embedding:** Utilizes the VoyageAI API to compute rich text embeddings for each extracted chunk of text, capturing semantic meaning.
* **Change Point Detection:** Employs change point detection algorithms on the text embeddings to identify shifts in topic or narrative, segmenting the book into meaningful units.
* **Hierarchical Summarization:** Leverages the power of Google's Gemini LLM to generate summaries for each identified segment, taking into account the context of previously summarized sections to ensure coherence and flow.

Built using the Kedro pipeline framework, the Book Summarizer project offers a modular and scalable approach to book summarization, allowing for customization and extension to incorporate different LLMs, embedding models, and summarization strategies.

## Getting Started

This section guides you through setting up the Book Summarizer project on your local machine.

### Prerequisites


1. **Clone the Repository:** Clone the Book Summarizer repository to your local machine using:

   ```bash
   git clone https://github.com/your-username/book-summarizer.git
   cd book-summarizer
   ```

2. **Create a Virtual Environment:** Create a virtual environment to isolate the project's dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Configure Credentials:**
    You need to obtain API keys for the VoyageAI and Google Cloud APIs to run the Book Summarizer project.
    Set the api keys in the `conf/base/credentials.yml` file:

   ```yaml
   voyageai_api_key: "YOUR_VOYAGEAI_API_KEY"
   google_api_key: "YOUR_GOOGLE_CLOUD_API_KEY"
   ```

## Pipelines

The Book Summarizer project utilizes a series of interconnected Kedro pipelines to process data and generate the final book summary. Each pipeline consists of individual nodes that perform specific tasks.

### Pipeline Overview

The project comprises three main pipelines:

1. **`parse_ebook`:** This pipeline is responsible for extracting textual content from the input EPUB file and preparing it for further processing.
2. **`compute_embedding`:** This pipeline focuses on generating meaningful numerical representations of the extracted text chunks using text embedding techniques.
3. **`summarize`:** This pipeline utilizes the generated embeddings to identify key segments in the book and leverages an LLM to produce a hierarchical summary.

### Pipeline Details and Data Flow

Here's a breakdown of each pipeline and the data flow between them:

1. **`parse_ebook`:**
    * **Input:** Raw EPUB file (`epub_content` dataset).
    * **Process:**
        * Infers paragraph levels to identify appropriate text chunks.
        * Extracts text chunks from the EPUB content.
        * Filters out very short chunks that are unlikely to contain meaningful information.
    * **Output:** Dataframe containing extracted text chunks (`chunks` dataset).

2. **`compute_embedding`:**
    * **Input:** Dataframe of text chunks (`chunks` dataset).
    * **Process:**
        * Makes API calls to VoyageAI to generate text embeddings for each chunk.
        * Applies dimensionality reduction techniques (PCA, UMAP) to the embeddings for visualization and change point detection.
    * **Output:** 
        * Dataframe containing text embeddings (`embeddings` dataset).
        * Dataframes containing PCA and UMAP projections of the embeddings (`pca_projection`, `umap_projection` datasets).

3. **`summarize`:**
    * **Input:** Dataframe of PCA-reduced embeddings (`pca_projection` dataset).
    * **Process:**
        * Builds a summary tree by applying change point detection algorithms to the embeddings, identifying key segments in the book.
        * Traverses the summary tree, using the Google Gemini LLM to generate summaries for each node, incorporating context from parent nodes.
    * **Output:** Summary tree object containing hierarchical summaries of the book (`summary_tree` dataset).

By chaining these pipelines together, the Book Summarizer project effectively transforms raw ebook data into a structured and informative summary, highlighting the key points and organization of the book.