import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://github.com/sanand0/tools-in-data-science-public/tree/tds-2025-01"
RAW_PREFIX = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/"

def fetch_markdown_files():
    urls = [
        "2025-01.md",
        "development-tools.md",
        "deployment-tools.md",
        "large-language-models.md",
        "data-sourcing.md",
        "data-preparation.md",
        "data-analysis.md",
        "data-visualization.md",
        "project-1.md",
        "project-2.md"
    ]
    os.makedirs("data", exist_ok=True)
    for url in urls:
        full_url = RAW_PREFIX + url
        print(f"Downloading {url}...")
        content = requests.get(full_url).text
        with open(f"data/{url}", "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    fetch_markdown_files()
