import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from fake_useragent import UserAgent

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

def is_valid_url(url):
    # Adjust the regex pattern to match the desired URL patterns
    pattern = re.compile(r'https?://[\w\-\.]+(/news/[\w\-]+|/blogs/[\w\-]+)?')
    return bool(pattern.match(url))

def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    
    # Remove script and style tags
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Get the text and remove extra spaces
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)

    return clean_text

def save_content_to_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def extract_links(url, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            full_url = urljoin(url, href)
            if is_valid_url(full_url):
                links.append(full_url)
    return links

def scrape_website(url, output_directory):
    visited_urls = set()
    urls_to_visit = [url]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    ua = UserAgent()

    while urls_to_visit:
        current_url = urls_to_visit.pop(0)
        headers = {'User-Agent': ua.random}
        response = requests.get(current_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch '{current_url}'")
            continue

        
        
        if current_url in visited_urls or not is_valid_url(current_url):
            continue
        
        visited_urls.add(current_url)
        
        print(f"Scraping '{current_url}'")

        html_content = response.text
        cleaned_content = clean_text(html_content)
        file_name = f"{len(visited_urls):04d}.txt"
        file_path = os.path.join(output_directory, file_name)
        save_content_to_file(cleaned_content, file_path)

        new_links = extract_links(current_url, html_content)
        urls_to_visit.extend(link for link in new_links if link not in visited_urls)


if __name__ == "__main__":
    url = input("Enter the starting URL: ")
    output_directory = input("Enter the output directory: ")
    scrape_website(url, output_directory)
