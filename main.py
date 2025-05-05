import zipfile
from bs4 import BeautifulSoup
import re
import pandas as pd
import argparse

# Define the pattern to find studies
study_pattern = re.compile(r'(?:study|research|experiment|scientists|researchers|evidence|data|findings).*?[.?!]', re.IGNORECASE)

def extract_studies_from_epub(epub_path, output_csv):
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        html_files = [f for f in file_list if f.endswith(('.xhtml', '.html'))]

        studies = []
        for file in html_files:
            with zip_ref.open(file) as html_file:
                soup = BeautifulSoup(html_file, 'html.parser')
                text = soup.get_text()
                found_studies = study_pattern.findall(text)
                studies.extend(found_studies)

    # Create DataFrame to save
    df_studies = pd.DataFrame(studies, columns=['Study Mentions'])
    df_studies.to_csv(output_csv, index=False)
    print(f"Extracted {len(df_studies)} studies. Data saved to '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract study mentions from an EPUB file.")
    parser.add_argument("epub_file_path", type=str, help="Path to the EPUB file")
    parser.add_argument("output_csv", type=str, nargs='?', default="extracted_studies.csv", help="Output CSV file name (optional)")

    args = parser.parse_args()

    extract_studies_from_epub(args.epub_file_path, args.output_csv)