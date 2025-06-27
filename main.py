import zipfile
from bs4 import BeautifulSoup
import re
import pandas as pd
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables from .env file
load_dotenv()

# Pattern to identify study mentions
study_pattern = re.compile(
    r"(?:study|research|experiment|scientists|researchers|evidence|data|findings|et al\\.?).*?[.?!]",
    re.IGNORECASE
)
# Split text into sentences based on punctuation
sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')

# Supported OpenAI models for different cost/length trade-offs
SUPPORTED_MODELS = [
    'gpt-3.5-turbo',       # Lower cost, 4k context window
    'gpt-3.5-turbo-16k',   # Moderate cost, 16k context
    'gpt-4',               # Higher cost, 8k context
    'gpt-4-32k'            # Highest cost, 32k context
]

API_KEY = os.getenv('OPENAI_API_KEY')
CHUNK_SIZE = 4000  # characters per chunk
OVERLAP = 400      # overlap between chunks

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

def load_epub_text(epub_path):
    text_parts = []
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        html_files = [f for f in zip_ref.namelist() if f.endswith(('.xhtml', '.html'))]
        for file in html_files:
            with zip_ref.open(file) as html_file:
                soup = BeautifulSoup(html_file, 'html.parser')
                text_parts.append(soup.get_text())
    return '\n'.join(text_parts)


def extract_research_contexts(text, pattern, window=5):
    """
    For each sentence containing a research keyword, extract N sentences before and after.
    Returns a list of context strings.
    """
    sentences = re.split(sentence_split_pattern, text)
    contexts = []
    for idx, sentence in enumerate(sentences):
        if pattern.search(sentence):
            start = max(0, idx - window)
            end = min(len(sentences), idx + window + 1)
            context = ' '.join(sentences[start:end])
            contexts.append(context)
    return contexts


def call_openai_for_chunk(chunk, model):
    prompt = (
        "Extract any scientific study mentions, their findings, and a set of actionable practices or recommendations for each study from the following text. "
        "Return a strict JSON array of objects with keys: study, findings, recommendation (where recommendation is a set of practices or actions) and nothing else. "
        "If no valid study is in the chunk, return [] exactly.\n" +
        chunk
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences
        if content.startswith('```') and content.endswith('```'):
            content = '\n'.join(content.split('\n')[1:-1]).strip()
        return content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return '[]'


def parse_json_responses(json_texts):
    records = []
    for text in json_texts:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records.extend(data)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        records.extend(data)
                except json.JSONDecodeError:
                    continue
    return records


def extract_with_openai(epub_path, output_csv, model, max_chunks=None):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model}' not supported. Choose from: {SUPPORTED_MODELS}")

    full_text = load_epub_text(epub_path)
    # Instead of chunking, extract research contexts
    contexts = extract_research_contexts(full_text, study_pattern, window=5)
    if max_chunks is not None and max_chunks > 0:
        contexts = contexts[:max_chunks]
    print(f"Total research contexts: {len(contexts)}")

    json_texts = []
    for i, context in enumerate(contexts, 1):
        print(f"Processing context {i}/{len(contexts)} with model {model}")
        json_texts.append(call_openai_for_chunk(context, model))

    records = parse_json_responses(json_texts)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Extracted {len(df)} records. Data saved to '{output_csv}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract studies from an EPUB using OpenAI API with configurable model.'
    )
    parser.add_argument('epub_file_path', help='Path to the EPUB file')
    parser.add_argument(
        'output_csv', nargs='?', default='extracted_studies.csv',
        help='Output CSV filename (optional)'
    )
    parser.add_argument(
        '--model', type=str, default='gpt-4', choices=SUPPORTED_MODELS,
        help='OpenAI model to use (default: gpt-4)'
    )
    parser.add_argument(
        '--max-chunks', type=int, default=None,
        help='Max number of chunks to process (optional)'
    )
    args = parser.parse_args()

    if not API_KEY:
        print('Error: OPENAI_API_KEY not found. Ensure .env contains your key.')
        exit(1)

    extract_with_openai(
        args.epub_file_path,
        args.output_csv,
        args.model,
        args.max_chunks
    )
