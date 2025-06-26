# Research-Extractor

A Python script to extract scientific study mentions, findings, and recommendations from EPUB files using the OpenAI API. The extracted data is saved to a CSV file.

## Features
- Processes EPUB files and extracts text from HTML/XHTML content.
- Splits text into manageable chunks and filters for study-related content.
- Uses OpenAI's GPT models to extract structured information about studies.
- Outputs results as a CSV file.

## Requirements
- Python 3.7+
- An OpenAI API key (set in a `.env` file)

### Python Dependencies
Install dependencies with:
```bash
pip install openai python-dotenv beautifulsoup4 pandas
```

## Usage

```bash
python main.py <epub_file_path> [output_csv] [--model MODEL] [--max-chunks N]
```

### Arguments
- `epub_file_path` (required): Path to the input EPUB file.
- `output_csv` (optional): Output CSV filename. Defaults to `extracted_studies.csv`.
- `--model` (optional): OpenAI model to use. Choices: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`, `gpt-4`, `gpt-4-32k`. Default: `gpt-4`.
- `--max-chunks` (optional): Maximum number of text chunks to process. Useful for limiting API usage during testing.

### Example
```bash
python main.py mybook.epub results.csv --model gpt-3.5-turbo --max-chunks 10
```

## Environment Variables
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

## Output
The script will create a CSV file with columns:
- `study`: The study or research mentioned
- `findings`: The findings of the study
- `recommendation`: Any recommended actions

## Notes
- Only chunks containing study-related keywords are sent to the OpenAI API.
- If no studies are found, the output CSV may be empty.
- Ensure you have sufficient OpenAI API quota for your chosen model and number of chunks.
