import json
import sys
import argparse
import html

def fix_encoding_and_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # Unescape HTML entities (e.g., &#304; -> İ)
    text = html.unescape(text)
    
    # Fix common Turkish encoding misinterpretations
    # These happen when Windows-1254 bytes are decoded as ISO-8859-1
    replacements = {
        'Ý': 'İ',
        'Þ': 'Ş',
        'Ð': 'Ğ',
        'ý': 'ı',
        'þ': 'ş',
        'ð': 'ğ'
    }
    
    # Fix common UTF-8 mojibake (UTF-8 decoded as ISO-8859-1)
    try:
        if 'Ã' in text or 'Ä' in text or 'Å' in text:
            text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    for bad, good in replacements.items():
        text = text.replace(bad, good)
        
    # Replace non-breaking spaces and remove zero-width spaces
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    
    # Trim leading/trailing whitespace
    return text.strip()

def process_record(record: dict) -> dict:
    """Recursively process dictionary to fix encoding and strip whitespace from strings."""
    processed = {}
    for key, value in record.items():
        if isinstance(value, str):
            processed[key] = fix_encoding_and_whitespace(value)
        elif isinstance(value, dict):
            processed[key] = process_record(value)
        elif isinstance(value, list):
            processed[key] = [
                fix_encoding_and_whitespace(item) if isinstance(item, str)
                else process_record(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            processed[key] = value
    return processed

def main():
    parser = argparse.ArgumentParser(description="Sanitize JSONL records by fixing Turkish encoding and trimming whitespace from titles and other text.")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(args.output_file, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    sanitized_record = process_record(record)
                    outfile.write(json.dumps(sanitized_record, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_number}: {e}", file=sys.stderr)
                    
        print(f"Successfully processed records. Output saved to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
