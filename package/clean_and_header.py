import re

def clean_text(text: str) -> str:
    """ Cleans text but preserves Newlines for Structure detection. """
    if not text: return ""
    text = text.replace('\t', ' ').replace('\u00a0', ' ')
    # Fix broken words (hyphenated line breaks)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r'[ ]+', ' ', text)
    # OCR Safety: Ensure headers (#) start on new lines
    text = re.sub(r'(?<!\n)(#+\s)', r'\n\1', text)
    return text.strip()

import re

def process_headers(text: str) -> list[str]:
    """ 
    Extracts H1-H3 headers and performs Aggressive Cleaning.
    """
    # 1. Capture Line: Line starts with #, then allow ANY space/tab, then capture text
    raw_headers = re.findall(r'^\s*#{1,3}\s+(.+)$', text, re.MULTILINE)
    
    clean_headers = []
    
    # Blocklist: Headers jo bilkul nahi chahiye
    IGNORE_LIST = [
        "table of contents", "document history", "copyright", "disclaimer",
        "document release note", "revision history", "attention:"
    ]
    
    for title in raw_headers:
        # Step A: Basic Strip (Markdown bolding removed)
        clean = title.strip().strip('*').strip()
        # Step B: Remove leading Hashes (Safety net if regex leaked #)
        clean = clean.lstrip('#').strip()
        clean = re.sub(r'^[\d\.]+\s*', '', clean)
        # Step D: Final Cleanup
        clean = clean.strip()
        # Step E: Filtering
        if len(clean) > 2 and clean.lower() not in IGNORE_LIST:
            clean_headers.append(clean)
            
    return clean_headers