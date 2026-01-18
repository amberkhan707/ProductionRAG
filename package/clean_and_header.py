import re

def clean_text(text: str) -> str:
    """ Cleans text but preserves Newlines for Structure detection. """
    if not text: 
        return ""
    text = text.replace('\t', ' ').replace('\u00a0', ' ')
    # Fix broken words (hyphenated line breaks)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r'[ ]+', ' ', text)
    # OCR Safety: Ensure headers (#) start on new lines
    text = re.sub(r'(?<!\n)(#+\s)', r'\n\1', text)
    return text.strip()

def process_headers(text: str) -> list[str]:
    """ 
    Extracts H1-H3 headers while strictly ignoring Tables and Noise.
    """
    # 1. Capture Line: Line starts with #, then allow ANY space/tab, then capture text
    raw_headers = re.findall(r'^\s*#{1,3}\s+(.+)$', text, re.MULTILINE)
    
    clean_headers = []
    
    # Blocklist: Known junk headers
    IGNORE_LIST = [
        "table of contents", "document history", "copyright", "disclaimer",
        "document release note", "revision history", "attention:", "tables",
        "figures", "index"
    ]
    
    for title in raw_headers:
        if "|" in title:
            continue
            
        # Step A: Basic Strip (Markdown bolding/italics removed)
        clean = title.strip().strip('*').strip('_').strip()
        
        # Step B: Remove leading Hashes 
        clean = clean.lstrip('#').strip()
        
        # Step C: Aggressive Number Removal (e.g. "4.1.2 Scope" -> "Scope")
        clean = re.sub(r'^[\d\.]+\s*', '', clean)
        
        # Step D: Final Cleanup
        clean = clean.strip()
        if (len(clean) > 2 and 
            len(clean) < 50 and 
            clean.lower() not in IGNORE_LIST and 
            not clean.isdigit()):
            
            clean_headers.append(clean)
            
    return clean_headers