"""
Affiliation Extractor Module
Extracts and normalizes author affiliations from paper headers
"""

from .text_preprocessing import normalize_text, fuzzy_match


def affliation_extractor(text, metadata):
    """
    Extract affiliations from text by removing title and author names
    
    Args:
        text (str): Text before abstract (containing affiliations)
        metadata (dict): Paper metadata with title and authors
        
    Returns:
        str: Extracted affiliation text
    """
    lines = text.strip().splitlines()
    norm_lines = [normalize_text(line) for line in lines]
    
    # Prepare normalized title and author names
    norm_title = normalize_text(metadata.get("title", ""))
    norm_authors = [normalize_text(name) for name in metadata.get("authors", [])]
    
    # Remove title lines (fuzzy match, allow split titles across lines)
    title_removed = []
    i = 0
    while i < len(norm_lines):
        combined = norm_lines[i]
        j = i
        # Combine next 1-2 lines for fuzzy matching (to capture multi-line titles)
        while j + 1 < len(norm_lines) and len(combined) < 120:
            j += 1
            combined += " " + norm_lines[j]
            if fuzzy_match(combined, norm_title):
                i = j + 1
                break
        else:
            title_removed.append((lines[i], norm_lines[i]))
            i += 1
    
    # Now remove any lines containing author names
    final_lines = []
    for orig, norm in title_removed:
        if not any(author in norm for author in norm_authors):
            final_lines.append(orig)
    
    return "\n".join(final_lines).strip()