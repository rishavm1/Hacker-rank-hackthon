"""
corpus_loader.py — Load the pre-shipped markdown support corpus into structured documents.

Recursively reads all .md files from data/{hackerrank,claude,visa}/ and returns
a list of document dicts with metadata (domain, category, title, content, filepath).
"""

import os
import re
from pathlib import Path


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-like frontmatter and return (metadata_dict, body)."""
    metadata = {}
    body = text
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if match:
        fm = match.group(1)
        body = text[match.end():]
        for line in fm.split('\n'):
            if ':' in line:
                key, _, val = line.partition(':')
                metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata, body


def _extract_title(metadata: dict, body: str, filepath: str) -> str:
    """Extract a meaningful title from metadata, markdown heading, or filename."""
    if 'title' in metadata and metadata['title']:
        return metadata['title']
    # Try first markdown heading
    match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fallback to filename
    name = Path(filepath).stem
    # Clean up numeric prefixes like "1258426835-best-practices-..."
    name = re.sub(r'^\d+-', '', name)
    return name.replace('-', ' ').replace('_', ' ').title()


def _infer_category(filepath: str, domain: str) -> str:
    """Infer the product area / category from the directory path."""
    parts = Path(filepath).parts
    # Find the domain directory and take the next directory as category
    try:
        domain_idx = None
        for i, p in enumerate(parts):
            if p == domain:
                domain_idx = i
                break
        if domain_idx is not None and domain_idx + 1 < len(parts) - 1:
            return parts[domain_idx + 1]
    except (ValueError, IndexError):
        pass
    return 'general'


def load_corpus(data_dir: str) -> list[dict]:
    """
    Load all markdown files from the data directory.
    
    Returns a list of dicts:
        {
            'content': str,        # Full document text (body without frontmatter)
            'domain': str,         # 'hackerrank', 'claude', or 'visa'
            'category': str,       # Sub-category from directory structure
            'title': str,          # Extracted title
            'filepath': str,       # Relative path to the file
            'metadata': dict,      # Frontmatter metadata
        }
    """
    documents = []
    data_path = Path(data_dir)
    
    for domain in ['hackerrank', 'claude', 'visa']:
        domain_path = data_path / domain
        if not domain_path.exists():
            continue
        
        for md_file in domain_path.rglob('*.md'):
            try:
                text = md_file.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue
            
            if not text.strip():
                continue
            
            metadata, body = _extract_frontmatter(text)
            
            # Skip very short / empty docs
            if len(body.strip()) < 20:
                continue
            
            rel_path = str(md_file.relative_to(data_path))
            title = _extract_title(metadata, body, str(md_file))
            category = _infer_category(rel_path, domain)
            
            documents.append({
                'content': body.strip(),
                'domain': domain,
                'category': category,
                'title': title,
                'filepath': rel_path,
                'metadata': metadata,
            })
    
    return documents


if __name__ == '__main__':
    # Quick test
    import sys
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    docs = load_corpus(data_dir)
    print(f"Loaded {len(docs)} documents")
    for domain in ['hackerrank', 'claude', 'visa']:
        count = sum(1 for d in docs if d['domain'] == domain)
        print(f"  {domain}: {count} docs")
    if docs:
        print(f"\nSample doc: {docs[0]['title'][:80]} ({docs[0]['domain']}/{docs[0]['category']})")
