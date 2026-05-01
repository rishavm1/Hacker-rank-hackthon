"""
classifier.py — Domain and request type classification.

Classifies tickets by:
1. Domain (hackerrank/claude/visa) when company is None
2. Request type (product_issue/feature_request/bug/invalid)
3. Product area mapping from corpus categories
"""

import re


# ─── Domain detection keywords ───────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    'hackerrank': [
        'hackerrank', 'hacker rank', 'test', 'assessment', 'candidate', 'interview',
        'coding test', 'screen', 'plagiarism', 'proctoring', 'recruiter', 'hiring',
        'code challenge', 'certification', 'hackerrank for work', 'mock interview',
        'skillup', 'library', 'engage', 'chakra', 'submissions', 'challenges',
    ],
    'claude': [
        'claude', 'anthropic', 'conversation', 'ai assistant', 'chat',
        'artifacts', 'projects', 'bedrock', 'api key', 'claude code',
        'claude desktop', 'claude pro', 'claude max', 'claude team',
        'lti', 'mcp', 'prompt',
    ],
    'visa': [
        'visa', 'card', 'credit card', 'debit card', 'transaction', 'atm',
        'merchant', 'payment', 'lost card', 'stolen card', 'travel',
        'cheque', 'fraud', 'chargeback', 'visa card', 'issuer', 'bank',
        'dispute a charge', 'traveller',
    ],
}


# ─── Request type patterns ───────────────────────────────────────────────────

BUG_PATTERNS = [
    r'\b(outage|broken|not working for all)\b',
]

FEATURE_REQUEST_PATTERNS = [
    r'\b(add feature|request feature)\b',
]

INVALID_PATTERNS = [
    r'\b(irrelevant|thank you|non-support)\b',
]

# ─── Prompt injection detection ──────────────────────────────────────────────

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'ignore\s+the\s+above',
    r'system\s*prompt',
    r'show\s+(me\s+)?(your|the)\s+(rules|instructions|prompt|internal)',
    r'display\s+(all|your)\s+(rules|logic|internal)',
    r'override\s+(your|the)\s+',
    r'forget\s+(everything|all|your)',
    r'you\s+are\s+now\s+',
    r'new\s+instructions?\s*:',
    r'(affiche|montre|muestra)\s+.*(règles|reglas|internal|logique)',
    r'delete\s+all\s+files',
    r'rm\s+-rf',
    r'format\s+c:',
    r'code\s+to\s+delete',
]


# ─── Product area mapping ────────────────────────────────────────────────────
# Product areas are now strictly derived from retrieved document metadata



def detect_injection(text: str) -> bool:
    """Check if text contains prompt injection attempts."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def classify_domain(issue: str, subject: str, company: str | None,
                    retriever=None) -> str:
    """
    Classify the domain for a ticket.
    
    If company is provided and valid, use it directly.
    Otherwise, use keyword matching + retrieval scores to infer domain.
    """
    # Direct company mapping
    if company and company.strip().lower() not in ('none', ''):
        company_lower = company.strip().lower()
        if company_lower in ('hackerrank', 'hacker rank'):
            return 'hackerrank'
        elif company_lower in ('claude', 'anthropic'):
            return 'claude'
        elif company_lower in ('visa',):
            return 'visa'
    
    # Keyword-based inference
    text = f"{issue} {subject}".lower()
    domain_scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        # Weight exact brand name matches more heavily
        if domain in text:
            score += 5
        domain_scores[domain] = score
    
    # If we have a clear keyword winner, use it
    max_score = max(domain_scores.values()) if domain_scores else 0
    if max_score >= 2:
        best = max(domain_scores, key=domain_scores.get)
        return best
    
    # Use retriever for semantic matching
    if retriever:
        best_domain, confidence = retriever.get_best_domain(f"{issue} {subject}")
        if confidence > 0.05:
            return best_domain
    
    return 'unknown'


def classify_request_type(issue: str, subject: str) -> str:
    """
    Classify the request type based on patterns.
    
    Returns: product_issue, feature_request, bug, or invalid
    """
    text = f"{issue} {subject}".lower()
    
    # Check for injection first → invalid
    if detect_injection(text):
        return 'invalid'
    
    # Check for invalid / out-of-scope
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            # But make sure it's not a real support question with a thank you
            if len(text.strip()) < 50 and not any(
                kw in text for kw_list in DOMAIN_KEYWORDS.values() for kw in kw_list
            ):
                return 'invalid'
    
    # Check for bugs
    for pattern in BUG_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 'bug'
    
    # Check for feature requests
    for pattern in FEATURE_REQUEST_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 'feature_request'
    
    return 'product_issue'


def classify_product_area(issue: str, subject: str, domain: str,
                          retrieved_docs: list[dict] | None = None) -> str:
    """
    Classify the product area deterministically from the top retrieved document.
    """
    if retrieved_docs and len(retrieved_docs) > 0:
        doc = retrieved_docs[0].get('document', retrieved_docs[0]) if isinstance(retrieved_docs[0], dict) else retrieved_docs[0]
        cat = doc.get('category', '').lower().replace('_', '-')
        
        if domain == 'hackerrank':
            if cat == 'screen': return 'screen'
            if cat == 'hackerrank-community': return 'community'
        elif domain == 'claude':
            if cat == 'privacy-and-legal': return 'privacy'
            if cat == 'account-management': return 'account'
            if cat == 'troubleshooting': return 'troubleshooting'
        elif domain == 'visa':
            if cat == 'travel-support': return 'travel_support'
            if cat == 'dispute-resolution': return 'general_support'
            if cat == 'fraud-protection': return 'general_support'
            
    # Default fallback
    return 'general_support'
