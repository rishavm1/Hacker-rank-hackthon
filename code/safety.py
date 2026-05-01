"""
safety.py — Rule-based safety and escalation checks.

Determines whether a ticket should be escalated based on risk factors:
- Fraud / unauthorized transactions / identity theft
- Service outages affecting all users
- Billing disputes requiring human review
- Account-level access (restore, admin actions)
- Legal / regulatory requests
- Prompt injection or malicious input
- Insufficient corpus coverage
"""

import re

from classifier import detect_injection


# ─── Escalation rule definitions ─────────────────────────────────────────────

ESCALATION_RULES = [
    {
        'name': 'fraud_or_unauthorized',
        'description': 'Fraud, unauthorized transactions, or account compromise',
        'patterns': [
            r'\b(fraud|unauthorized\s+transaction|account\s+compromise[d]?)\b',
            r'\b(identity\s+(theft|stolen))\b',
            r'\b(stolen\s+(identity|credentials?))\b',
            r'\b(suspicious\s+(activity|transaction|charge))\b',
            r'\b(hacked|compromised|breached)\b',
        ],
    },
    {
        'name': 'service_outage',
        'description': 'Service outage or site-wide bug',
        'patterns': [
            r'\b(site\s+is\s+down|service\s+(is\s+)?down|outage)\b',
            r'\b(all\s+(requests?|pages?|submissions?)\s+(are\s+)?failing)\b',
            r'\b(completely\s+(down|broken|stopped))\b',
            r'\b(nothing\s+(is\s+)?working)\b',
            r'\b(none\s+of\s+the\s+.{0,30}\s+working)\b',
            r'\b(stopped\s+working\s+completely)\b',
        ],
    },
    {
        'name': 'billing_dispute',
        'description': 'Billing dispute requiring human review',
        'patterns': [
            r'\b(charge\s*back|billing\s+dispute)\b',
            r'\b(overcharged|double\s+charged)\b',
            r'\b(cancel\s+(my\s+)?subscription|pause\s+(our\s+)?subscription)\b',
        ],
    },
    {
        'name': 'account_access_admin',
        'description': 'Account-level access requiring admin intervention',
        'patterns': [
            r'\b(restore\s+(my\s+)?access)\b',
            r'\b(locked\s+out|cannot\s+log\s*in|lost\s+access)\b',
            r'\b(not\s+(the\s+)?(workspace\s+)?owner|not\s+(an?\s+)?admin)\b',
            r'\b(remove[d]?\s+(my\s+)?seat)\b',
            r'\b(increase\s+(my\s+)?score)\b',
            r'\b(change\s+(my|the)\s+(grade|score|result))\b',
        ],
    },
    {
        'name': 'legal_regulatory',
        'description': 'Legal or regulatory request',
        'patterns': [
            r'\b(legal|lawyer|attorney|lawsuit|court|subpoena)\b',
            r'\b(regulatory|compliance|gdpr|ccpa|data\s+subject)\b',
            r'\b(infosec|security\s+questionnaire|security\s+assessment)\b',
            r'\b(vulnerability|bug\s+bounty|security\s+vulnerability)\b',
        ],
    },
    {
        'name': 'score_manipulation',
        'description': 'Request to manipulate test scores or override results',
        'patterns': [
            r'\b(increase\s+(my\s+)?score)\b',
            r'\b(graded?\s+(me\s+)?unfairly)\b',
            r'\b(review\s+my\s+answers?\s+.{0,30}(increase|change))\b',
            r'\b(tell\s+the\s+company\s+to\s+move\s+me)\b',
            r'\b(reschedul(e|ing)\s+(of\s+)?my\b)',
        ],
    },
]


def check_safety(issue: str, subject: str, retrieval_score: float = 10.0,
                 confidence_threshold: float = 5.0) -> dict:
    """
    Run all safety checks on a ticket.
    
    Args:
        issue: The ticket body text.
        subject: The ticket subject.
        retrieval_score: Best retrieval similarity score (0-1).
        confidence_threshold: Minimum retrieval score to consider corpus sufficient.
        
    Returns:
        {
            'should_escalate': bool,
            'reasons': list[str],  
            'is_injection': bool,
            'risk_level': str,  # 'low', 'medium', 'high'
        }
    """
    text = f"{issue} {subject}".lower()
    reasons = []
    risk_level = 'low'
    
    # Check for prompt injection
    is_injection = detect_injection(text)
    if is_injection:
        reasons.append('Prompt injection attempt detected')
        # Injection → invalid reply, not escalation
    
    # Check escalation rules
    for rule in ESCALATION_RULES:
        for pattern in rule['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                reasons.append(f"{rule['name']}: {rule['description']}")
                risk_level = 'high'
                break
    
    # Check corpus confidence
    if retrieval_score < confidence_threshold and not is_injection:
        reasons.append(f'Low corpus confidence (score: {retrieval_score:.3f})')
        if risk_level == 'low':
            risk_level = 'medium'
    
    # Determine escalation
    # Injection → NOT escalated (reply as invalid)
    # Real safety risks or low confidence → escalate
    should_escalate = len(reasons) > 0 and not is_injection
    
    return {
        'should_escalate': should_escalate,
        'reasons': reasons,
        'is_injection': is_injection,
        'risk_level': risk_level,
    }


def get_escalation_response(reasons: list[str]) -> str:
    """Generate a human-friendly escalation response."""
    if not reasons:
        return "This issue requires human review. We are escalating it to our support team."
    
    primary_reason = reasons[0].split(':')[0] if ':' in reasons[0] else reasons[0]
    
    responses = {
        'fraud_or_unauthorized': (
            "This issue involves potential fraud or unauthorized activity. "
            "For your security, we are escalating this to our specialized support team "
            "who can assist you immediately. Please do not share any sensitive information "
            "in this channel."
        ),
        'service_outage': (
            "We understand you're experiencing a service disruption. "
            "This has been escalated to our engineering team for immediate investigation. "
            "We apologize for the inconvenience."
        ),
        'billing_dispute': (
            "This billing matter requires human review to resolve properly. "
            "We are escalating this to our billing support team who will review your case "
            "and get back to you."
        ),
        'account_access_admin': (
            "This request requires account-level access that our automated system cannot provide. "
            "We are escalating this to a human support agent who can assist you."
        ),
        'legal_regulatory': (
            "This request involves legal, regulatory, or security compliance matters. "
            "We are escalating this to the appropriate team for proper handling."
        ),
        'score_manipulation': (
            "This request involves test scoring which requires human review. "
            "We are escalating this to our support team."
        ),
    }
    
    return responses.get(primary_reason, 
        "This issue requires specialized attention. We are escalating it to our support team "
        "for further review and resolution."
    )
