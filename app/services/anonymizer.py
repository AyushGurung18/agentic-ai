"""
PII Anonymization & De-anonymization Service for Legal Documents.

Consistently replaces sensitive entity types (names, emails, phones,
corporate entities, identification numbers) with deterministic placeholders
like [PERSON_1], [ORG_1], [EMAIL_1], [PHONE_1], while maintaining a
strict, reversible bidirectional mapping.
"""

import re
from typing import Dict, Tuple, List, Any


class PIIAnonymizer:
    # Common legal company suffixes
    ORG_PATTERN = re.compile(
        r'\b([A-Z][A-Za-z0-9&.,\- ]+?\s+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|LLP|GmbH|PLC|Pvt\.?\s*Ltd\.?))\b'
    )
    
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    )
    
    PHONE_PATTERN = re.compile(
        r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,5}\b'
    )
    
    # Honorifics followed by Capitalized Names (e.g., Mr. John Smith, Dr. Alice Doe)
    PERSON_HONORIFIC_PATTERN = re.compile(
        r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Adv\.|Judge|Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    )
    
    # Legal party introductory pattern (e.g., "between Alice Johnson and Bob Smith")
    LEGAL_PARTY_PATTERN = re.compile(
        r'\b(?:by and between|entered into by|represented by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    )
    
    # Generic SSN / Tax ID patterns
    TAX_ID_PATTERN = re.compile(
        r'\b(?:\d{3}-\d{2}-\d{4}|\d{2}-\d{7})\b'
    )

    @classmethod
    def anonymize(cls, text: str) -> Tuple[str, Dict[str, str], Dict[str, List[str]]]:
        """
        Anonymizes the input text and returns:
        1. Anonymized text with placeholders
        2. Reversible mapping {placeholder: original_text}
        3. Breakdown of detected entities by category
        """
        mapping: Dict[str, str] = {}
        reverse_seen: Dict[str, str] = {}  # original -> placeholder (for consistent replacement)
        entities: Dict[str, List[str]] = {
            "emails": [],
            "phones": [],
            "organizations": [],
            "persons": [],
            "tax_ids": []
        }
        
        counts = {
            "EMAIL": 0,
            "PHONE": 0,
            "ORG": 0,
            "PERSON": 0,
            "TAX_ID": 0
        }

        # 1. Mask Emails
        def replace_email(match: re.Match) -> str:
            val = match.group(0)
            if val not in reverse_seen:
                counts["EMAIL"] += 1
                token = f"[EMAIL_{counts['EMAIL']}]"
                reverse_seen[val] = token
                mapping[token] = val
                entities["emails"].append(val)
            return reverse_seen[val]

        text = cls.EMAIL_PATTERN.sub(replace_email, text)

        # 2. Mask Phone Numbers
        def replace_phone(match: re.Match) -> str:
            val = match.group(0)
            if len(re.sub(r'\D', '', val)) >= 7:  # Avoid matching short section numbers
                if val not in reverse_seen:
                    counts["PHONE"] += 1
                    token = f"[PHONE_{counts['PHONE']}]"
                    reverse_seen[val] = token
                    mapping[token] = val
                    entities["phones"].append(val)
                return reverse_seen[val]
            return val

        text = cls.PHONE_PATTERN.sub(replace_phone, text)

        # 3. Mask Tax IDs / SSNs
        def replace_tax_id(match: re.Match) -> str:
            val = match.group(0)
            if val not in reverse_seen:
                counts["TAX_ID"] += 1
                token = f"[TAX_ID_{counts['TAX_ID']}]"
                reverse_seen[val] = token
                mapping[token] = val
                entities["tax_ids"].append(val)
            return reverse_seen[val]

        text = cls.TAX_ID_PATTERN.sub(replace_tax_id, text)

        # 4. Mask Organizations
        def replace_org(match: re.Match) -> str:
            val = match.group(1).strip()
            if val not in reverse_seen:
                counts["ORG"] += 1
                token = f"[ORG_{counts['ORG']}]"
                reverse_seen[val] = token
                mapping[token] = val
                entities["organizations"].append(val)
            return reverse_seen[val]

        text = cls.ORG_PATTERN.sub(replace_org, text)

        # 5. Mask Persons (with honorifics)
        def replace_person(match: re.Match) -> str:
            val = match.group(1).strip()
            if val not in reverse_seen:
                counts["PERSON"] += 1
                token = f"[PERSON_{counts['PERSON']}]"
                reverse_seen[val] = token
                mapping[token] = val
                entities["persons"].append(val)
            # Retain the honorific or wrap
            prefix = match.group(0).split()[0]
            return f"{prefix} {reverse_seen[val]}"

        text = cls.PERSON_HONORIFIC_PATTERN.sub(replace_person, text)

        return text, mapping, entities

    @classmethod
    def deanonymize(cls, text: str, mapping: Dict[str, str]) -> str:
        """
        Reverses placeholders in anonymized text back to original values.
        """
        if not mapping:
            return text
        
        # Sort tokens by length descending to prevent substring collisions
        sorted_tokens = sorted(mapping.keys(), key=len, reverse=True)
        pattern = re.compile('|'.join(re.escape(k) for k in sorted_tokens))
        return pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


anonymizer = PIIAnonymizer()
