import re

def parse_constitution_text(text: str) -> dict:
    """Parse a constitution text and extract rules and parameters."""
    rules = {}
    text_lower = text.lower()
    # Simple regex-based extraction
    rules['ubi'] = 'universal basic income' in text_lower
    rules['recall'] = 'recall' in text_lower
    rules['proportional'] = 'proportional' in text_lower
    rules['no_supreme_court'] = 'no supreme court' in text_lower
    # Try to extract election interval
    match = re.search(r'elected every (\d+) years', text_lower)
    if match:
        rules['election_interval'] = int(match.group(1))
    else:
        rules['election_interval'] = 4
    # Advanced NLP extraction
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        # Extract rights (freedom, equality, etc.)
        rights = set()
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ['freedom', 'equality', 'speech', 'press', 'association']):
                rights.add(sent.text.strip())
        rules['rights'] = list(rights)
        # Extract powers (executive, legislative, judicial)
        powers = {'executive': [], 'legislative': [], 'judicial': []}
        for sent in doc.sents:
            for branch in powers:
                if branch in sent.text.lower():
                    powers[branch].append(sent.text.strip())
        rules['powers'] = powers
        # Extract amendments
        amendments = [sent.text.strip() for sent in doc.sents if 'amendment' in sent.text.lower()]
        rules['amendments'] = amendments
        # Extract entities
        rules['entities'] = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    except ImportError:
        # Fallback: regex for rights and amendments
        rights = re.findall(r'(freedom|equality|speech|press|association)', text_lower)
        rules['rights'] = list(set(rights))
        amendments = re.findall(r'amendment[^.\n]*', text_lower)
        rules['amendments'] = amendments
        rules['powers'] = {}
        rules['entities'] = []
    return rules

def compare_constitutions(text1: str, text2: str) -> dict:
    """Compare two constitutions and highlight differences in structure and extracted rules."""
    rules1 = parse_constitution_text(text1)
    rules2 = parse_constitution_text(text2)
    diff = {}
    all_keys = set(rules1.keys()).union(rules2.keys())
    for key in all_keys:
        v1 = rules1.get(key)
        v2 = rules2.get(key)
        if v1 != v2:
            diff[key] = {'constitution1': v1, 'constitution2': v2}
    return diff 