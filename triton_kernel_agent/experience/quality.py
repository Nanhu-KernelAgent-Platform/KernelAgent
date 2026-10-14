"""Conservative text quality and redundancy gates, never evidence promotion."""
import re
import unicodedata
import json

VERSION = 'scoped_text_quality_v1'
_SUMMARY_ONLY = re.compile(r'(?:applied .+\.\s*)?performance changed by [-+]?\d+(?:\.\d+)?%[.!\s]*')


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def tokens(text):
    return set(re.findall(r'[\w]+', normalized(text)))


def informative(text, *, error=False):
    value = normalized(text)
    if not value:
        return False
    if _SUMMARY_ONLY.fullmatch(value):
        return False
    boilerplate = re.compile(r'^(?:generated (?:a )?(?:correct|correctness[- ]verified) kernel|'
                            r'(?:the )?kernel (?:is correct|passed verification|generated successfully)|'
                            r'optimization (?:completed|succeeded)|生成了?正确(?:的)?\s*kernel|'
                            r'生成成功|优化完成|正确性验证通过)[.!。！\s]*$')
    words = tokens(value)
    generic = {'a','an','the','and','is','was','kernel','generation','generated','produced',
               'correct','correctly','correctness','verified','verification','passed','success',
               'successful','successfully','succeeded','optimization','completed','complete'}
    if words <= generic and words & {'kernel','optimization'}:
        return False
    return not boilerplate.fullmatch(value) and (
        len(tokens(value)) >= (2 if error else 3)
        or len(re.findall(r'[\u4e00-\u9fff]', value)) >= 10)


def useful(record):
    if informative(record.error_message, error=True):
        return True
    reflection = record.environment.get('raw_reflection') or {}
    if isinstance(reflection, dict):
        if reflection.get('generation_status') == 'failed':
            return False
        # Read-time compatibility for old unmarked fallback reflections. No
        # old row is rewritten, and an actual failure diagnostic remains useful.
        reasoning = normalized(str(reflection.get('reasoning') or ''))
        if (not reflection.get('lessons') and not reflection.get('claims')
                and _SUMMARY_ONLY.fullmatch(reasoning)):
            return False
    return (informative(record.lesson)
            or (informative(record.action) and informative(record.bottleneck)))


def scope_key(record):
    signature = record.signature
    encode = lambda value: json.dumps(value, sort_keys=True, default=str)
    graph = (signature.graph_version, signature.graph_digest, encode(signature.interface)) if signature.graph_digest else (
        signature.semantic_type, tuple(signature.dtypes), tuple(map(tuple, signature.shapes)))
    return (record.kind, record.platform, record.kernel_backend, record.direction,
            tuple(record.applicable_directions), record.text_evidence_status,
            record.claim_type, encode(record.claim_conditions), record.hardware, record.layout,
            normalized(record.action), encode(record.config_changes), graph)


def same_idea(left, right):
    """Suggestion/suppression only; retain quantities and negation, no aliases."""
    a, b = tokens(left), tokens(right)
    protected = lambda text: (set(re.findall(r'[-+]?\d+(?:\.\d+)?%?', text)),
        tokens(text) & {'not', 'no', 'never', 'without', 'avoid', 'slower', 'faster', '不', '禁止'})
    stop = {'a','an','the','this','that','these','those','is','are','was','were','be','been'}
    return bool(a and b and a-stop == b-stop and protected(left) == protected(right)
                and len(a & b) / len(a | b) >= .85)


def redundant(record, selected):
    text = record.lesson or record.error_message
    return any(scope_key(record) == scope_key(other)
               and same_idea(text, other.lesson or other.error_message) for other in selected)
