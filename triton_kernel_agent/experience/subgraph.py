"""Bounded injective matching of an entire recorded ATen region into a DAG.

Not symbolic equivalence: exact overloads, literals, operand positions and
tensor shape/dtype/stride are retained. Boundary inputs can bind intermediate
values. A whole-task lesson is never attributed to an arbitrary shared fragment.
"""
from __future__ import annotations

from collections import Counter

from .models import OperatorSignature
from .graph_identity import canonical_nodes

MATCH_VERSION = "aten_connected_region_v1"
MAX_NODES = 256
MAX_STATES = 20000


def contained_region(source: OperatorSignature, target: OperatorSignature) -> dict:
    if (not source.graph_version or source.graph_version != target.graph_version
            or not source.graph_tensor_specs or not target.graph_tensor_specs):
        return {}
    a, b = source.graph_nodes, target.graph_nodes
    if not a or not b or max(len(a), len(b)) > MAX_NODES:
        return {}
    a = canonical_nodes(a, source.graph_tensor_specs)
    b = canonical_nodes(b, target.graph_tensor_specs)
    region = [i for i, n in enumerate(a) if n['op'] not in {'input', 'output', 'constant'}]
    if not region or len(region) >= sum(n['op'] not in {'input', 'output', 'constant'} for n in b):
        return {}
    if any(not a[i]['op'].startswith('aten::') for i in region):
        return {}
    wanted = Counter(a[i]['op'] for i in region)
    available = Counter(n['op'] for n in b)
    if any(available[op] < count for op, count in wanted.items()):
        return {}
    def refs(value):
        if isinstance(value, dict):
            if set(value) == {'node'}:
                yield value['node']
            else:
                for v in value.values():
                    yield from refs(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from refs(v)
    # Connectivity excludes unrelated multi-output task lessons.
    neighbors = {i: set() for i in region}
    for i in region:
        for j in refs([a[i]['args'], a[i]['kwargs']]):
            if j in neighbors:
                neighbors[i].add(j); neighbors[j].add(i)
    reached, pending = set(), [region[0]]
    while pending:
        i = pending.pop()
        if i not in reached:
            reached.add(i); pending.extend(neighbors[i] - reached)
    if len(reached) != len(region):
        return {}
    states = 0
    def bind(i, j, mapping):
        nonlocal states
        states += 1
        if states > MAX_STATES or not (0 <= i < len(a) and 0 <= j < len(b)):
            return False
        if i in mapping:
            return mapping[i] == j
        if j in mapping.values():
            return False
        left = source.graph_tensor_specs.get(str(i))
        right = target.graph_tensor_specs.get(str(j))
        # Constants lacking value provenance are deliberately not matched.
        if a[i]['op'] == 'constant' or left is None or right is None or left != right:
            return False
        if a[i]['op'] == 'input':
            if b[j]['op'] in {'output', 'constant'}:
                return False
            mapping[i] = j
            return True
        if a[i]['op'] != b[j]['op']:
            return False
        mapping[i] = j
        return equal(a[i]['args'], b[j]['args'], mapping) and equal(a[i]['kwargs'], b[j]['kwargs'], mapping)
    def equal(x, y, mapping):
        if isinstance(x, dict) and set(x) == {'node'}:
            return isinstance(y, dict) and set(y) == {'node'} and bind(x['node'], y['node'], mapping)
        if type(x) is not type(y):
            return False
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(v, y[k], mapping) for k, v in x.items())
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(equal(u, v, mapping) for u, v in zip(x, y))
        return x == y
    def visit(index, mapping):
        if index == len(region):
            return mapping
        i = region[index]
        if i in mapping:
            return visit(index + 1, mapping)
        for j, node in enumerate(b):
            if states >= MAX_STATES:
                break
            if node['op'] == a[i]['op']:
                trial = mapping.copy()
                if bind(i, j, trial):
                    result = visit(index + 1, trial)
                    if result is not None:
                        return result
        return None
    mapping = visit(0, {})
    if mapping is None:
        return {}
    return {'version': MATCH_VERSION, 'source_nodes': region,
            'target_nodes': [mapping[i] for i in region],
            'boundary_mapping': {str(i): j for i, j in mapping.items() if i not in region},
            'operators': [a[i]['op'] for i in region],
            'compatibility': 'exact_shape_dtype_stride_literals',
            'scope': 'entire_recorded_region_text_only'}
