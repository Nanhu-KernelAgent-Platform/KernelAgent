"""Direction-scoped generation text channels; implementation seeds stay joint."""
from dataclasses import dataclass

from .signature import build_operator_signature
from .store import format_experience_context


@dataclass
class GenerationMemory:
    context: str
    seeds: list
    root_event: str
    text_events: list[str]


def generation_memory(store, source, *, direction, platform, backend, session_id, logger):
    signature = build_operator_signature(source, trace_graph=True, execution_direction=direction)
    # No forward/backward seed is ever admitted into a joint implementation.
    seeds = store.search_verified_seeds(signature, platform, backend, limit=2, direction=direction)
    if direction == 'unknown':
        seeds = []
    channels = [('joint', signature),
                ('forward', build_operator_signature(source, trace_graph=True, execution_direction='forward')),
                ('backward', build_operator_signature(source, trace_graph=True, execution_direction='backward'))] if direction == 'joint' else [(direction, signature)]
    sections, events, seen = [], [], set()
    root_event = ''
    remaining_budget = 11000 # reserve headings; unused channel budgets transfer
    for channel_index, (channel, probe) in enumerate(channels):
        records = store.retrieve(probe, platform, backend, stage='generation',
                                 limit=2 if direction == 'joint' else 3, direction=channel)
        records = [item for item in records if item.id not in seen]
        budget = remaining_budget // (len(channels)-channel_index) if direction == 'joint' else 12000
        kept = []
        for item in records:
            trial = format_experience_context(kept + [item], max_chars=1000000, compact=True)
            if len(trial) <= budget:
                kept.append(item)
        records = kept
        context = format_experience_context(records, max_chars=budget, compact=True)
        if direction == 'joint':
            remaining_budget -= len(context)
        seen.update(item.id for item in records)
        event = store.log_retrieval(session_id=session_id,
            stage='generation' if channel == direction else f'generation_{channel}_text',
            signature=probe, records=records, seeds=seeds if channel == direction else [], direction=channel)
        if channel == direction:
            root_event = event
        else:
            events.append(event)
        if records:
            if direction == 'joint':
                sections.append(f'## {channel.upper()} text experience channel\n'
                                'Apply only within the explicitly stated direction and conditions; '
                                'no source code or gradient interface transfer.\n' + context)
            else:
                sections.append(context)
        logger.info('Generation experience channel %s: %d text records, %d connected graph regions, %d exact graph contracts; graph=%s',
                    channel, len(records), sum(r.retrieval_match == 'substructure' and bool(r.graph_match_evidence) for r in records),
                    sum(r.retrieval_reason == 'full_graph_contract_text_only' for r in records), probe.graph_version or 'unavailable')
    if direction == 'joint':
        sections.insert(0, '## Direction-isolated experience for joint generation\n'
                        'Generate one complete bound forward/backward implementation. These are '
                        'text-only scoped hypotheses, not separate code seeds. Missing backward '
                        'graph/evidence is unknown, never inferred from the forward graph.')
    return GenerationMemory('\n\n'.join(sections), seeds, root_event, events)
