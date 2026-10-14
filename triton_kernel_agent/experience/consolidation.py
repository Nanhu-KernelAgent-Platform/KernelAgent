"""Reviewed, reversible claim aliases. Raw claims/evidence are never moved."""
import json
from datetime import datetime, timezone

from .quality import same_idea

SCHEMA = '''CREATE TABLE IF NOT EXISTS experiment_claim_aliases (
    source_id TEXT NOT NULL REFERENCES experiment_claims(id),
    target_id TEXT NOT NULL REFERENCES experiment_claims(id),
    reviewer TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
    revoked_by TEXT NOT NULL DEFAULT '', revocation_reason TEXT NOT NULL DEFAULT '',
    revoked_at TEXT NOT NULL DEFAULT '', PRIMARY KEY(source_id,created_at),
    CHECK(source_id != target_id));
    CREATE UNIQUE INDEX IF NOT EXISTS idx_active_claim_alias
    ON experiment_claim_aliases(source_id) WHERE revoked_at='';'''

FIELDS = ('claim_type','predicate','conditions_json','scope','semantic_type',
          'platform','kernel_backend','applicable_directions_json')


def compatible(a,b):
    return all((json.loads(a[k]) == json.loads(b[k]) if k.endswith('_json') else a[k] == b[k]) for k in FIELDS)


def roots(connection):
    links = dict(connection.execute("select source_id,target_id from experiment_claim_aliases where revoked_at=''"))
    def root(key):
        seen=set()
        while key in links:
            if key in seen:
                raise ValueError('claim alias cycle')
            seen.add(key);key=links[key]
        return key
    return {key:root(key) for key in links}


def review_merge(connection,source_id,target_id,*,reviewer,reason):
    if not isinstance(reviewer,str) or not isinstance(reason,str) or not reviewer.strip() or not reason.strip():
        raise ValueError('reviewer and reason are required')
    rows={r['id']:r for r in connection.execute('select * from experiment_claims where id in (?,?)',(source_id,target_id))}
    if source_id==target_id or len(rows)!=2 or not compatible(rows[source_id],rows[target_id]):
        raise ValueError('claims must have identical type, predicate, conditions, scope, direction and domain')
    links=roots(connection)
    if source_id in links or links.get(target_id,target_id)==source_id:
        raise ValueError('already aliased source or cyclic merge')
    connection.execute('insert into experiment_claim_aliases (source_id,target_id,reviewer,reason,created_at) values (?,?,?,?,?)',
        (source_id,links.get(target_id,target_id),reviewer.strip(),reason.strip(),datetime.now(timezone.utc).isoformat()))


def review_unmerge(connection,source_id,*,reviewer,reason):
    if not isinstance(reviewer,str) or not isinstance(reason,str) or not reviewer.strip() or not reason.strip():
        raise ValueError('reviewer and reason are required')
    cursor=connection.execute("update experiment_claim_aliases set revoked_by=?,revocation_reason=?,revoked_at=? where source_id=? and revoked_at=''",
        (reviewer.strip(),reason.strip(),datetime.now(timezone.utc).isoformat(),source_id))
    return bool(cursor.rowcount)


def proposals(connection,limit=30):
    if limit <= 0: return []
    rows=connection.execute('select * from experiment_claims order by created_at desc limit 500').fetchall()
    aliased=roots(connection);result=[]
    for i,a in enumerate(rows):
        if a['id'] in aliased: continue
        for b in rows[i+1:]:
            if b['id'] not in aliased and compatible(a,b) and same_idea(a['text'],b['text']):
                result.append({'source_id':a['id'],'target_id':b['id'],
                    'source_text':a['text'],'target_text':b['text'],'requires_review':True})
                if len(result)>=limit:return result
    return result
