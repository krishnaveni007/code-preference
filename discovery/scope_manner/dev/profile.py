import json, collections, itertools, numpy as np
msgs=[json.loads(l) for l in open('/data/user_data/zheyuan4/research/code-preference/outputs/discovery/scope/gate/positives_dedup.jsonl')]
R={o['id']:o for o in (json.loads(l) for l in open('full_v4.jsonl')) if 'error' not in o}
S=[o for o in [json.loads(l) for l in open('stance_0.3.jsonl')] if 'error' not in o and o.get('one_matter')]
def level(a,b):
    A,B=msgs[a],msgs[b]
    if A['user_id']!=B['user_id']: return 'population'
    if A['session_id']==B['session_id']: return 'session'
    if A['repo_id']==B['repo_id']: return 'repo'
    return 'user'
rows=[]
for o in S:
    st={}; 
    for g in o.get('stances',[]):
        for i in g.get('members',[]):
            try: st[int(i)]=g['label']
            except: pass
    opp={frozenset(p) for p in o.get('opposite',[]) if isinstance(p,list) and len(p)==2}
    mem=[i for i in st if st[i]!='off-matter' and i in R]
    cnt={l:[0,0,0] for l in ['session','repo','user','population']}  # pairs, same, opposite
    for a,b in itertools.combinations(mem,2):
        l=level(a,b); cnt[l][0]+=1
        if st[a]==st[b]: cnt[l][1]+=1
        elif frozenset([st[a],st[b]]) in opp: cnt[l][2]+=1
    users=len({msgs[i]['user_id'] for i in mem}); labels=collections.Counter(st[i] for i in mem)
    rows.append(dict(cluster=o['cluster'],matter=o.get('matter',''),n=len(mem),users=users,n_stances=len(labels),n_opp=len(opp),cnt=cnt,labels=labels))
json.dump(rows,open('profiles_0.3.json','w'),indent=1,ensure_ascii=False)
def fmt(c): return ' | '.join(f"{l[:4]} {c[l][2]}/{c[l][0]}" for l in ['session','repo','user','population']) 
print('matters:',len(rows),'| with >=2 stances:',sum(1 for r in rows if r['n_stances']>1),'| with opposite stances:',sum(1 for r in rows if r['n_opp']>0))
print('\n=== matters with opposite stances and >=2 users (opposite/pairs per level) ===')
for r in sorted([r for r in rows if r['n_opp']>0 and r['users']>=2],key=lambda r:-r['n'])[:25]:
    print(f"[{r['cluster']}] {r['matter'][:55]:55s} n={r['n']:3d} users={r['users']:2d} | {fmt(r['cnt'])}")
    print('      stances:', '; '.join(f"{k[:50]} ({v})" for k,v in r['labels'].most_common(4)))
print('\n=== pooled over all matters: opposite rate per level ===')
tot={l:[0,0,0] for l in ['session','repo','user','population']}
for r in rows:
    for l in tot:
        for k in range(3): tot[l][k]+=r['cnt'][l][k]
for l in tot: print(f"  {l:10s} pairs={tot[l][0]:6d} same={tot[l][1]/max(1,tot[l][0]):.2f} opposite={tot[l][2]/max(1,tot[l][0]):.3f}")
