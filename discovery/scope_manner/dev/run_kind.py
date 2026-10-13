import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import litellm, dotenv
litellm.suppress_debug_info=True
dotenv.load_dotenv('/data/user_data/zheyuan4/research/code-preference/.env')
BASE='https://ai-gateway.andrew.cmu.edu/'
PROMPT=open('prompt_kind.txt').read(); model=sys.argv[1]
KEYS=["kind","manner","quote"]
SCHEMA={"type":"object","additionalProperties":False,"properties":{"kind":{"type":"string","enum":["task","manner","both"]},"manner":{"type":"string"},"quote":{"type":"string"}},"required":KEYS}
sample={r['id']:r for r in json.load(open('sample400.json'))}
res=[o for o in json.load(open('out_v3_400.json')) if 'error' not in o]
def one(o):
    r=sample[o['id']]; prev=(r['prev_agent_reply'] or '')[-1200:]
    user=f"<agent_previous_reply>\n{prev}\n</agent_previous_reply>\n<developer_message>\n{' '.join(r['content'].split()[:300])}\n</developer_message>\n<restatement>\n{o['asked']}\n</restatement>"
    for attempt in range(4):
        try:
            resp=litellm.completion(api_key=os.environ['LITELLM_API_KEY'],base_url=BASE,model=model,messages=[{"role":"system","content":PROMPT},{"role":"user","content":user}],
                response_format={"type":"json_schema","json_schema":{"name":"kind","strict":True,"schema":SCHEMA}},temperature=0)
            k=json.loads(resp.choices[0].message.content); break
        except Exception as e:
            k={"error":str(e)[:200]}; time.sleep(2**attempt)
    k['id']=o['id']; return k
with ThreadPoolExecutor(8) as ex: out=list(ex.map(one,res))
json.dump(out,open('out_kind_400.json','w'),indent=1,ensure_ascii=False)
print('done',len(out),'errors',sum(1 for k in out if 'error' in k))
