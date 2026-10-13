import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import litellm, dotenv
litellm.suppress_debug_info=True
dotenv.load_dotenv('/data/user_data/zheyuan4/research/code-preference/.env')
BASE='https://ai-gateway.andrew.cmu.edu/'
prompt_file, model, sample_file, out = sys.argv[1:5]
PROMPT=open(prompt_file).read()
KEYS=["topic","asked","corrects","condition","trigger","quote"]
SCHEMA={"type":"object","additionalProperties":False,"properties":{k:{"type":"string"} for k in KEYS},"required":KEYS}
sample=json.load(open(sample_file))
def one(r):
    prev=(r['prev_agent_reply'] or '')[-1500:]
    user=f"<agent_previous_reply>\n{prev}\n</agent_previous_reply>\n<developer_message>\n{' '.join(r['content'].split()[:300])}\n</developer_message>"
    for attempt in range(4):
        try:
            resp=litellm.completion(api_key=os.environ['LITELLM_API_KEY'],base_url=BASE,model=model,
                messages=[{"role":"system","content":PROMPT},{"role":"user","content":user}],
                response_format={"type":"json_schema","json_schema":{"name":"restate","strict":True,"schema":SCHEMA}},temperature=0)
            o=json.loads(resp.choices[0].message.content); break
        except Exception as e:
            o={"error":str(e)[:200]}; time.sleep(2**attempt)
    o['id']=r['id']; return o
with ThreadPoolExecutor(8) as ex: res=list(ex.map(one,sample))
json.dump(res,open(out,'w'),indent=1,ensure_ascii=False)
print('done',len(res),'errors',sum(1 for o in res if 'error' in o))
