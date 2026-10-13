import os, sys, json, time, re, random
import pandas as pd, litellm, dotenv
litellm.suppress_debug_info=True
dotenv.load_dotenv('/data/user_data/zheyuan4/research/code-preference/.env')
BASE='https://ai-gateway.andrew.cmu.edu/'
prompt_file, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPT=open(prompt_file).read()
SCHEMA={"type":"object","additionalProperties":False,"properties":{k:{"type":"string"} for k in ["topic","asked","corrects","condition","trigger","quote"]},"required":["topic","asked","corrects","condition","trigger","quote"]}
sample=json.load(open(os.environ.get('SAMPLE','sample.json')))
res=[]
for r in sample:
    prev=(r['prev_agent_reply'] or '')[-1500:]
    user=f"<agent_previous_reply>\n{prev}\n</agent_previous_reply>\n<developer_message>\n{' '.join(r['content'].split()[:300])}\n</developer_message>"
    for attempt in range(3):
        try:
            resp=litellm.completion(api_key=os.environ['LITELLM_API_KEY'],base_url=BASE,model=model,
                messages=[{"role":"system","content":PROMPT},{"role":"user","content":user}],
                response_format={"type":"json_schema","json_schema":{"name":"restate","strict":True,"schema":SCHEMA}},temperature=0)
            o=json.loads(resp.choices[0].message.content); break
        except Exception as e:
            o={"error":str(e)[:200]}; time.sleep(2**attempt)
    o['id']=r['id']; o['quote_ok']=o.get('quote','') in r['content']; res.append(o)
json.dump(res,open(out,'w'),indent=1,ensure_ascii=False)
for r,o in (zip(sample,res) if not os.environ.get("QUIET") else []):
    print(f"\n[{o['id']}] {r['user_id']} | RAW: {' '.join(r['content'].split())[:260]}")
    print(f"   topic={o.get('topic')!r}\n   asked={o.get('asked')!r}\n   corrects={o.get('corrects')!r} cond={o.get('condition')!r} trigger={o.get('trigger')} quote_ok={o['quote_ok']}")
