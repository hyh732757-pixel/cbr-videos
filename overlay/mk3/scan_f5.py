# f5 프레임 폴더가 있는 경기: 모든 우리팀 상자에 tri3 → tri3.json
import json,cv2,sys,os
sys.path.insert(0,'/home/claude/mk3'); from tri import tri3
fn='feat_labeled_before_gk.json' if os.path.exists('feat_labeled_before_gk.json') else 'feat_labeled.json'
f=json.load(open(fn)); out={}
for k,v in f.items():
    if not any(o.get('team2') in ('us','us_gk') and o.get('real',True) for o in v or []): continue
    im=cv2.imread(f'f5/{int(k)+1:05d}.jpg')
    if im is None: continue
    tags=[x['tag'] for x in v if x.get('tag')]
    out[k]={j:tri3(im,o['box'],tags) for j,o in enumerate(v) if o.get('team2') in ('us','us_gk') and o.get('real',True)}
json.dump(out,open('tri3.json','w')); print('done',len(out))
