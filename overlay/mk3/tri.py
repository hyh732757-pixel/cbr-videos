"""머리 위 삼각형 마커 색 읽기 v3 (연한 색 마커 포함): 주변 잔디 대비 밝고 다른 색 덩어리 중 아래로 좁아지는 모양"""
import cv2, numpy as np
def tri3(img, box, tags=()):
    x1,y1,x2,y2=map(int,box[:4]); cx=(x1+x2)//2; h=max(10,y2-y1)
    ry1,ry2=max(0,y1-int(max(50,0.9*h))),max(1,y1+int(0.05*h)); rx1,rx2=max(0,cx-30),min(img.shape[1],cx+30)
    r=img[ry1:ry2,rx1:rx2]
    if r.size==0: return None
    lab=cv2.cvtColor(r,cv2.COLOR_BGR2LAB).astype(np.float32)
    g=np.median(lab.reshape(-1,3),0)
    d=np.linalg.norm(lab-g,axis=2)
    m=((d>28)&(lab[...,0]>g[0]-10)).astype(np.uint8)   # 잔디와 색이 다르고 어둡지 않은 픽셀(검은 테두리·선수 몸 제외)
    for t in tags:   # 이름표 글자 제외
        tx1,ty1,tx2,ty2=t; m[max(0,ty1-ry1-4):max(0,ty2-ry1+4), max(0,tx1-rx1-30):max(0,tx2-rx1+6)]=0
    n,labs,st,cen=cv2.connectedComponentsWithStats(m)
    best=None
    for i in range(1,n):
        x,y,w,hh,a=st[i]
        if not(6<=a<=300 and 3<=w<=26 and 3<=hh<=24): continue
        ys,xs=np.nonzero(labs==i); top=(ys<=ys.min()+hh/3).sum(); bot=(ys>=ys.max()-hh/3).sum()
        if top<=bot*1.3: continue
        sc=abs(rx1+x+w/2-cx)+abs((ry1+y+hh)-(y1-0.25*h))*0.3
        if best is None or sc<best[0]: best=(sc,i)
    if best is None: return None
    px=r[labs==best[1]]
    hsv=cv2.cvtColor(px.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3); lb=cv2.cvtColor(px.reshape(-1,1,3),cv2.COLOR_BGR2LAB).reshape(-1,3)
    return [float(v) for v in np.median(hsv,0)]+[float(v) for v in np.median(lb,0)]+[int(len(px)), int(rx1+st[best[1]][0]+st[best[1]][2]/2), int(ry1+st[best[1]][1]+st[best[1]][3]/2)]
