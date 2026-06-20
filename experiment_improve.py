import numpy as np, torch, time
from env import ParticleBoxEnv
from encoder import ParticleEncoder, pretrain_encoder
from bio_loop import BioWorldModelLoop, calculate_centroid
from eval_harness import FrameStacker, get_actual_future_latent_path
import cl_sdk as cl

np.random.seed(0); torch.manual_seed(0)
device='cpu'
H=8
STEPS_TRAIN=1000
STEPS_EVAL=200

# ---------- Shared perception ----------
env=ParticleBoxEnv()
encoder=ParticleEncoder(in_channels=3).to(device)
pretrain_encoder(encoder, num_samples=800, epochs=15)
encoder.eval()
stacker=FrameStacker(num_frames=3)

# ---------- Continuous velocity-aware linear world model ----------
# state s=[x,y,vx,vy] (grid units), input u=[s(4), action_onehot(4), 1] -> next s (4)
# Learned online by ridge-regularized recursive least squares.
class LinearWM:
    def __init__(self, d_in=9, d_out=4, ridge=1e-2):
        self.W=np.zeros((d_in,d_out))
        self.P=np.eye(d_in)/ridge
    def feat(self,s,a):
        oh=np.zeros(4); oh[a]=1.0
        return np.concatenate([s,oh,[1.0]])
    def update(self,s,a,s_next):
        x=self.feat(s,a)
        Px=self.P@x
        denom=1.0+x@Px
        K=Px/denom
        pred=x@self.W
        self.W+=np.outer(K,(s_next-pred))
        self.P-=np.outer(K,Px)
    def predict(self,s,a):
        return self.feat(s,a)@self.W

wm=LinearWM()

# ---------- Training: drive env, learn BNN (CONFIG_C) AND linear WM on the SAME stream ----------
noisy,_,_=env.reset(); stacker.reset()
cur=stacker.push_and_get(noisy)
action=np.random.choice([0,1,2,3])
prev_c=None
with cl.open() as loop:
    bio=BioWorldModelLoop(loop, use_boundary_penalty=True, use_velocity_decoding=True, gain_factor=1.2)
    for step in range(1,STEPS_TRAIN+1):
        if step%15==0: action=np.random.choice([0,1,2,3])
        s_curr_t=torch.tensor(cur,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            zc=torch.softmax(encoder(s_curr_t).squeeze(0),0).numpy()
        c_curr=calculate_centroid(zc)
        nn,_,_,col=env.step(action); nxt=stacker.push_and_get(nn)
        s_next_t=torch.tensor(nxt,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            zn=torch.softmax(encoder(s_next_t).squeeze(0),0).numpy()
        c_next=calculate_centroid(zn)
        # velocity estimate from consecutive centroids
        vx,vy=(c_next-c_curr)
        # BNN learns (its own internal target)
        y_target=np.array([c_next[0],c_next[1],env.dx/8.0,env.dy/8.0],dtype=np.float32)
        bio.predict_and_learn(zc,action,zn,y_target,boundary_penalty=col)
        # Linear WM learns continuous transition with velocity in the state
        if prev_c is not None:
            s=np.array([c_curr[0],c_curr[1],c_curr[0]-prev_c[0],c_curr[1]-prev_c[1]])
            s_next=np.array([c_next[0],c_next[1],vx,vy])
            wm.update(s,action,s_next)
        prev_c=c_curr
        cur=nxt

# ---------- Evaluation: compare 3 predictors on IDENTICAL true future paths ----------
def mspe(pred,true): return np.mean(np.sum((pred-true)**2,axis=2),axis=0)
def pathvar(p):
    m=np.mean(p,axis=1,keepdims=True); return np.mean(np.sum((p-m)**2,axis=2),axis=1)

true_log=[]; bnn_log=[]; lin_log=[]; cv_log=[]; start_log=[]
encoder.eval()
prev_c=None
for step in range(1,STEPS_EVAL+1):
    if step%15==0: action=np.random.choice([0,1,2,3])
    s_curr_t=torch.tensor(cur,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        zc=torch.softmax(encoder(s_curr_t).squeeze(0),0).numpy()
    c0=calculate_centroid(zc); start_log.append(c0)
    vel0=(c0-prev_c) if prev_c is not None else np.array([env.dx/8.0,env.dy/8.0])

    true_path=get_actual_future_latent_path(env,encoder,cur,action,horizon=H,device=device)

    # (1) BNN CONFIG_C rollout
    bnn_traj=bio.predict_trajectory(zc,[action]*H,np.array([env.dx/8.0,env.dy/8.0]),horizon=H)
    bnn_path=np.array([calculate_centroid(z) for z in bnn_traj])

    # (2) Learned continuous linear WM rollout (velocity-aware)
    s=np.array([c0[0],c0[1],vel0[0],vel0[1]]); lp=[]
    for h in range(H):
        s=wm.predict(s,action)
        s[0]=np.clip(s[0],0.1,7.9); s[1]=np.clip(s[1],0.1,7.9)
        lp.append(s[:2].copy())
    lin_path=np.array(lp)

    # (3) Constant-velocity dead reckoning (physics-free baseline)
    cv=[]; p=c0.copy(); v=vel0.copy()
    for h in range(H):
        p=np.clip(p+v,0.1,7.9); cv.append(p.copy())
    cv_path=np.array(cv)

    true_log.append(true_path); bnn_log.append(bnn_path); lin_log.append(lin_path); cv_log.append(cv_path)
    nn,_,_,_=env.step(action); cur=stacker.push_and_get(nn); prev_c=c0

true_log=np.array(true_log); bnn_log=np.array(bnn_log); lin_log=np.array(lin_log); cv_log=np.array(cv_log)
base=np.repeat(np.expand_dims(np.array(start_log),1),H,axis=1)

def report(name,pred):
    m=mspe(pred,true_log)
    tvd=np.mean(pathvar(pred))/(np.mean(pathvar(true_log))+1e-6)
    eph=sum(1 for h in range(H) if m[h]<mspe(base,true_log)[h])
    print(f'{name:28s} MSPE@1={m[0]:6.3f} MSPE@8={m[7]:6.3f} mean={m.mean():6.3f} | TVD={tvd:5.3f} | EPH={eph}')
    return m

print('\n'+'='*78)
print('TRUE per-step motion magnitude (grid cells):',round(np.mean(np.linalg.norm(np.diff(true_log,axis=1),axis=2)),3))
print('='*78)
report('Unmoving baseline',base)
report('Constant-velocity dead-reckon',cv_log)
report('BNN CONFIG_C (current)',bnn_log)
report('Learned continuous linear WM',lin_log)

# ============================================================================
# HONEST EVALUATION: ground truth = real physical (x,y), not encoder centroids
# ============================================================================
print('\n'+'='*78)
print('HONEST METRIC (ground truth = true physical position in grid units)')
print('='*78)
np.random.seed(1)
env2=ParticleBoxEnv(); noisy,_,_=env2.reset(); st2=FrameStacker(3); cur2=st2.push_and_get(noisy)
action=np.random.choice([0,1,2,3]); prev_c=None
true_p=[]; lin_p=[]; cv_p=[]; base_p=[]
for step in range(1,STEPS_EVAL+1):
    if step%15==0: action=np.random.choice([0,1,2,3])
    with torch.no_grad():
        zc=torch.softmax(encoder(torch.tensor(cur2,dtype=torch.float32).unsqueeze(0)).squeeze(0),0).numpy()
    c0=calculate_centroid(zc)
    vel0=(c0-prev_c) if prev_c is not None else np.array([env2.dx/8.0,env2.dy/8.0])
    # true physical future
    ox,oy,odx,ody=env2.x,env2.y,env2.dx,env2.dy
    tp=[]
    for h in range(H):
        env2.step(action); tp.append(np.array([env2.x/8.0,env2.y/8.0]))
    env2.x,env2.y,env2.dx,env2.dy=ox,oy,odx,ody
    true_p.append(np.array(tp))
    # learned linear WM rollout
    s=np.array([c0[0],c0[1],vel0[0],vel0[1]]); lp=[]
    for h in range(H):
        s=wm.predict(s,action); s[0]=np.clip(s[0],0.1,7.9); s[1]=np.clip(s[1],0.1,7.9); lp.append(s[:2].copy())
    lin_p.append(np.array(lp))
    # constant velocity (use TRUE physical velocity as the honest dead-reckoner)
    p=np.array([env2.x/8.0,env2.y/8.0]); v=np.array([env2.dx/8.0,env2.dy/8.0]); cv=[]
    for h in range(H):
        p=np.clip(p+v,0.0,8.0); cv.append(p.copy())
    cv_p.append(np.array(cv))
    base_p.append(np.repeat(c0[None,:],H,axis=0))
    env2.step(action); cur2=st2.push_and_get(env2._get_frame()[0]); prev_c=c0

true_p=np.array(true_p); lin_p=np.array(lin_p); cv_p=np.array(cv_p); base_p=np.array(base_p)
for name,pred in [('Unmoving baseline',base_p),('Constant-velocity (true v)',cv_p),('Learned linear WM',lin_p)]:
    m=mspe(pred,true_p)
    print(f'{name:28s} MSPE@1={m[0]:6.3f} MSPE@8={m[7]:6.3f} mean={m.mean():6.3f}')
