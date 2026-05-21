"""
Diferansiyel Robot Navigasyonu — Sıralı Çalışma
Robot haritayı bilmez; sadece hedefin konumunu bilir.
LiDAR ile engelleri keşfeder, anlık yeniden planlama yapar.
Yöntemler sırayla çalışır: D* → A* → Dijkstra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import heapq
import time
from collections import defaultdict

# ══════════════════════════════════════════════
# YÖNTEMLER ve RENKLER
# ══════════════════════════════════════════════
METHODS = ["D*", "A*", "Dijkstra", "RRT", "PRM"]
COLORS  = {"D*": "#4ECDC4", "A*": "#FF6B6B", "Dijkstra": "#45B7D1",
           "RRT": "#A29BFE", "PRM": "#FD79A8"}

# ══════════════════════════════════════════════
# PARAMETRELER
# ══════════════════════════════════════════════
WORLD_W, WORLD_H = 20.0, 20.0
START    = np.array([1.5,  1.5])
GOAL     = np.array([18.5, 18.5])
ROBOT_R  = 0.15
MAX_V    = 1.2
MAX_OMEGA= 2.5
DT       = 0.05
T_MAX    = 120.0

GRID_RES    = 0.5
GW          = int(WORLD_W / GRID_RES)
GH          = int(WORLD_H / GRID_RES)
PLAN_MARGIN = ROBOT_R               # robot yarıçapı kadar tampon yeterli

LIDAR_RANGE = 6.0
LIDAR_RAYS  = 72
LIDAR_NOISE = 0.05
IMU_NOISE   = 0.02
IMU_BIAS    = 0.005
ENC_NOISE   = 0.01
ENC_SLIP    = 0.002

# ══════════════════════════════════════════════
# HARİTA TANIMLARI  (cx, cy, genişlik, yükseklik)
# ══════════════════════════════════════════════
MAP_DEFS = {
    "Harita 1 — Dağınık": [
        ( 4.0, 16.5, 2.2, 1.6),
        ( 9.5, 13.5, 2.2, 1.6),
        (15.0, 15.0, 2.2, 1.6),
        ( 3.5,  9.5, 2.2, 1.6),
        ( 9.5,  9.5, 2.2, 1.6),
        (14.5,  9.5, 2.2, 1.6),
        (17.5, 11.0, 1.6, 2.2),
        ( 6.0,  4.5, 2.2, 1.6),
        (11.5,  4.0, 2.2, 1.6),
        (16.5,  5.5, 1.6, 2.2),
    ],
    "Harita 2 — Koridor": [
        # Yatay duvar üst bölge, ortada geçit bırakılmış
        ( 5.0, 15.0, 8.0, 1.2),
        (15.0, 15.0, 6.0, 1.2),
        # Yatay duvar orta bölge, iki geçit
        ( 2.5, 10.0, 3.5, 1.2),
        ( 9.0, 10.0, 4.0, 1.2),
        (16.5, 10.0, 3.5, 1.2),
        # Yatay duvar alt bölge, ortada geçit
        ( 2.5,  5.0, 6.0, 1.2),
        (13.0,  5.0, 7.0, 1.2),
        # Köşe engelleri
        ( 2.0, 17.5, 2.0, 2.0),
        (18.0,  2.5, 2.0, 2.0),
    ],
    "Harita 3 — Labirent": [
        # 1. Ana Dikey Duvar (Aşağıdan başlar, yukarıda geçiş bırakır)
        ( 7.0,  7.0, 1.0, 14.0),  
        
        # 2. Ana Dikey Duvar (Yukarıdan başlar, aşağıda geçiş bırakır)
        (13.0, 13.0, 1.0, 14.0),  
        
        # 3. Sol Koridor Engeli (1. Duvara tam temas eder, kesişmez)
        ( 4.5, 10.0, 4.0,  1.0),  
        
        # 4. Orta Koridor Engeli (2. Duvara tam temas eder, kesişmez)
        (12.0, 9.0, 3.0,  1.0),  
        
        # 5. Sağ Alt Ters "T" Şekli (Yatay zemin)
        (14.5,  6.0, 4.0,  1.0),  
        
        # 6. Sağ Alt Ters "T" Şekli (Dikey sütun, yataya tam oturur)
        (16.0, 11.5, 1.0, 10.0),  
    ],
}

OBSTACLES = list(MAP_DEFS.values())[0]   # varsayılan (seçim öncesi)


def select_map():
    """3 haritanın önizlemesini göster, kullanıcı seçsin."""
    map_names = list(MAP_DEFS.keys())
    chosen    = [map_names[0]]           # varsayılan

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('#0a0f1a')
    fig.suptitle('Bir harita seçin', color='white', fontsize=14, fontweight='bold')

    # Önizlemeler
    for ax, name in zip(axes, map_names):
        ax.set_facecolor('#0a1520')
        ax.set_xlim(0, WORLD_W); ax.set_ylim(0, WORLD_H)
        ax.set_aspect('equal')
        ax.set_title(name, color='white', fontsize=9, fontweight='bold', pad=4)
        ax.tick_params(colors='gray', labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor('#334')
        for (cx, cy, w, h) in MAP_DEFS[name]:
            ax.add_patch(patches.Rectangle(
                (cx - w/2, cy - h/2), w, h,
                lw=0.5, edgecolor='#889', facecolor='#2a3a5a', alpha=0.85))
        ax.plot(*START, 'o', color='lime',   ms=7, zorder=5)
        ax.plot(*GOAL,  '*', color='tomato', ms=9, zorder=5)
        ax.add_patch(patches.Rectangle((0, 0), WORLD_W, WORLD_H,
                                        lw=1, edgecolor='#446', facecolor='none'))

    # Seçim butonları
    btn_objs = []
    for idx, (ax, name) in enumerate(zip(axes, map_names)):
        ax_b = ax.inset_axes([0.05, -0.18, 0.90, 0.13])
        b = Button(ax_b, 'Seç', color='#1d4e89', hovercolor='#2980b9')
        b.label.set_color('white'); b.label.set_fontsize(9)

        def _cb(event, n=name):
            chosen[0] = n
            plt.close(fig)

        b.on_clicked(_cb)
        btn_objs.append(b)   # GC'den korumak için

    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.04, right=0.98, wspace=0.12)
    plt.show(block=True)
    return MAP_DEFS[chosen[0]]


# ══════════════════════════════════════════════
# GRID
# ══════════════════════════════════════════════
def world2cell(x, y):
    return (int(np.clip(y/GRID_RES, 0, GH-1)),
            int(np.clip(x/GRID_RES, 0, GW-1)))

def cell2world(r, c):
    return c*GRID_RES + GRID_RES/2, r*GRID_RES + GRID_RES/2

def in_obs(x, y, m=ROBOT_R):
    if x<m or x>WORLD_W-m or y<m or y>WORLD_H-m: return True
    for (cx,cy,w,h) in OBSTACLES:
        if abs(x-cx)<w/2+m and abs(y-cy)<h/2+m: return True
    return False

DIRS8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
COST8 = [1.0,1.0,1.0,1.0,1.414,1.414,1.414,1.414]

def get_nbrs(r, c, grid):
    for (dr,dc),cost in zip(DIRS8,COST8):
        nr,nc = r+dr,c+dc
        if 0<=nr<GH and 0<=nc<GW and not grid[nr,nc]:
            yield nr,nc,cost

_INF = int(np.ceil(PLAN_MARGIN/GRID_RES))

def lidar_update(known, rx, ry, rtheta, angles, dists):
    """Sadece gerçek vurma noktalarını işaretle — şişirme yok (görsel doğru)."""
    new_obs = set()
    wa = rtheta + angles
    for i in range(LIDAR_RAYS):
        if dists[i] >= LIDAR_RANGE - 0.2: continue
        hx = rx + dists[i]*np.cos(wa[i])
        hy = ry + dists[i]*np.sin(wa[i])
        hr, hc = world2cell(hx, hy)
        if 0 <= hr < GH and 0 <= hc < GW and not known[hr, hc]:
            known[hr, hc] = True
            new_obs.add((hr, hc))
    return new_obs


def inflate_obs(new_obs, plan):
    """Yeni engel hücrelerini PLAN_MARGIN kadar şişir → planlama gridi."""
    new_plan_obs = set()
    for r, c in new_obs:
        for dr in range(-_INF, _INF+1):
            for dc in range(-_INF, _INF+1):
                r2, c2 = r+dr, c+dc
                if 0 <= r2 < GH and 0 <= c2 < GW and not plan[r2, c2]:
                    plan[r2, c2] = True
                    new_plan_obs.add((r2, c2))
    return new_plan_obs

def path_blocked(cells, grid):
    return any(grid[r,c] for r,c in cells)

# ══════════════════════════════════════════════
# PLANLACILAR
# ══════════════════════════════════════════════
class DStarOnline:
    def __init__(self, start_rc, goal_rc):
        self.s0  = start_rc
        self.gr  = goal_rc
        self.grid= np.zeros((GH,GW),dtype=bool)
        self.km  = 0.0
        self.last= start_rc
        INF = float('inf')
        self.g   = defaultdict(lambda: INF)
        self.rhs = defaultdict(lambda: INF)
        self.rhs[goal_rc] = 0.0
        self._h  = []
        self._ih = {}
        self._push(*goal_rc)
        self._compute()

    def _heur(self,r,c):
        sr,sc=self.last; return np.hypot(r-sr,c-sc)

    def _key(self,r,c):
        v=min(self.g[(r,c)],self.rhs[(r,c)])
        return (v+self._heur(r,c)+self.km, v)

    def _push(self,r,c):
        k=self._key(r,c); heapq.heappush(self._h,(k,r,c)); self._ih[(r,c)]=k

    def _topk(self):
        while self._h:
            k,r,c=self._h[0]
            if self._ih.get((r,c))==k: return k
            heapq.heappop(self._h)
        return (float('inf'),float('inf'))

    def _pop(self):
        while self._h:
            k,r,c=heapq.heappop(self._h)
            if self._ih.get((r,c))==k:
                del self._ih[(r,c)]; return k,r,c
        return None,None,None

    def _upd(self,r,c):
        gr,gc=self.gr
        if (r,c)!=(gr,gc):
            self.rhs[(r,c)]=min((cost+self.g[(nr,nc)]
                for nr,nc,cost in get_nbrs(r,c,self.grid)),default=float('inf'))
        self._ih.pop((r,c),None)
        if self.g[(r,c)]!=self.rhs[(r,c)]: self._push(r,c)

    def _compute(self):
        sr,sc=self.last; INF=float('inf')
        for _ in range(100000):
            tk=self._topk(); ck=self._key(sr,sc)
            if tk>=ck and self.rhs[(sr,sc)]==self.g[(sr,sc)]: break
            k,r,c=self._pop()
            if r is None: break
            kn=self._key(r,c)
            if k<kn: self._push(r,c)
            elif self.g[(r,c)]>self.rhs[(r,c)]:
                self.g[(r,c)]=self.rhs[(r,c)]
                for nr,nc,cost in get_nbrs(r,c,self.grid): self._upd(nr,nc)
            else:
                self.g[(r,c)]=INF; self._upd(r,c)
                for nr,nc,cost in get_nbrs(r,c,self.grid): self._upd(nr,nc)

    def add_obs(self, cells):
        if not cells: return
        sr,sc=self.last; self.km+=self._heur(sr,sc)
        for r,c in cells:
            self.grid[r,c]=True; self._upd(r,c)
            for nr,nc,cost in get_nbrs(r,c,self.grid): self._upd(nr,nc)
        self._compute()

    def move(self, rc):
        old=self.last
        self.km+=np.hypot(rc[0]-old[0],rc[1]-old[1])
        self.last=rc

    def get_path(self):
        return astar(self.grid, self.last, self.gr)

def astar(grid, s, g):
    sr,sc=s; gr,gc=g; INF=float('inf')
    h=lambda r,c: np.hypot(r-gr,c-gc)
    heap=[(h(sr,sc),0.0,sr,sc)]; came={}
    gs=defaultdict(lambda:INF); gs[(sr,sc)]=0.0
    while heap:
        f,g_,r,c=heapq.heappop(heap)
        if (r,c)==(gr,gc):
            p=[]; n=(r,c)
            while n in came: p.append(n); n=came[n]
            p.append((sr,sc)); return p[::-1]
        if g_>gs[(r,c)]: continue
        for nr,nc,cost in get_nbrs(r,c,grid):
            ng=g_+cost
            if ng<gs[(nr,nc)]:
                gs[(nr,nc)]=ng; came[(nr,nc)]=(r,c)
                heapq.heappush(heap,(ng+h(nr,nc),ng,nr,nc))
    return None

def dijkstra(grid, s, g):
    sr,sc=s; gr,gc=g; INF=float('inf')
    heap=[(0.0,sr,sc)]; came={}
    dist=defaultdict(lambda:INF); dist[(sr,sc)]=0.0
    while heap:
        d,r,c=heapq.heappop(heap)
        if (r,c)==(gr,gc):
            p=[]; n=(r,c)
            while n in came: p.append(n); n=came[n]
            p.append((sr,sc)); return p[::-1]
        if d>dist[(r,c)]: continue
        for nr,nc,cost in get_nbrs(r,c,grid):
            nd=d+cost
            if nd<dist[(nr,nc)]:
                dist[(nr,nc)]=nd; came[(nr,nc)]=(r,c)
                heapq.heappush(heap,(nd,nr,nc))
    return None

def replan(method, known, s_rc, g_rc, dstar_obj):
    if method=="D*":
        return dstar_obj.get_path()
    if method=="A*":
        return astar(known, s_rc, g_rc)
    if method=="Dijkstra":
        return dijkstra(known, s_rc, g_rc)

def cells_to_world(cells):
    return [cell2world(r,c) for r,c in cells]

def smooth(wp, known):
    if len(wp)<3: return wp
    res=[wp[0]]; i=0
    while i<len(wp)-1:
        j=len(wp)-1
        while j>i+1:
            x1,y1=res[-1]; x2,y2=wp[j]
            ok=True
            for k in range(21):
                t=k/20
                r2,c2=world2cell(x1+t*(x2-x1),y1+t*(y2-y1))
                if known[r2,c2]: ok=False; break
            if ok: break
            j-=1
        res.append(wp[j]); i=j
    return res

def plen(pts):
    if len(pts)<2: return 0.0
    return float(sum(np.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1])
                     for i in range(len(pts)-1)))

# ══════════════════════════════════════════════
# RRT
# ══════════════════════════════════════════════
def _seg_free_rc(grid, r1, c1, r2, c2):
    """Grid üzerinde iki hücre arasındaki çizginin serbest olup olmadığını kontrol eder."""
    n = max(abs(r2-r1), abs(c2-c1), 1)
    for k in range(n+1):
        t = k/n
        r = int(np.clip(round(r1+t*(r2-r1)), 0, GH-1))
        c = int(np.clip(round(c1+t*(c2-c1)), 0, GW-1))
        if grid[r, c]:
            return False
    return True

def rrt(grid, start, goal, max_iter=3000, step=1):
    """
    RRT — grid üzerinde çalışır, (row,col) tuple'ları döner.
    step: hücre cinsinden adım uzunluğu
    """
    sr, sc = start; gr, gc = goal
    if grid[sr,sc] or grid[gr,gc]:
        return None

    nodes  = np.array([[sr, sc]], dtype=float)  # (N,2)
    parent = [-1]

    for _ in range(max_iter):
        # %15 hedef yönlü örnek, %85 rastgele
        if np.random.random() < 0.15:
            qr, qc = float(gr), float(gc)
        else:
            qr = np.random.uniform(0, GH-1)
            qc = np.random.uniform(0, GW-1)

        # En yakın düğüm (numpy ile hızlı)
        diffs = nodes - np.array([qr, qc])
        dists = np.hypot(diffs[:,0], diffs[:,1])
        ni    = int(np.argmin(dists))
        nr, nc = nodes[ni]

        # Adım at
        d = dists[ni]
        if d < 0.5: continue
        scale  = min(step/d, 1.0)
        new_r  = int(np.clip(round(nr + scale*(qr-nr)), 0, GH-1))
        new_c  = int(np.clip(round(nc + scale*(qc-nc)), 0, GW-1))

        if grid[new_r, new_c]: continue
        if not _seg_free_rc(grid, int(nr), int(nc), new_r, new_c): continue

        nodes  = np.vstack([nodes, [new_r, new_c]])
        parent.append(ni)

        # Hedefe ulaşıldı mı?
        if np.hypot(new_r-gr, new_c-gc) <= step+0.5:
            if _seg_free_rc(grid, new_r, new_c, gr, gc):
                nodes  = np.vstack([nodes, [gr, gc]])
                parent.append(len(nodes)-2)
                path = []
                idx  = len(nodes)-1
                while idx != -1:
                    r,c = nodes[idx]
                    path.append((int(r), int(c)))
                    idx = parent[idx]
                return path[::-1]

    return None

# ══════════════════════════════════════════════
# PRM
# ══════════════════════════════════════════════
def prm(grid, start, goal, n_samples=200, k=8):
    """
    PRM — bilinen grid üzerinde yol haritası oluşturur.
    Başlangıç (index 0) ve hedef (index 1) her çağrıda eklenir.
    """
    sr, sc = start; gr, gc = goal
    if grid[sr,sc] or grid[gr,gc]: return None

    # --- 1. Rastgele serbest düğüm örnekleme ---
    nodes = [(sr,sc), (gr,gc)]

    # Başlangıç ve hedef çevresine garantili düğümler — bağlantı kopukluğunu önler
    for (cr, cc) in [(sr,sc), (gr,gc)]:
        for dr, dc in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2),
                       (-1,0),(1,0),(0,-1),(0,1)]:
            r2 = int(np.clip(cr+dr, 0, GH-1))
            c2 = int(np.clip(cc+dc, 0, GW-1))
            if not grid[r2,c2] and (r2,c2) not in nodes:
                nodes.append((r2,c2))

    attempts = 0
    while len(nodes) < n_samples+2 and attempts < n_samples*5:
        attempts += 1
        r = np.random.randint(0, GH)
        c = np.random.randint(0, GW)
        if not grid[r,c]:
            nodes.append((r,c))

    nodes_arr = np.array(nodes, dtype=float)   # (N, 2)
    N = len(nodes_arr)

    # --- 2. Vektörleştirilmiş mesafe matrisi (O(N²) numpy, çok hızlı) ---
    INF   = float('inf')
    MAX_D = 10.0
    diff  = nodes_arr[:, np.newaxis, :] - nodes_arr[np.newaxis, :, :]  # (N,N,2)
    dist_mat = np.hypot(diff[:,:,0], diff[:,:,1])                       # (N,N)
    np.fill_diagonal(dist_mat, INF)

    nearest_all = np.argsort(dist_mat, axis=1)[:, :k]   # (N, k)

    # --- 3. Kenar inşası: tekrar eden çiftleri atla ---
    adj        = [[] for _ in range(N)]
    seen_edges = set()
    for i in range(N):
        for j in nearest_all[i]:
            if dist_mat[i,j] > MAX_D: continue
            edge = (min(i,j), max(i,j))
            if edge in seen_edges: continue
            seen_edges.add(edge)
            r1,c1 = int(nodes_arr[i,0]), int(nodes_arr[i,1])
            r2,c2 = int(nodes_arr[j,0]), int(nodes_arr[j,1])
            if _seg_free_rc(grid, r1,c1, r2,c2):
                cost = float(dist_mat[i,j])
                adj[i].append((j, cost))
                adj[j].append((i, cost))

    # --- 4. A* ile roadmap üzerinde yol bul ---
    h    = lambda i: np.hypot(nodes_arr[i,0]-gr, nodes_arr[i,1]-gc)
    heap = [(h(0), 0.0, 0)]
    came = {}
    gs   = [INF]*N; gs[0] = 0.0

    while heap:
        f, g_, i = heapq.heappop(heap)
        if i == 1:
            path = []
            cur  = i
            while cur in came:
                path.append((int(nodes_arr[cur,0]), int(nodes_arr[cur,1])))
                cur = came[cur]
            path.append((sr, sc))
            return path[::-1]
        if g_ > gs[i]: continue
        for j, cost in adj[i]:
            ng = g_ + cost
            if ng < gs[j]:
                gs[j] = ng; came[j] = i
                heapq.heappush(heap, (ng+h(j), ng, j))

    return None

# ══════════════════════════════════════════════
# SENSÖRLER + EKF
# ══════════════════════════════════════════════
def lidar(rx,ry,rth):
    ang=np.linspace(0,2*np.pi,LIDAR_RAYS,endpoint=False)
    wa=rth+ang; ca=np.cos(wa); sa=np.sin(wa)
    d=np.full(LIDAR_RAYS,LIDAR_RANGE); act=np.ones(LIDAR_RAYS,dtype=bool)
    r=0.1
    while r<=LIDAR_RANGE and act.any():
        px=rx+r*ca; py=ry+r*sa
        hits=np.zeros(LIDAR_RAYS,dtype=bool)
        for (cx,cy,w,h) in OBSTACLES:
            hits|=(np.abs(px-cx)<w/2)&(np.abs(py-cy)<h/2)
        hits|=(px<=0)|(px>=WORLD_W)|(py<=0)|(py>=WORLD_H)
        nh=hits&act; d[nh]=r; act[nh]=False; r+=0.1
    d+=np.random.normal(0,LIDAR_NOISE,LIDAR_RAYS)
    return ang, np.clip(d,0,LIDAR_RANGE)

def _wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

class EKF:
    def __init__(self,x,y,t):
        self.mu=np.array([x,y,t],dtype=float)
        self.P=np.diag([.1,.1,.05])
        self.Q=np.diag([ENC_NOISE**2+1e-4]*2+[IMU_NOISE**2+1e-4])
        self.R=np.eye(2)*(LIDAR_NOISE*4)**2
    def predict(self,v,w,dt):
        x,y,th=self.mu
        self.mu=np.array([x+v*np.cos(th)*dt,y+v*np.sin(th)*dt,_wrap(th+w*dt)])
        F=np.array([[1,0,-v*np.sin(th)*dt],[0,1,v*np.cos(th)*dt],[0,0,1]])
        self.P=F@self.P@F.T+self.Q
    def update(self,zx,zy):
        H=np.array([[1,0,0],[0,1,0]],dtype=float)
        inn=np.array([zx,zy])-H@self.mu
        S=H@self.P@H.T+self.R; K=self.P@H.T@np.linalg.inv(S)
        self.mu+=K@inn; self.mu[2]=_wrap(self.mu[2])
        self.P=(np.eye(3)-K@H)@self.P

# ══════════════════════════════════════════════
# KONTROLCÜ
# ══════════════════════════════════════════════
class Ctrl:
    SD=0.55; RK=1.8
    def __init__(self): self.wp=[START.copy()]; self.idx=0
    def set_path(self,wp,pos):
        self.wp=[np.array(p) for p in wp]
        if len(self.wp)>1:
            ds=[np.hypot(pos[0]-p[0],pos[1]-p[1]) for p in self.wp]
            self.idx=max(0,int(np.argmin(ds)))
        else: self.idx=0
    def step(self,rx,ry,rth,ld):
        if len(self.wp)<2: return 0.0,0.0
        while self.idx<len(self.wp)-1:
            if np.hypot(rx-self.wp[self.idx][0],ry-self.wp[self.idx][1])<0.5:
                self.idx+=1
            else: break
        tgt=self.wp[self.idx]
        ga=np.arctan2(tgt[1]-ry,tgt[0]-rx); he=_wrap(ga-rth)
        ra=np.linspace(0,2*np.pi,LIDAR_RAYS,endpoint=False)
        fm=np.abs(_wrap(ra))<np.pi/3
        mf=float(np.min(ld[fm])) if fm.any() else LIDAR_RANGE
        sk=float(np.clip((mf-ROBOT_R*2)/1.5,0.0,1.0))
        # LEFT rays: ra in (0,pi)  RIGHT rays: ra in (pi,2pi)
        lm=(ra>0)&(ra<np.pi); rm=(ra>np.pi)&(ra<2*np.pi)
        dl=float(np.min(ld[lm])) if lm.any() else LIDAR_RANGE  # left dist
        dr=float(np.min(ld[rm])) if rm.any() else LIDAR_RANGE  # right dist
        rep=0.0
        if dl<self.SD: rep-=self.RK*(self.SD-dl)   # LEFT obstacle → turn right (negative omega)
        if dr<self.SD: rep+=self.RK*(self.SD-dr)   # RIGHT obstacle → turn left (positive omega)
        v=MAX_V*sk*max(0.0,float(np.cos(he)))
        w=float(np.clip(2.8*he+rep,-MAX_OMEGA,MAX_OMEGA))
        return v,w

# ══════════════════════════════════════════════
# SİMÜLASYON ADIMI (tüm yöntemler için ortak)
# ══════════════════════════════════════════════
def step_sim_state(s, goal_rc):
    if s['done']: return
    method = s['method']

    ang, ld = lidar(s['rx'], s['ry'], s['rth'])
    s['la'], s['ld'] = ang, ld

    # known = görsel (şişirilmemiş), plan = planlama için şişirilmiş
    new_obs = lidar_update(s['known'], s['rx'], s['ry'], s['rth'], ang, ld)
    cur_rc  = world2cell(s['rx'], s['ry'])

    if new_obs:
        new_plan = inflate_obs(new_obs, s['plan'])   # planlama gridini güncelle
        if method == "D*":
            s['ds'].add_obs(new_plan)
            s['ds'].move(cur_rc)
        if len(s['pcells']) == 0 or path_blocked(s['pcells'], s['plan']):
            if   method == "D*":  raw = s['ds'].get_path()
            elif method == "A*":  raw = astar(s['plan'], cur_rc, goal_rc)
            elif method == "RRT": raw = rrt(s['plan'], cur_rc, goal_rc)
            elif method == "PRM": raw = prm(s['plan'], cur_rc, goal_rc)
            else:                 raw = dijkstra(s['plan'], cur_rc, goal_rc)
            if raw:
                s['pcells'] = raw
                wp = smooth(cells_to_world(raw), s['plan'])
                s['pworld'] = wp
                s['ctrl'].set_path(wp, (s['rx'], s['ry']))
                s['rc'] += 1
    elif method == "D*":
        s['ds'].move(cur_rc)

    if len(s['pcells']) == 0:
        s['ctrl'].set_path([(s['rx'], s['ry']), tuple(GOAL)], (s['rx'], s['ry']))
        s['pworld'] = []

    v, w  = s['ctrl'].step(s['rx'], s['ry'], s['rth'], ld)
    ve    = v + np.random.normal(0, ENC_NOISE) + ENC_SLIP*v*np.random.uniform(-1,1)
    wi    = w + np.random.normal(0, IMU_NOISE) + IMU_BIAS

    s['ekf'].predict(ve, wi, DT)
    md = float(np.min(ld))
    if md < LIDAR_RANGE*0.9:
        ns = 0.06 + 0.10*(md/LIDAR_RANGE)
        s['ekf'].update(s['rx']+np.random.normal(0,ns), s['ry']+np.random.normal(0,ns))

    s['drx'] += ve*np.cos(s['drt'])*DT
    s['dry'] += ve*np.sin(s['drt'])*DT
    s['drt'] += wi*DT

    nx = s['rx'] + v*np.cos(s['rth'])*DT
    ny = s['ry'] + v*np.sin(s['rth'])*DT
    if not in_obs(nx, ny, ROBOT_R): s['rx'], s['ry'] = nx, ny
    s['rth'] = _wrap(s['rth'] + w*DT)
    s['t']  += DT

    s['dxs'].append(s['drx'])
    s['dys'].append(s['dry'])
    s['rths'].append(s['rth'])
    s['ekf_ths'].append(s['ekf'].mu[2])
    if not s['lidar_scan_log'] or s['t'] - s['lidar_scan_log'][-1]['t'] >= 1.0:
        s['lidar_scan_log'].append({'angles': ang.copy(), 'raw': ld.copy(), 't': s['t']})

    cur = np.array([s['rx'], s['ry']])
    s['real_len'] += float(np.linalg.norm(cur - s['prev']))
    s['prev'] = cur.copy()
    s['txs'].append(s['rx']); s['tys'].append(s['ry'])
    s['exs'].append(s['ekf'].mu[0]); s['eys'].append(s['ekf'].mu[1])
    s['tlog'].append(s['t'])
    s['e_ekf'].append(np.hypot(s['ekf'].mu[0]-s['rx'], s['ekf'].mu[1]-s['ry']))
    s['e_dr'].append(np.hypot(s['drx']-s['rx'], s['dry']-s['ry']))

    if np.hypot(s['rx']-GOAL[0], s['ry']-GOAL[1]) < 0.6:
        s['reached'] = True; s['done'] = True
    if s['t'] >= T_MAX: s['done'] = True

    if s['t'] - s['stuck_t'] > 2.5:
        moved = float(np.hypot(s['rx']-s['stuck_pos'][0], s['ry']-s['stuck_pos'][1]))
        if moved < 0.25:
            s['pcells'] = []
            s['rth'] = _wrap(s['rth'] + np.pi*0.5)
        s['stuck_pos'] = np.array([s['rx'], s['ry']])
        s['stuck_t']   = s['t']


def _make_state(method, goal_rc):
    rx, ry, rth = START[0], START[1], np.deg2rad(45)
    start_rc = world2cell(rx, ry)
    ds = DStarOnline(start_rc, goal_rc) if method == "D*" else None
    return {
        'method': method,
        'rx': rx, 'ry': ry, 'rth': rth,
        'drx': rx, 'dry': ry, 'drt': rth,
        'ekf': EKF(rx+np.random.normal(0,.08),
                   ry+np.random.normal(0,.08),
                   rth+np.random.normal(0,.03)),
        'ctrl': Ctrl(),
        'known': np.zeros((GH,GW), dtype=bool),  # görsel (şişirilmemiş)
        'plan':  np.zeros((GH,GW), dtype=bool),  # planlama (şişirilmiş)
        'ds': ds,
        'pcells': [], 'pworld': [],
        'rc': 0, 't': 0.0,
        'reached': False, 'done': False,
        'txs': [rx], 'tys': [ry],
        'exs': [rx], 'eys': [ry],
        'e_ekf': [0.0], 'e_dr': [0.0],
        'tlog': [0.0],
        'dxs': [rx], 'dys': [ry],       # dead-reckoning yol
        'rths': [rth],                   # gerçek yön geçmişi
        'ekf_ths': [rth],                # EKF yön geçmişi
        'lidar_scan_log': [],            # ham & filtrelenmiş LiDAR anlık görüntüler
        'real_len': 0.0,
        'prev': np.array([rx, ry]),
        'stuck_pos': np.array([rx, ry]),
        'stuck_t': 0.0,
        'la': np.linspace(0, 2*np.pi, LIDAR_RAYS, endpoint=False),
        'ld': np.full(LIDAR_RAYS, LIDAR_RANGE),
    }


def _setup_ax(ax, method):
    """Subplot'u hazırla, statik artist'leri ekle."""
    color = COLORS[method]
    ax.set_facecolor('#0a1520')
    ax.set_xlim(0, WORLD_W); ax.set_ylim(0, WORLD_H)
    ax.set_aspect('equal')
    ax.tick_params(colors='gray', labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor('#334')

    for (cx,cy,w,h) in OBSTACLES:
        ax.add_patch(patches.Rectangle(
            (cx-w/2,cy-h/2), w, h,
            lw=0.5, edgecolor='#556', facecolor='#1a1a35', alpha=0.55, zorder=1))
    ax.add_patch(patches.Rectangle(
        (0,0), WORLD_W, WORLD_H,
        lw=1.0, edgecolor='#446', facecolor='none', zorder=2))
    ax.plot(*START, 'o', color='lime',   ms=6, zorder=6)
    ax.plot(*GOAL,  '*', color='tomato', ms=9, zorder=6)

    kg  = ax.imshow(np.zeros((GH,GW)), origin='lower',
                    extent=[0,WORLD_W,0,WORLD_H],
                    cmap='Reds', vmin=0, vmax=1,
                    alpha=0.35, zorder=2, interpolation='nearest')
    lt, = ax.plot([], [], lw=1.1, color=color, zorder=5)
    le, = ax.plot([], [], '--', color='#FFEAA7', lw=0.6, alpha=0.5, zorder=4)
    pl, = ax.plot([], [], '-',  color=color,    lw=0.8, alpha=0.6, zorder=4)
    circ = plt.Circle(START, ROBOT_R, facecolor=color, zorder=7, alpha=0.92)
    ax.add_patch(circ)
    dl, = ax.plot([], [], color='white', lw=1.4, zorder=8)
    rays = [ax.plot([],[],color='deepskyblue',lw=0.3,alpha=0.4,zorder=3)[0]
            for _ in range(LIDAR_RAYS)]
    ttl = ax.set_title(method, fontsize=9, fontweight='bold', color=color, pad=2)
    sts = ax.text(0.5, 0.02, "Başlıyor...", transform=ax.transAxes,
                  ha='center', va='bottom', fontsize=6.5, color='white',
                  bbox=dict(facecolor='#0a1520', alpha=0.75,
                            edgecolor='none', boxstyle='round'))
    return dict(kg=kg, lt=lt, le=le, pl=pl, circ=circ, dl=dl, rays=rays,
                ttl=ttl, sts=sts)


def _update_art(s, a):
    a['kg'].set_data(s['known'].astype(float))   # gerçek boyut, şişirilmemiş
    a['lt'].set_data(s['txs'], s['tys'])
    a['le'].set_data(s['exs'], s['eys'])
    if s['pworld']:
        a['pl'].set_data([p[0] for p in s['pworld']],
                         [p[1] for p in s['pworld']])
    a['circ'].set_center((s['rx'], s['ry']))
    th = s['rth']
    a['dl'].set_data([s['rx'], s['rx']+ROBOT_R*2.5*np.cos(th)],
                     [s['ry'], s['ry']+ROBOT_R*2.5*np.sin(th)])
    wa = s['rth'] + s['la']
    for i, ray in enumerate(a['rays']):
        ray.set_data([s['rx'], s['rx']+s['ld'][i]*np.cos(wa[i])],
                     [s['ry'], s['ry']+s['ld'][i]*np.sin(wa[i])])
    t_el, rc = s['t'], s['rc']
    if s['reached']:
        txt = f"ULAŞILDI  {t_el:.1f}s | {s['real_len']:.1f}m | {rc} replan"
        a['sts'].set_color('lime')
    elif s['done']:
        txt = f"BAŞARISIZ | {rc} replan"
        a['sts'].set_color('tomato')
    else:
        txt = f"t={t_el:.1f}s | {rc} replan | EKF={s['e_ekf'][-1]:.2f}m"
        a['sts'].set_color('white')
    a['sts'].set_text(txt)


# ══════════════════════════════════════════════
# DETAY ANALİZ PENCERESİ
# ══════════════════════════════════════════════
def show_method_detail(method, s):
    color = COLORS[method]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0f1a')
    fig.suptitle(f'{method} — Detaylı Analiz', color=color, fontsize=14, fontweight='bold')

    gs2 = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34,
                           top=0.92, bottom=0.08, left=0.07, right=0.98)

    # ── 1. LiDAR polar: ham + filtrelenmiş ──────────────────────────────
    ax_lidar = fig.add_subplot(gs2[0, 0], projection='polar')
    if s['lidar_scan_log']:
        scan  = s['lidar_scan_log'][-1]
        raw_d = scan['raw'].copy()
        angs  = scan['angles'].copy()
    else:
        raw_d = s['ld'].copy()
        angs  = s['la'].copy()
    fw = 5
    padded = np.pad(raw_d, fw // 2, mode='wrap')
    filtered_d = np.array([np.median(padded[i:i + fw]) for i in range(len(raw_d))])
    ax_lidar.set_facecolor('#0a1520')
    ax_lidar.plot(angs, raw_d, color='cyan', lw=0.8, alpha=0.7, label='Ham')
    ax_lidar.fill(angs, raw_d, color='cyan', alpha=0.15)
    ax_lidar.plot(angs, filtered_d, color='orange', lw=1.2, label='Filtrelenmiş')
    ax_lidar.set_ylim(0, LIDAR_RANGE)
    ax_lidar.set_title('LiDAR Tarama', color='white', fontsize=9, pad=8)
    ax_lidar.tick_params(colors='gray', labelsize=6)
    ax_lidar.legend(fontsize=7, loc='lower right',
                    labelcolor='white', framealpha=0.2, facecolor='#0a1520')
    ax_lidar.grid(color='gray', alpha=0.2)

    # ── 2. 2D yol karşılaştırması ────────────────────────────────────────
    ax2d = fig.add_subplot(gs2[0, 1])
    ax2d.set_facecolor('#0a1520')
    ax2d.set_xlim(0, WORLD_W); ax2d.set_ylim(0, WORLD_H)
    ax2d.set_aspect('equal')
    for (cx, cy, ow, oh) in OBSTACLES:
        ax2d.add_patch(patches.Rectangle(
            (cx - ow/2, cy - oh/2), ow, oh,
            lw=0.5, edgecolor='#556', facecolor='#1a1a35', alpha=0.7))
    ax2d.plot(s['txs'], s['tys'], '-', color=color,     lw=1.0, label='Gerçek yol', zorder=5)
    ax2d.plot(s['exs'], s['eys'], '--', color='#FFEAA7', lw=0.8, alpha=0.8, label='EKF tahmini', zorder=4)
    if len(s['dxs']) > 1:
        ax2d.plot(s['dxs'], s['dys'], ':', color='#74b9ff', lw=0.7, alpha=0.7, label='Ölü hesap', zorder=3)
    ax2d.plot(*START, 'o', color='lime',   ms=7, zorder=6)
    ax2d.plot(*GOAL,  '*', color='tomato', ms=10, zorder=6)
    ax2d.set_title('2D Yol Karşılaştırması', color='white', fontsize=9)
    ax2d.legend(fontsize=7, loc='upper left',
                labelcolor='white', framealpha=0.3, facecolor='#0a1520')
    ax2d.tick_params(colors='gray', labelsize=6)
    for sp in ax2d.spines.values(): sp.set_edgecolor('#334')

    # ── 3. Konum hatası zamanla + RMSE/MAE ──────────────────────────────
    ax_err = fig.add_subplot(gs2[0, 2])
    ax_err.set_facecolor('#0a1520')
    tlog   = s['tlog']
    ekf_e  = s['e_ekf']
    dr_e   = s['e_dr']
    ne = min(len(tlog), len(ekf_e), len(dr_e))
    ax_err.plot(tlog[:ne], ekf_e[:ne], color='#FFEAA7', lw=1.0, label='EKF hatası')
    ax_err.plot(tlog[:ne], dr_e[:ne],  '--', color='#74b9ff', lw=0.8, label='DR hatası')
    rmse_ekf = float(np.sqrt(np.mean(np.array(ekf_e[:ne])**2))) if ne else 0.0
    mae_ekf  = float(np.mean(np.abs(ekf_e[:ne])))              if ne else 0.0
    rmse_dr  = float(np.sqrt(np.mean(np.array(dr_e[:ne])**2))) if ne else 0.0
    mae_dr   = float(np.mean(np.abs(dr_e[:ne])))               if ne else 0.0
    if ne:
        mid_t = tlog[ne // 2]
        ax_err.axhline(rmse_ekf, color='#FFEAA7', ls=':', lw=0.8, alpha=0.5)
        ax_err.text(mid_t, rmse_ekf + 0.01, f'RMSE={rmse_ekf:.3f}m',
                    color='#FFEAA7', fontsize=7)
    stats_txt = (f"EKF  RMSE={rmse_ekf:.3f}m  MAE={mae_ekf:.3f}m\n"
                 f"DR   RMSE={rmse_dr:.3f}m   MAE={mae_dr:.3f}m")
    ax_err.text(0.02, 0.97, stats_txt, transform=ax_err.transAxes,
                va='top', color='white', fontsize=7,
                bbox=dict(facecolor='#0a1520', alpha=0.7, edgecolor='#334', boxstyle='round'))
    ax_err.set_xlabel('t (s)', color='gray', fontsize=8)
    ax_err.set_ylabel('Hata (m)', color='gray', fontsize=8)
    ax_err.set_title('Konum Hatası', color='white', fontsize=9)
    ax_err.legend(fontsize=7, labelcolor='white', framealpha=0.3, facecolor='#0a1520')
    ax_err.tick_params(colors='gray', labelsize=6)
    ax_err.grid(True, ls='--', alpha=0.15, color='white')
    for sp in ax_err.spines.values(): sp.set_edgecolor('#334')

    # ── 4. x(t) karşılaştırma ───────────────────────────────────────────
    ax_x = fig.add_subplot(gs2[1, 0])
    ax_x.set_facecolor('#0a1520')
    nt = min(len(tlog), len(s['txs']), len(s['exs']))
    nd = min(len(tlog), len(s['dxs']))
    ax_x.plot(tlog[:nt], s['txs'][:nt], color=color,     lw=1.0, label='Gerçek x')
    ax_x.plot(tlog[:nt], s['exs'][:nt], '--', color='#FFEAA7', lw=0.8, label='EKF x')
    if nd > 1:
        ax_x.plot(tlog[:nd], s['dxs'][:nd], ':', color='#74b9ff', lw=0.7, label='DR x')
    ax_x.set_xlabel('t (s)', color='gray', fontsize=8)
    ax_x.set_ylabel('x (m)', color='gray', fontsize=8)
    ax_x.set_title('x(t) Karşılaştırma', color='white', fontsize=9)
    ax_x.legend(fontsize=7, labelcolor='white', framealpha=0.3, facecolor='#0a1520')
    ax_x.tick_params(colors='gray', labelsize=6)
    ax_x.grid(True, ls='--', alpha=0.15, color='white')
    for sp in ax_x.spines.values(): sp.set_edgecolor('#334')

    # ── 5. y(t) karşılaştırma ───────────────────────────────────────────
    ax_y = fig.add_subplot(gs2[1, 1])
    ax_y.set_facecolor('#0a1520')
    ny = min(len(tlog), len(s['tys']), len(s['eys']))
    ndy = min(len(tlog), len(s['dys']))
    ax_y.plot(tlog[:ny], s['tys'][:ny], color=color,     lw=1.0, label='Gerçek y')
    ax_y.plot(tlog[:ny], s['eys'][:ny], '--', color='#FFEAA7', lw=0.8, label='EKF y')
    if ndy > 1:
        ax_y.plot(tlog[:ndy], s['dys'][:ndy], ':', color='#74b9ff', lw=0.7, label='DR y')
    ax_y.set_xlabel('t (s)', color='gray', fontsize=8)
    ax_y.set_ylabel('y (m)', color='gray', fontsize=8)
    ax_y.set_title('y(t) Karşılaştırma', color='white', fontsize=9)
    ax_y.legend(fontsize=7, labelcolor='white', framealpha=0.3, facecolor='#0a1520')
    ax_y.tick_params(colors='gray', labelsize=6)
    ax_y.grid(True, ls='--', alpha=0.15, color='white')
    for sp in ax_y.spines.values(): sp.set_edgecolor('#334')

    # ── 6. θ(t) karşılaştırma ───────────────────────────────────────────
    ax_th = fig.add_subplot(gs2[1, 2])
    ax_th.set_facecolor('#0a1520')
    nrth = min(len(tlog), len(s['rths']))
    neth = min(len(tlog), len(s['ekf_ths']))
    ax_th.plot(tlog[:nrth], np.rad2deg(s['rths'][:nrth]),    color=color,     lw=1.0, label='Gerçek θ')
    ax_th.plot(tlog[:neth], np.rad2deg(s['ekf_ths'][:neth]), '--', color='#FFEAA7', lw=0.8, label='EKF θ')
    ax_th.set_xlabel('t (s)', color='gray', fontsize=8)
    ax_th.set_ylabel('θ (°)', color='gray', fontsize=8)
    ax_th.set_title('θ(t) Karşılaştırma', color='white', fontsize=9)
    ax_th.legend(fontsize=7, labelcolor='white', framealpha=0.3, facecolor='#0a1520')
    ax_th.tick_params(colors='gray', labelsize=6)
    ax_th.grid(True, ls='--', alpha=0.15, color='white')
    for sp in ax_th.spines.values(): sp.set_edgecolor('#334')

    plt.show(block=False)
    fig.canvas.draw()


# ══════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════
def run():
    global OBSTACLES
    OBSTACLES = select_map()

    np.random.seed(42)
    goal_rc     = world2cell(*GOAL)
    all_results = []

    # ── Figure: 2×3 ızgara (5 yöntem + boş) ───────────────────────────
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor('#0a0f1a')
    gs  = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.18,
                           top=0.94, bottom=0.09, left=0.03, right=0.99)

    pos = {"D*":(0,0), "A*":(0,1), "Dijkstra":(0,2), "RRT":(1,0), "PRM":(1,1)}
    axs  = {m: fig.add_subplot(gs[r,c]) for m,(r,c) in pos.items()}
    arts = {m: _setup_ax(axs[m], m)     for m in METHODS}

    # 6. hücre: boş arka plan
    ax6 = fig.add_subplot(gs[1,2])
    ax6.set_facecolor('#0a0f1a')
    ax6.set_xlim(0,1); ax6.set_ylim(0,1)
    ax6.axis('off')
    ax6.text(0.5, 0.97, 'Detaylı Analiz', transform=ax6.transAxes,
             ha='center', va='top', color='#aaa', fontsize=9, fontweight='bold')

    # ── Karşılaştırma butonu ───────────────────────────────────────────
    ax_btn = fig.add_axes([0.36, 0.01, 0.28, 0.055])
    btn    = Button(ax_btn, 'Grafikleri Göster',
                    color='#1d4e89', hovercolor='#2980b9')
    btn.label.set_color('white'); btn.label.set_fontsize(10)
    btn.on_clicked(lambda e: show_graphs(all_results) if all_results else None)

    # ── Yöntem detay butonları (sağ alt panel) ─────────────────────────
    # Approximate position of gs[1,2]: x≈0.71-0.99, y≈0.09-0.46
    detail_btns = {}
    btn_labels  = ['D*', 'A*', 'Dijkstra', 'RRT', 'PRM']
    btn_colors  = [COLORS[m] for m in btn_labels]
    btn_y_start = 0.41   # top of first button
    btn_h       = 0.054
    btn_gap     = 0.010
    for i, (label, bc) in enumerate(zip(btn_labels, btn_colors)):
        by = btn_y_start - i * (btn_h + btn_gap)
        ax_b = fig.add_axes([0.715, by, 0.265, btn_h])
        b    = Button(ax_b, label, color='#14213d', hovercolor='#1d4e89')
        b.label.set_color(bc)
        b.label.set_fontsize(9)
        b.label.set_fontweight('bold')
        detail_btns[label] = b

    # ── Simülasyon durumları ───────────────────────────────────────────

    states = {m: _make_state(m, goal_rc) for m in METHODS}

    # Detay butonlarına callback bağla (states hazır olduğunda)
    def _make_detail_cb(m):
        return lambda e: show_method_detail(m, states[m])
    for m in METHODS:
        detail_btns[m].on_clicked(_make_detail_cb(m))

    EVERY = 2   # paralel çalışma — frame başına adım sayısı

    def animate(frame):
        for method in METHODS:
            s = states[method]
            if s['done']: continue
            try:
                for _ in range(EVERY):
                    step_sim_state(s, goal_rc)
            except Exception as e:
                import traceback; traceback.print_exc()
                s['done'] = True

            _update_art(s, arts[method])

            if s['done'] and not any(r['method']==method for r in all_results):
                rmse = float(np.sqrt(np.mean(np.array(s['e_ekf'])**2)))
                all_results.append({
                    'method': method, 'success': s['reached'],
                    'time':   s['t'] if s['reached'] else None,
                    'real_len': s['real_len'],
                    'rc': s['rc'], 'rmse': rmse,
                })
                tag = 'OK' if s['reached'] else 'FAIL'
                print(f"{method}: {tag}  t={s['t']:.1f}s  "
                      f"yol={s['real_len']:.1f}m  replan={s['rc']}")

        if all(states[m]['done'] for m in METHODS):
            fig.suptitle("Tüm yöntemler tamamlandı! →  Grafikleri Göster",
                         color='white', fontsize=11)

        fig.canvas.flush_events()

    max_f = int(T_MAX/DT/EVERY) + 300
    anim  = FuncAnimation(fig, animate, frames=max_f,
                          interval=50, blit=False, repeat=False)
    plt.show()


def show_graphs(results):
    if not results: return
    fig2,axes=plt.subplots(1,3,figsize=(14,5))
    fig2.patch.set_facecolor('#1a1a2e')
    fig2.suptitle("Yöntem Karşılaştırması",color='white',
                  fontsize=13,fontweight='bold')
    methods=[r['method'] for r in results]
    colors =[COLORS[m] for m in methods]
    for ax,(data,title,ylabel) in zip(axes,[
        ([r['real_len'] for r in results],"Gerçek Yol Uzunluğu","m"),
        ([r['time'] if r['time'] else 0 for r in results],"Ulaşma Süresi","s"),
        ([r['rc'] for r in results],"Yeniden Planlama Sayısı",""),
    ]):
        ax.set_facecolor('#16213e')
        bars=ax.bar(methods,data,color=colors,edgecolor='white',linewidth=0.5,width=0.5)
        ax.set_title(title,color='white',fontsize=11,fontweight='bold')
        ax.set_ylabel(ylabel,color='white',fontsize=9)
        ax.tick_params(colors='white',labelsize=9)
        ax.grid(True,ls='--',alpha=0.2,color='white',axis='y')
        for sp in ax.spines.values(): sp.set_edgecolor('gray')
        mx=max(data) if max(data)>0 else 1
        for bar,val,res in zip(bars,data,results):
            lbl=f'{val:.1f}'
            if not res['success']: lbl+='\n✗'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+mx*0.02,lbl,
                    ha='center',va='bottom',color='white',fontsize=9)
    fig2.tight_layout(); fig2.patch.set_facecolor('#1a1a2e')
    plt.show(block=False); fig2.canvas.draw()


if __name__=="__main__":
    run()
