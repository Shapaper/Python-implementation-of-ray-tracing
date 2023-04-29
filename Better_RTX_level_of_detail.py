import copy
import random
import time
from numba import jit

import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import threading
sys.setrecursionlimit(100000)

#---------------------------光追部分↓-------------------------------

def normalize(x):
    return x / np.linalg.norm(x)

def intersect(origin, dir, obj):    # 射线与物体的相交测试
    if obj['type'] == 'plane':
        return intersect_plane(origin, dir, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(origin, dir, obj['position'], obj['radius'])

def intersect_plane(origin, dir, point, normal):    # 射线与平面的相交测试
    dn = np.dot(dir, normal)
    if np.abs(dn) < 1e-6:   # 射线与平面几乎平行
        return np.inf       # 交点为无穷远处
    d = np.dot(point - origin, normal) / dn         # 交点与射线原点的距离（相似三角形原理）
    return d if d>0 else np.inf     # 负数表示射线射向平面的反方向

def intersect_sphere(origin, dir, center, radius):  # 射线与球的相交测试
    OC = center - origin
    if (np.linalg.norm(OC) < radius) or (np.dot(OC, dir) < 0):
        return np.inf
    l = np.linalg.norm(np.dot(OC, dir))
    m_square = np.linalg.norm(OC) * np.linalg.norm(OC) - l * l
    q_square = radius*radius - m_square
    return (l - np.sqrt(q_square)) if q_square >= 0 else np.inf

def get_normal(obj, point):         # 获得物体表面某点处的单位法向量
    if obj['type'] == 'sphere':
        return normalize(point - obj['position'])
    if obj['type'] == 'plane':
        return obj['normal']

def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def sphere(position, radius, color, reflection=.85, diffuse=1., specular_c=.6, specular_k=50):
    return dict(type='sphere', position=np.array(position), radius=np.array(radius),
                color=np.array(color), reflection=reflection, diffuse=diffuse, specular_c=specular_c, specular_k=specular_k)

def plane(position, normal, color=np.array([1.,1.,1.]), reflection=0.15, diffuse=.75, specular_c=.3, specular_k=50):
    return dict(type='plane', position=np.array(position), normal=np.array(normal),
                color=lambda M: (np.array([1.,1.,1.]) if (int(M[0]*2)%2) == (int(M[2]*2)%2) else (np.array([0.,0.,0.]))),
                reflection=reflection, diffuse=diffuse, specular_c=specular_c, specular_k=specular_k)

scene = [sphere([.75, .1, 1.], .6, [.8, .3, 0.]),           # 球心位置，半径，颜色
         sphere([-.3, .01, .2], .3, [.0, .0, .9]),
         sphere([-2.75, .1, 3.5], .6, [.1, .572, .184]),
         plane([0., -.5, 0.], [0., 1., 0.])]                # 平面上一点的位置，法向量
#scene = [plane([0., -.5, 0.], [0., 1., 0.])]
#for i in range(5):
#    for j in range(5):
#        scene += [
#            sphere([i / 10 - 0.2, j / 10, j / 10 - 0.6], .05, [random.random(), random.random(), random.random()])]
light_point = np.array([5., 5., -10.])                      # 点光源位置
light_color = np.array([1., 1., 1.])                        # 点光源的颜色值
ambient = 0.05                                              # 环境光

def intersect_color(origin, dir, intensity):
    min_distance = np.inf
    for i, obj in enumerate(scene):
        current_distance = intersect(origin, dir, obj)
        if current_distance < min_distance:
            min_distance, obj_index = current_distance, i   # 记录最近的交点距离和对应的物体
    if (min_distance == np.inf) or (intensity < 0.01):
        return np.array([0., 0., 0.])

    obj = scene[obj_index]
    P = origin + dir * min_distance     # 交点坐标
    color = get_color(obj, P)
    N = get_normal(obj, P)                  # 交点处单位法向量
    PL = normalize(light_point - P)
    PO = normalize(origin - P)

    c = ambient * color

    l = [intersect(P + N * .0001, PL, obj_shadow_test)
            for i, obj_shadow_test in enumerate(scene) if i != obj_index]       # 阴影测试
    if not (l and min(l) < np.linalg.norm(light_point - P)):
        c += obj['diffuse'] * max(np.dot(N, PL), 0) * color * light_color
        c += obj['specular_c'] * max(np.dot(N, normalize(PL + PO)), 0) ** obj['specular_k'] * light_color

    reflect_ray = dir - 2 * np.dot(dir, N) * N  # 计算反射光线
    c += obj['reflection'] * intersect_color(P + N * .0001, reflect_ray, obj['reflection'] * intensity)

    return np.clip(c, 0, 1)

#---------------------------光追部分↑-------------------------------

#------------------------LOD部分↓--------------------

#处理取值问题
def quzhi(place,size):
    if place=="z" or place=="s":#左或上的取值
        if size%2==0:
            return int(size/2)
        else:
            return int((size+1)/2)
    elif place=='y' or place=='x':#右或下的取值
        if size%2==0:
            return int(size/2)
        else:
            return int((size-1)/2)

#LOD w长度为1
def lodrender_w1(fatherlod,iw,ih,jw,jh):
    lodcache = []
    for i in range(ih):
        if lodcache_oks[jw][jh+i]==False:
            lodcache_oks[jw][jh + i ] =True
            lodcache+=[[fatherlod+1,jw,jh+i,jw,jh+i,1,1]]
    return lodcache
#LOD h长度为1
def lodrender_h1(fatherlod,iw,ih,jw,jh):
    lodcache = []
    for i in range(iw):
        if lodcache_oks[jw+i][jh]==False:
            lodcache_oks[jw+i][jh] =True
            lodcache+=[[fatherlod+1,jw+i,jh,jw+i,jh,1,1]]
    return lodcache
#LOD 常规LOD
def lodrender(fatherlod,iw,ih,jw,jh):#父lod等级,父LOD大小*2
    #print('深度=',fatherlod+1,w,h)
    global lodcache_oks
    if iw>1 and ih>1:
        lodcache=[]
        if lodcache_oks[jw][jh]==False:
            lodcache_oks[jw][jh]=True
            lodcache+=[[fatherlod + 1, jw, jh, jw, jh, quzhi("z", iw), quzhi("s", ih)]]
        if lodcache_oks[jw + iw - 1][jh]==False:
            lodcache_oks[jw + iw - 1][jh]=True
            lodcache+=[[fatherlod + 1, jw + iw - 1, jh, jw - 1 + quzhi("z", iw) + 1, jh, quzhi("y", iw), quzhi("s", ih)]]
        if lodcache_oks[jw][jh + ih - 1]==False:
            lodcache_oks[jw][jh + ih - 1]=True
            lodcache+=[[fatherlod + 1, jw, jh + ih - 1, jw, jh - 1 + quzhi("s", ih) + 1, quzhi("z", iw), quzhi("x", ih)]]
        if lodcache_oks[jw + iw - 1][jh + ih - 1]==False:
            lodcache_oks[jw + iw - 1][jh + ih - 1]=True
            lodcache+=[[fatherlod + 1, jw + iw - 1, jh + ih - 1, jw - 1 + quzhi("z", iw) + 1, jh - 1 + quzhi("s", ih) + 1, quzhi("y", iw), quzhi("x", ih)]]


        #lodcache += [[fatherlod+1, 1+jw-1, 1+jh-1, 1+jw-1, 1+jh-1, quzhi("z", w), quzhi("s", h)]]  # 左上
        #lodcache += [[fatherlod+1, w+jw-1, 1+jh-1, quzhi("z", w) + 1+jw-1, 1+jh-1, quzhi("y", w), quzhi("s", h)]]  # 右上
        #lodcache += [[fatherlod+1, 1+jw-1, h+jh-1, 1+jw-1, quzhi("s", h) + 1+jh-1, quzhi("z", w), quzhi("x", h)]]  # 左下
        #lodcache += [[fatherlod+1, w+jw-1, h+jh-1, quzhi("z", w) + 1+jw-1, quzhi("s", h) + 1+jh-1, quzhi("y", w), quzhi("x", h)]]  # 右下

        #lodcache += lodrender(fatherlod+1, quzhi("z", w), quzhi("s", h),jw,jh)  # 左上递归
        #lodcache += lodrender(fatherlod+1, quzhi("y", w), quzhi("s", h),jw+quzhi("z",w),jh)  # 右上递归
        #lodcache += lodrender(fatherlod+1, quzhi("z", w), quzhi("x", h),jw,jh+quzhi("s",h))  # 左下递归
        #lodcache += lodrender(fatherlod+1, quzhi("y", w), quzhi("x", h),jw+quzhi("z",w),jh+quzhi("s",h))  # 右下递归

        lodcache += lodrender(fatherlod + 1, quzhi("z", iw), quzhi("s", ih), jw, jh)  # 左上递归
        lodcache += lodrender(fatherlod + 1, quzhi("y", iw), quzhi("s", ih), jw + quzhi("z", iw), jh)  # 右上递归
        lodcache += lodrender(fatherlod + 1, quzhi("z", iw), quzhi("x", ih), jw, jh + quzhi("s", ih))  # 左下递归
        lodcache += lodrender(fatherlod + 1, quzhi("y", iw), quzhi("x", ih), jw + quzhi("z", iw),
                              jh + quzhi("s", ih))  # 右下递归
        return lodcache
    elif iw==1 and ih>1:
        return lodrender_w1(fatherlod + 1, 1, ih, jw, jh)
    elif iw>1 and ih==1:
        return lodrender_h1(fatherlod + 1, iw, 1, jw, jh)
    else:
        return []

#------------------------LOD部分↑--------------------

#------------------------渲染部分↓--------------------

def rendertopygame(my_render_number):
    cishu=0
    SQ=copy.deepcopy(Q)
    for i in lodcache:
        #print(i)
        #time.sleep(1)
        cishu+=1
        x=guanzhui_ij_to_xy[i[1]-1][i[2]-1][0]
        y=guanzhui_ij_to_xy[i[1]-1][i[2]-1][1]
        SQ[:2] = (x, y)
        cache1_color = intersect_color(O, normalize(SQ - O), 1).tolist()
        cache1_color[0] = cache1_color[0] * 255
        cache1_color[1] = cache1_color[1] * 255
        cache1_color[2] = cache1_color[2] * 255
        cache_color = tuple(cache1_color)
        screen.fill((cache1_color[0],cache1_color[1],cache1_color[2]),((i[3]-1,i[4]-1),(i[5],i[6])))

        #print(i)
        #screen.set_at((int(i[3]-1),int(i[4]-1)),(255,255,255))
        pygame.display.update()

        if render_number > my_render_number:
            return
        if cishu%1000==0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
    print("渲染完毕")


#------------------------渲染部分↑--------------------

w, h = 400, 400     # 默认屏幕宽高
O = np.array([0., 0.35, -1.])   # 摄像机位置
Q = np.array([0., 0., 0.])      # 摄像机指向
img = np.zeros((h, w, 3))
r = float(w) / h
S = (-1., -1. / r + .25, 1., 1. / r + .25)

pygame.init()
screen = pygame.display.set_mode((w, h),pygame.RESIZABLE|pygame.DOUBLEBUF)
pygame.display.set_caption('实时渲染')
render_number=0
if True:#第一次处理
    print("屏幕大小为：", w, "×", h)
    lodcache_oks=[[False for i in range(h+1)] for i in range(w+1)]
    #print(lodcache_oks)
    O = np.array([0., 0.35, -1.])  # 摄像机位置
    Q = np.array([0., 0., 0.])  # 摄像机指向
    img = np.zeros((h, w, 3))
    r = float(w) / h
    S = (-1., -1. / r + .25, 1., 1. / r + .25)
    lodcache = []
    lodcache += [[0, 1, 1, 1, 1, quzhi("z", w), quzhi("s", h)]]  # 左上
    lodcache_oks[1][1]=True
    lodcache += [[0, w, 1, quzhi("z", w) + 1, 1, quzhi("y", w), quzhi("s", h)]]  # 右上
    lodcache_oks[w][1]=True
    lodcache += [[0, 1, h, 1, quzhi("s", h)+1, quzhi("z", w), quzhi("x", h)]]  # 左下
    lodcache_oks[1][h]=True
    lodcache += [[0, w, h, quzhi("z", w) + 1, quzhi("s", h) + 1, quzhi("y", w), quzhi("x", h)]]  # 右下
    lodcache_oks[w][h]=True
    lodcache += lodrender(0, quzhi("z", w), quzhi("s", h), 1, 1)  #左上递归
    lodcache += lodrender(0, quzhi("y", w), quzhi("s", h), 1 + quzhi("z", w), 1)  #右上递归
    lodcache += lodrender(0, quzhi("z", w), quzhi("x", h), 1, 1 + quzhi("s", h))  #左下递归
    lodcache += lodrender(0, quzhi("y", w), quzhi("x", h), 1 + quzhi("z", w), 1 + quzhi("s", h))  #右下递归
    lodcache.sort()
    #print(lodcache)
    print("体素块预渲染数组长度：",len(lodcache))

    guanzhui_ij_to_xy=[[[None,None] for i in range(h+1)] for i in range(w+1)]

    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            guanzhui_ij_to_xy[i][h-j]=[x,y]
            #Q[:2] = (x, y)
            #img[h - j - 1, i, :] = intersect_color(O, normalize(Q - O), 1)
    render_number+=1
    RenderThreading=threading.Thread(target=rendertopygame,args=(render_number,))
    RenderThreading.start()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
            render_number+=1
            time.sleep(0.1)
            w=event.w
            h=event.h
            print("屏幕大小改变：",w,"×",h)
            lodcache_oks = [[False for i in range(h + 1)] for i in range(w + 1)]
            # print(lodcache_oks)
            O = np.array([0., 0.35, -1.])  # 摄像机位置
            Q = np.array([0., 0., 0.])  # 摄像机指向
            img = np.zeros((h, w, 3))
            r = float(w) / h
            S = (-1., -1. / r + .25, 1., 1. / r + .25)
            lodcache = []
            lodcache += [[0, 1, 1, 1, 1, quzhi("z", w), quzhi("s", h)]]  # 左上
            lodcache_oks[1][1] = True
            lodcache += [[0, w, 1, quzhi("z", w) + 1, 1, quzhi("y", w), quzhi("s", h)]]  # 右上
            lodcache_oks[w][1] = True
            lodcache += [[0, 1, h, 1, quzhi("s", h) + 1, quzhi("z", w), quzhi("x", h)]]  # 左下
            lodcache_oks[1][h] = True
            lodcache += [[0, w, h, quzhi("z", w) + 1, quzhi("s", h) + 1, quzhi("y", w), quzhi("x", h)]]  # 右下
            lodcache_oks[w][h] = True
            lodcache += lodrender(0, quzhi("z", w), quzhi("s", h), 1, 1)  # 左上递归
            lodcache += lodrender(0, quzhi("y", w), quzhi("s", h), 1 + quzhi("z", w), 1)  # 右上递归
            lodcache += lodrender(0, quzhi("z", w), quzhi("x", h), 1, 1 + quzhi("s", h))  # 左下递归
            lodcache += lodrender(0, quzhi("y", w), quzhi("x", h), 1 + quzhi("z", w), 1 + quzhi("s", h))  # 右下递归
            lodcache.sort()
            # print(lodcache)
            print("体素块预渲染数组长度：", len(lodcache))

            guanzhui_ij_to_xy = [[[None, None] for i in range(h + 1)] for i in range(w + 1)]

            for i, x in enumerate(np.linspace(S[0], S[2], w)):
                for j, y in enumerate(np.linspace(S[1], S[3], h)):
                    guanzhui_ij_to_xy[i][h - j] = [x, y]
                    # Q[:2] = (x, y)
                    # img[h - j - 1, i, :] = intersect_color(O, normalize(Q - O), 1)
            RenderThreading=threading.Thread(target=rendertopygame,args=(render_number,))
            RenderThreading.start()
            #print(lodcache)
            #lodrenders=lodrender("zs",w,h,lodcache)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                O[2] += 1#np.array([0., 0.35, -1.])
                render_number+=1
                time.sleep(0.1)
                RenderThreading = threading.Thread(target=rendertopygame, args=(render_number,))
                RenderThreading.start()
            elif event.key == pygame.K_s:
                O[2] -= 1#np.array([0., 0.35, -1.])
                render_number+=1
                time.sleep(0.1)
                RenderThreading = threading.Thread(target=rendertopygame, args=(render_number,))
                RenderThreading.start()
            elif event.key == pygame.K_a:
                O[0] -= 1#np.array([0., 0.35, -1.])
                render_number+=1
                time.sleep(0.1)
                RenderThreading = threading.Thread(target=rendertopygame, args=(render_number,))
                RenderThreading.start()
            elif event.key == pygame.K_d:
                O[0] += 1#np.array([0., 0.35, -1.])
                render_number+=1
                time.sleep(0.1)
                RenderThreading = threading.Thread(target=rendertopygame, args=(render_number,))
                RenderThreading.start()
    #screen.fill((255,0,0),((lodcache[0][3]-1,lodcache[0][4]-1),(lodcache[0][5],lodcache[0][6])))
    #screen.fill((0,255,0),((lodcache[1][3]-1,lodcache[1][4]-1),(lodcache[1][5],lodcache[1][6])))
    #screen.fill((0,0,255),((lodcache[2][3]-1,lodcache[2][4]-1),(lodcache[2][5],lodcache[2][6])))
    #screen.fill((128,128,128),((lodcache[3][3]-1,lodcache[3][4]-1),(lodcache[3][5],lodcache[3][6])))
    time.sleep(0.1)
    #pygame.display.update()
#for i, x in enumerate(np.linspace(S[0], S[2], w)):
#    print("%.2f" % (i / float(w) * 100), "%")
#    for j, y in enumerate(np.linspace(S[1], S[3], h)):
#        Q[:2] = (x, y)
#        img[h - j - 1, i, :] = intersect_color(O, normalize(Q - O), 1)
plt.imsave('test.png', img)
'''
lodrenders的结构：
{[lod级别,像素实际w坐标,像素实际h坐标,绘制w坐标,绘制h坐标,绘制大小w,绘制大小h]}
#对于像素实际位置由于pygame和RTX渲染器的特点（200,200）等于（0,0）~（199~199），算法提供的是（1,1）~（200，200）
于是渲染时坐标需要减去1
'''