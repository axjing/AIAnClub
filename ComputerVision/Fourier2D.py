# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np


# def add_ax(ax,mat):
#     for i in range(len(mat)):
#         for j in range(len(mat[i])):
#             for k in range(len(mat[i][j])):
#                 if mat[i][j][k]==1:
#                     ax.scatter(i,j,k,c='r',marker='o')
                    
#     return ax


# a=np.zeros((100,100,100))

# a[:,:,50]=1

# #for each voxel:real+i_img+j_img

# fig=plt.figure(0)
# ax=fig.add_subplot(111,projection='3d')

# ax=add_ax(ax,a)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
 
# plt.show()


import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

test_plane=np.ones((101,101))
plane=np.ones((101,101))
for i in range(101):
    for j in range(101):
        #plane[i][j]=j
        if ((i-50)**2+(j-50)**2)**0.5<10:
            plane[i][j]=1.5
plane=plane#/101
center_mat=np.zeros((101,101,2))

r_mat=np.zeros((101,101))

vector_mat=np.zeros((101,101,3))

angle_mat=np.zeros((101,101))

down_center=[len(r_mat)//2,len(r_mat)//2]

res=np.zeros((101,101,101))

def add_ax(ax,mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            for k in range(len(mat[i][j])):
                if mat[i][j][k]==1:
                    ax.scatter(i,j,k,c='r',marker='o')
    return ax

def split(n):
    if n>100:
        return 100
    elif n<0:
        return 0
    else:
        return n
#生成直角坐标系矩阵
for i in range(len(center_mat)):
    for j in range(len(center_mat[i])):
        center_mat[i][j][0]=-(i-50)
        center_mat[i][j][1]=j-50
#生成半径矩阵
for i in range(len(r_mat)):
    for j in range(len(r_mat[i])):
        if center_mat[i][j][0]==0 or center_mat[i][j][1]==0:
            r=1
        else:
            big_v=abs(center_mat[i][j][0]) if abs(center_mat[i][j][0])>abs(center_mat[i][j][1]) else abs(center_mat[i][j][1])
            small_v=abs(center_mat[i][j][1]) if abs(center_mat[i][j][0])>abs(center_mat[i][j][1]) else abs(center_mat[i][j][0])
            r=np.cos(abs(np.arctan(small_v/big_v)))
        r_mat[i][j]=r
        
#生成旋转向量矩阵
for i in range(len(vector_mat)):
    for j in range(len(vector_mat[i])):
        #v=[]
        if center_mat[i][j][0]==0:
            v=[0,1,0]
        if center_mat[i][j][1]==0:
            v=[1,0,0]
        
        #vk=np.tan((np.arctan(center_mat[i][j][0]/center_mat[i][j][1])/np.pi+0.5)*np.pi)
        vk=[center_mat[i][j][0],center_mat[i][j][1]]/(center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5
        #print(vk,i,j)
        if center_mat[i][j][0]>0 and center_mat[i][j][1]>0:
            v=[abs(vk[1]),-abs(vk[0]),0]
        if center_mat[i][j][0]>0 and center_mat[i][j][1]<0: 
            v=[-abs(vk[1]),-abs(vk[0]),0]
        if center_mat[i][j][0]<0 and center_mat[i][j][1]<0:
            v=[-abs(vk[1]),abs(vk[0]),0]
        if center_mat[i][j][0]<0 and center_mat[i][j][1]>0: 
            v=[abs(vk[1]),abs(vk[0]),0]
        vector_mat[i][j][0]=v[0]
        vector_mat[i][j][1]=v[1]
        vector_mat[i][j][2]=v[2]
#生成角矩阵
for i in range(len(center_mat)):
    for j in range(len(center_mat[i])):
        l=((center_mat[i][j][0]**2+center_mat[i][j][1]**2)**0.5*r_mat[i][j])/50.5
        angle_mat[i][j]=l

#缠绕
for i in range(len(test_plane)):
    for j in range(len(test_plane[i])):
        #p_high=test_plane[i][j]
        p_high=plane[i][j]
        #p=[center_mat[i][j][0],center_mat[i][j][1],0]
        #p=[0,0,-50]
        p=np.quaternion(0,0,0,-50)
        angle=angle_mat[i][j]*np.pi
        v=vector_mat[i][j]
        a=np.cos(angle/2)*np.quaternion(1,0,0,0)
        b=np.sin(angle/2)*v[0]*np.quaternion(0,1,0,0)
        c=np.sin(angle/2)*v[1]*np.quaternion(0,0,1,0)
        d=np.sin(angle/2)*v[2]*np.quaternion(0,0,0,1)
        h=a+b+c+d
        h_=a-b-c-d
        r=r_mat[i][j]
        p_new=h*p*h_
        #print(p_new,r,i,j)
        p_new=p_new#*r
        #print(p_new)
        x_=int(p_new.x*p_high)+50
        y_=int(p_new.y*p_high)+50
        z_=int(p_new.z*p_high)+50
        x_=split(x_)
        y_=split(y_)
        z_=split(z_)
        res[x_][y_][z_]=1

#画球
fig=plt.figure(0)
ax=fig.add_subplot(111,projection='3d')

ax=add_ax(ax,res)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
 
plt.show()