import numpy as np
print(np.__version__)

np.show_config()

"""長度為10的空向量"""
print(np.zeros(10))

"""找到數組內存大小"""
data=np.zeros((10,10))
print("%d bytes"%(data.size*data.itemsize))

"""得到numpy中add的說明文黨"""
np.info(np.add)

"""創建長度10並除了第五個值為1的空向量"""
data=np.zeros(10)
data[4]=1
print(data)

"""倒數"""
data8=np.arange(50)
data8=data8[::-1]
print(data8)
#調換數字
data8[0],data8[49]=data8[49],data8[0]
print(data8)

"""3*3且0~8矩陣"""
data9 = np. arange(9).reshape(3,3)
print(data9)

"""找到非0位置索引"""
data10=np.nonzero([1,2,0,0,4,0])
print(data10)

"""3*3單位矩陣"""
data11=np.eye(3)
print(data11)

"""3*3*3的隨機數組"""
data12=np.random.random((3,3,3))
print(data12)

"""10*10隨機數組並找出max,min"""
data13=np.random.random((3,3))
#10*10太大
print(data13)
data13min,data13max=data13.min(),data13.max()
print(data13min,data13max)

"""長度30的隨機並計算平均"""
data14=np.random.random(30)
print(data14)
data14=data14.mean()
print(data14)

"""創建二為數組邊界值1,其餘為0"""
data15=np.ones((4,4))
data15[1:-1,1:-1]=0
print(data15)

"""以0填充邊界"""
data16=np.random.random((4,4))*9
data16=np.pad(data16,pad_width=1,mode='constant',constant_values=0)
#這裡mode除了constant的功能還有1.edge 2.linear_ramp 3.maximum 4.minimum 5.mean 6.reflect 7.symmetric 8.wrap等不同方式
data16=np.round(data16)
#四捨五入數值
print(data16)

"""創建5*5矩陣，並設置1,2,3,4在四角落"""
data18 = np.zeros((5, 5))
data18[0,0],data18[0,4],data18[4,0],data18[4,4]=1,2,3,4
print(data18)
#不明白

"""*8*8棋盤"""
data19=np.zeros((8,8),dtype=int)
data19[1::2,::2]=1
data19[::2,1::2]=1
print(data19)

"""考慮(6,7,8)數組，第100的索引?"""
np.unravel_index(100,(6,7,8))

"""使用tile創建一個8*8的棋盤"""
data21=np.tile(np.array([[5,8],[8,5]]),(4,4))
#感覺是創建一個[[x,y],[y,x]]的矩陣後把這矩陣堆疊4*4次
print(data21)

"""對一個5*5矩陣做歸一畫(data-datamin)/(datamax-datamin)"""
data22=np.random.random((5,5))
data22max,data22min=data22.max(),data22.min()
data22=(data22-data22min)/(data22max-data22min)
print(data22)

"""創建一個將顏色描述為(RGBA)四個無符號字節的自定義dtype"""
color=np.dtype([("r",np.ubyte,1),
                ("g",np.ubyte,1),
                ("b",np.ubyte,1),
                ("#a",np.ubyte,1)])
print(color)

"""創建5*3矩陣與3*2的矩陣並相乘"""
data241=np.random.random((5,3))*9
data241=np.round(data241)
#print(data241)
data242=np.random.random((3,2))*9
data242=np.round(data242)
#print(data242)
data24=np.dot(data241,data242)
print(data24)

"""定一個一為數組，對其3~8間所有元素取反"""
data25=np.arange(10)
data25[(3<data25)&(data25<8)]*=-1
print(data25)

"""運行結果"""

print(sum(range(5), -1))  # 输出: 9

from numpy import sum as np_sum

print(np_sum(range(5), -1))  # 输出: 10

"""對浮點位做捨入"""
data29=np.random.random(5)*9
data29=np.round(data29)
print(data29)

"""更換頻道"""
array=np.array([[1,2,3],
                [4,5,6]])
print(array)
print('number of dim:',array.ndim)
print('shape:',array.shape)
print('size:',array.size)

datan=np.array([2.27,50,46],dtype=np.int32)
print(datan)

datan=np.array([2.27,50,46],dtype=np.float32)
print(datan)
#位數越小占用的內存越大，精確使用64，預留空間用16

datan=np.arange(20).reshape((4,5))
print(datan)

datan=np.linspace(1,10,9).reshape((3,3)) 
#從1~10中間有9個數字
print(datan)

datan=np.array([10,20,30,40])
datam=np.arange(4)
print(datan,datam)
datat=5*np.sin(datam)-datan
print(datat)

datan=np.arange(2,14).reshape((3,4))
print(datan)
#最小值索引
print(np.argmin(datan))
#最大值索引
print(np.argmax(datan))
#平均值
print(np.mean(datan))
#累加
print(np.cumsum(datan))
#輸出行列位置
print(np.nonzero(datan))

datan=np.arange(14,2,-1).reshape((3,4))
#對它排序
print(datan)
print((np.sort))

"""矩陣反向(行列對調)"""
datan=np.arange(14,2,-1).reshape((3,4))
print(datan)
print(np.transpose(datan))
#反向
print((datan.T).dot(datan))

datan=np.random.random((4,5))*9
datan=np.round(datan)
print(np.clip(datan,4,8))
#所有小於4都變4，大於8的都變8
print(np.mean(datan,axis=0))
#在mean(計算平均值)中，axis=0代表計算列，axis=1代表計算行

datan=np.arange(3,15).reshape((3,4))
print(datan)
print(datan[2][1])#=print(datan[2,1])
print(datan[2,:])#第二行的所有數
print(datan[1,1:3])#第一行的第一列到第三列(不包含第三)
print(datan[0:2,1:3])
print('---------------')
"""for row in datan:
    print(row)
    for column in datan.T:
        print(column)"""#列印出每一行;列印出被反轉的datan
print(datan.flatten())#列出所有
for item in datan.flat:
    print(item)#一個個列印出空間內的物件
print('---------------')

#8合併
A=np.array([1,1,1])
B=np.array([2,2,2])

C=np.vstack((A,B))#上下合併[[1,1,1][2,2,2]]
D=np.hstack((A,B))#左右合併[1,1,1,2,2,2]
print(C)
print(D)
print(A.shape)
print(A[:,np.newaxis].shape)
print(A[np.newaxis,:].shape)
#np.newaxis會加一維度，:則代表原陣列。類似原陣列平面長度為3，為它添加了寬度使它變為立體的
""" A[:,np.newaxis].shape 會變成(3,1)的陣列
    A[np.newaxis,:].shape 會變成(1,3)的陣列"""
print(A[:,np.newaxis])
print(A[np.newaxis,:])

print('------------')
A=np.array([1,1,1])[:,np.newaxis]
B=np.array([2,2,2])[:,np.newaxis]
print(np.hstack((A,A,B)))#但這[不能]指定[縱向合併]
C=np.concatenate((A,B,B,A),axis=0)#axis=0縱向合併
print(C)
D=np.concatenate((A,B,B,A),axis=1)#axis=1橫向合併
print(D)
print('------------')

#9
datan=np.arange(12).reshape((3,4))
print(datan)
print(np.split(datan,2,axis=1))#axis代表對列操作
print(np.split(datan,3,axis=0))#axis代表對行操作
"""以上只能實現等量分割"""
print('------------')
print(np.array_split(datan,3,axis=1))
"""這可實現不等量分割"""
print(np.vsplit(datan,3))
print(np.hsplit(datan,2))
print('------------')
datam=(np.vsplit(datan,3))
print(datam)
"""選取切割後的資料"""
datax=datam[0]
print(datax)
datay=datam[1]
print(datay)



