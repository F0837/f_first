import sys
import random
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import math
import matplotlib.pyplot as plt

def bb84_protocol(length):
    # Alice 随机生成比特串和基
    alice_bits = ''.join(random.choice(['0', '1']) for _ in range(length))
    alice_bases = ''.join(random.choice(['0', '1']) for _ in range(length))

    # Bob 也随机生成测量基
    bob_bases = ''.join(random.choice(['0', '1']) for _ in range(length))

    # 假设Bob测量的结果
    bob_results = ''.join(random.choice(['0', '1']) if bob_bases[i] == alice_bases[i] else '?' for i in range(length))

    # Alice 和 Bob 交换基，对比结果，获得密钥
    key = ''.join(bob_results[i] for i in range(length) if bob_results[i] != '?')

    return key


def generate_key(key_length):
    bb84_key = ''
    while len(bb84_key) < key_length:
        bb84_key += bb84_protocol(10)  # 每次生成最多10位密钥
    return bb84_key[:key_length]

def select_image():
    # 创建一个虚拟的 Tkinter 根窗口以便使用 filedialog
    root = tk.Tk()
    root.withdraw()  # 隐藏Tkinter根窗口

    # 使用 filedialog 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    # 确保用户选择了文件后再进行后续操作
    if file_path:
        imageOrig = cv2.imread(file_path)
        return imageOrig


# AES 加密类
class AESCipher:
    def __init__(self, mode=AES.MODE_CBC):
        self.mode = mode
        self.keySize = 16
        self.ivSize = AES.block_size if mode == AES.MODE_CBC else 0
        self.iv = get_random_bytes(self.ivSize)
        self.quantum_key = generate_key(self.keySize)
        self.key = self.quantum_key.encode('utf-8')[:self.keySize]

    def encrypt(self, imageOrig):
        # 输出图像大小
        rowOrig, columnOrig, depthOrig = imageOrig.shape
        print(imageOrig.shape)
        # 检查图像的宽度是否低于图像加密的宽度限制
        print("AES.block_size:" + str(AES.block_size))
        minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
        print("minWidth:" + str(minWidth))
        if columnOrig < minWidth:
            print(
                'The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(
                    minWidth))
            sys.exit()
        # 将图像转化成字节
        imageOrigBytes = imageOrig.tobytes()
        print("imageOrigBytes:" + str(len(imageOrigBytes)))
        # 初始化AES加密器
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv) if self.mode == AES.MODE_CBC else AES.new(self.key, AES.MODE_ECB)
        # 将字节数据进行填充，得到填充后的数据
        imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
        # 得到密文
        ciphertext = cipher.encrypt(imageOrigBytesPadded)
        # 填充的位数
        paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
        print('paddedSize:' + str(paddedSize))
        void = columnOrig * depthOrig - self.ivSize - paddedSize
        ivCiphertextVoid = self.iv + ciphertext + bytes(void)
        # 因为进行了数据填充，所有加密后的图像会比原图像多1行
        imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)
        return imageEncrypted

    def decrypt(self, imageEncrypted):
        rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape
        rowOrig = rowEncrypted - 1
        # np矩阵转字节
        encryptedBytes = imageEncrypted.tobytes()
        # 取前IvSize位为IV
        iv = encryptedBytes[:self.ivSize]
        imageOrigBytesSize = rowOrig * columnOrig * depthOrig
        # 确定填充的字节数
        paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
        # 确定图像的密文
        encrypted = encryptedBytes[self.ivSize: self.ivSize + imageOrigBytesSize + paddedSize]
        # 解密
        cipher = AES.new(self.key, AES.MODE_CBC, iv) if self.mode == AES.MODE_CBC else AES.new(self.key, AES.MODE_ECB)
        decryptedImageBytesPadded = cipher.decrypt(encrypted)
        # 去除填充的数据
        decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)
        # 把字节转化成图像
        decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)
        return decryptedImage

# 显示图像
def show_image(image, title):
    # 将图像转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用 matplotlib 显示图像
    plt.imshow(image)
    plt.axis('off')  # 关闭坐标轴
    plt.title(title)
    plt.show()

'''
绘制灰度直方图
'''
def hist(img):
    B, G, R = cv2.split(img)

    # 转成一维
    R = R.flatten(order='C')
    G = G.flatten(order='C')
    B = B.flatten(order='C')

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示原图像的直方图
    plt.hist(img.flatten(order='C'), bins=range(257), color='gray')
    plt.title('原图像')
    plt.show()

    # 显示通道R的直方图
    plt.hist(R, bins=range(257), color='red')
    plt.title('通道R')
    plt.show()

    # 显示通道G的直方图
    plt.hist(G, bins=range(257), color='green')
    plt.title('通道G')
    plt.show()

    # 显示通道B的直方图
    plt.hist(B, bins=range(257), color='blue')
    plt.title('通道B')
    plt.show()

'''
分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
'''
def RGB_correlation(channel,N):
  #计算channel通道
  h,w=channel.shape
  #随机产生pixels个[0,w-1)范围内的整数序列
  row=np.random.randint(0,h-1,N)
  col=np.random.randint(0,w-1,N)
  #绘制相邻像素相关性图,统计x,y坐标
  x=[]
  h_y=[]
  v_y=[]
  d_y=[]
  for i in range(N):
    #选择当前一个像素
    x.append(channel[row[i]][col[i]])
    #水平相邻像素是它的右侧也就是同行下一列的像素
    h_y.append(channel[row[i]][col[i]+1])
    #垂直相邻像素是它的下方也就是同列下一行的像素
    v_y.append(channel[row[i]+1][col[i]])
    #对角线相邻像素是它的右下即下一行下一列的那个像素
    d_y.append(channel[row[i]+1][col[i]+1])
  #三个方向的合到一起
  x=x*3
  y=h_y+v_y+d_y


  #计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
  ex=0
  for i in range(N):
    ex+=channel[row[i]][col[i]]
  ex=ex/N
  #计算D(x)
  dx=0
  for i in range(N):
    dx+=(channel[row[i]][col[i]]-ex)**2
  dx/=N

  #水平相邻像素h_y
  #计算E(y)
  h_ey=0
  for i in range(N):
    h_ey+=channel[row[i]][col[i]+1]
  h_ey/=N
  #计算D(y)
  h_dy=0
  for i in range(N):
    h_dy+=(channel[row[i]][col[i]+1]-h_ey)**2
  h_dy/=N
  #计算协方差
  h_cov=0
  for i in range(N):
    h_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]][col[i]+1]-h_ey)
  h_cov/=N
  h_Rxy=h_cov/(np.sqrt(dx)*np.sqrt(h_dy))

  #垂直相邻像素v_y
  #计算E(y)
  v_ey=0
  for i in range(N):
    v_ey+=channel[row[i]+1][col[i]]
  v_ey/=N
  #计算D(y)
  v_dy=0
  for i in range(N):
    v_dy+=(channel[row[i]+1][col[i]]-v_ey)**2
  v_dy/=N
  #计算协方差
  v_cov=0
  for i in range(N):
    v_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]]-v_ey)
  v_cov/=N
  v_Rxy=v_cov/(np.sqrt(dx)*np.sqrt(v_dy))

  #对角线相邻像素d_y
  #计算E(y)
  d_ey=0
  for i in range(N):
    d_ey+=channel[row[i]+1][col[i]+1]
  d_ey/=N
  #计算D(y)
  d_dy=0
  for i in range(N):
    d_dy+=(channel[row[i]+1][col[i]+1]-d_ey)**2
  d_dy/=N
  #计算协方差
  d_cov=0
  for i in range(N):
    d_cov+=(channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]+1]-d_ey)
  d_cov/=N
  d_Rxy=d_cov/(np.sqrt(dx)*np.sqrt(d_dy))

  return h_Rxy,v_Rxy,d_Rxy,x,y

'''
分别计算图像img的各通道相邻像素的相关系数，默认随机选取3000对相邻像素
'''
def correlation(img, N=3000):
    h, w, _ = img.shape
    B, G, R = cv2.split(img)
    R_Rxy = RGB_correlation(R, N)
    G_Rxy = RGB_correlation(G, N)
    B_Rxy = RGB_correlation(B, N)

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示原图像
    plt.imshow(img[:, :, (2, 1, 0)])
    plt.title('原图像')
    plt.show()

    # 显示通道R的相关性散点图
    plt.scatter(R_Rxy[3], R_Rxy[4], s=1, c='red')
    plt.title('通道R的相关性')
    plt.show()

    # 显示通道G的相关性散点图
    plt.scatter(G_Rxy[3], G_Rxy[4], s=1, c='green')
    plt.title('通道G的相关性')
    plt.show()

    # 显示通道B的相关性散点图
    plt.scatter(B_Rxy[3], B_Rxy[4], s=1, c='blue')
    plt.title('通道B的相关性')
    plt.show()

    return R_Rxy[0:3], G_Rxy[0:3], B_Rxy[0:3]

'''
计算图像的信息熵
'''


def entropy(img):
    w, h, _ = img.shape
    B, G, R = cv2.split(img)

    # 获取每个通道的唯一值及其计数
    _, num1 = np.unique(R, return_counts=True)
    _, num2 = np.unique(G, return_counts=True)
    _, num3 = np.unique(B, return_counts=True)

    R_entropy = 0
    G_entropy = 0
    B_entropy = 0

    # 计算每个通道的熵
    for i in range(len(num1)):  # 应该基于num1的长度循环
        p1 = num1[i] / (w * h)
        R_entropy -= p1 * (math.log(p1, 2)) if p1 > 0 else 0  # 避免除以零

    for i in range(len(num2)):  # 应该基于num2的长度循环
        p2 = num2[i] / (w * h)
        G_entropy -= p2 * (math.log(p2, 2)) if p2 > 0 else 0  # 避免除以零

    for i in range(len(num3)):  # 应该基于num3的长度循环
        p3 = num3[i] / (w * h)
        B_entropy -= p3 * (math.log(p3, 2)) if p3 > 0 else 0  # 避免除以零

    return R_entropy, G_entropy, B_entropy

# 主函数
def main():
    # 选择图像
    imageOrig = select_image()
    if imageOrig is None:
        return

    # 创建 AESCipher 实例
    aes_cipher = AESCipher()

    # 加密图像
    imageEncrypted = aes_cipher.encrypt(imageOrig)

    # 显示原图
    show_image(imageOrig, "Original Image")

    # 显示加密图
    show_image(imageEncrypted, "Encrypted Image")

    # 解密图像
    decryptedImage = aes_cipher.decrypt(imageEncrypted)
    show_image(decryptedImage, "Decrypted Image")

    # 绘制直方图
    hist(imageOrig)
    hist(imageEncrypted)

    # 计算相关系数
    R_Rxy, G_Rxy, B_Rxy = correlation(imageOrig)
    # 输出结果保留四位有效数字
    print("******原图的各通道各方向的相关系数为*****")
    print('通道\tHorizontal\tVertical\tDiagonal')
    print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0], R_Rxy[1], R_Rxy[2]))
    print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0], G_Rxy[1], G_Rxy[2]))
    print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0], B_Rxy[1], B_Rxy[2]))

    R_Rxy, G_Rxy, B_Rxy = correlation(imageEncrypted)
    # 输出结果保留四位有效数字
    print("******加密图的各通道各方向的相关系数为*****")
    print('通道\tHorizontal\tVertical\tDiagonal')
    print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0], R_Rxy[1], R_Rxy[2]))
    print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0], G_Rxy[1], G_Rxy[2]))
    print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0], B_Rxy[1], B_Rxy[2]))

    # 计算信息熵
    R_entropy, G_entropy, B_entropy = entropy(imageOrig)
    print('***********原图信息熵*********')
    print('通道R:{:.4}'.format(R_entropy))
    print('通道G:{:.4}'.format(G_entropy))
    print('通道B:{:.4}'.format(B_entropy))

    # 加密图像lena的熵
    R_entropy, G_entropy, B_entropy = entropy(imageEncrypted)
    print('***********加密图信息熵*********')
    print('通道R:{:.4}'.format(R_entropy))
    print('通道G:{:.4}'.format(G_entropy))
    print('通道B:{:.4}'.format(B_entropy))

if __name__ == "__main__":
    main()