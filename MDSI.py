import cv2
import numpy as np
import cmath  # 负数开方必备[https://blog.csdn.net/u014647208/article/details/53678198]
import sys


sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


def MDSI(RefImg, DistImg, combMethod='sum'):
	# cv2.imshow('Ref Image', RefImg)
	# cv2.imshow('Dist Image', DistImg)

	B_ref, G_ref, R_ref = cv2.split(RefImg)  # 分离RGB颜色通道
	B_dist, G_dist, R_dist = cv2.split(DistImg)

	C1 = 140
	C2 = 55
	C3 = 550
	dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3
	dy = np.transpose(dx)
	rows, cols = np.shape(R_ref)
	minDimension = min(rows, cols)
	f = max(1, round(minDimension / 256))

	# 跳行（列）提取矩阵的元素，参考numpy advanced indexing [https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html]
	row_index, col_index = np.meshgrid([i for i in range(0, rows, f)], [i for i in range(0, cols, f)])
	row_index = np.transpose(row_index)
	col_index = np.transpose(col_index)

	aveR1 = cv2.blur(R_ref, (f, f))  # 当卷积核对称时，卷积等于滤波
	aveR2 = cv2.blur(R_dist, (f, f))
	R1 = aveR1[row_index, col_index]
	R2 = aveR2[row_index, col_index]

	aveG1 = cv2.blur(G_ref, (f, f))
	aveG2 = cv2.blur(G_dist, (f, f))
	G1 = aveG1[row_index, col_index]
	G2 = aveG2[row_index, col_index]

	aveB1 = cv2.blur(B_ref, (f, f))
	aveB2 = cv2.blur(B_dist, (f, f))
	B1 = aveB1[row_index, col_index]
	B2 = aveB2[row_index, col_index]

	# Luminance
	L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
	L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2
	F = 0.5 * (L1 + L2)  # Fusion

	# Opponent color space
	H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
	H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
	M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
	M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

	# Gradient magnitudes

	IxL1 = cv2.filter2D(L1, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)  # 当卷积核不对称时，需要将卷积核旋转90度
	IyL1 = cv2.filter2D(L1, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
	gR = np.sqrt(IxL1 ** 2 + IyL1 ** 2)

	IxL2 = cv2.filter2D(L2, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)
	IyL2 = cv2.filter2D(L2, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
	gD = np.sqrt(IxL2 ** 2 + IyL2 ** 2)

	IxF = cv2.filter2D(F, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)
	IyF = cv2.filter2D(F, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
	gF = np.sqrt(IxF ** 2 + IyF ** 2)

	# Gradient Similarity(GS)
	GS12 = (2 * gR * gD + C1) / (gR ** 2 + gD ** 2 + C1)  # GS of R and D
	GS13 = (2 * gR * gF + C2) / (gR ** 2 + gF ** 2 + C2)
	GS23 = (2 * gD * gF + C2) / (gD ** 2 + gF ** 2 + C2)
	GS_HSV = GS12 + GS23 - GS13

	# Chromaticity Similarity(CS)
	CS = (2 * (H1 * H2 + M1 * M2) + C3) / (H1 ** 2 + H2 ** 2 + M1 ** 2 + M2 ** 2 + C3)
	# cv2.imshow('CS', CS)

	GCS = CS
	if combMethod == 'sum':
		alpha = 0.6
		GCS = alpha * GS_HSV + (1 - alpha) * CS
	elif combMethod is 'mult':
		gamma = 0.2
		beta = 0.1
		GCS = (GS_HSV ** gamma) * (CS ** beta)
	# cv2.imshow("GCS", GCS)
	# cv2.waitKey()
	cv2.destroyAllWindows()
	flatten = np.reshape(GCS, (-1, 1))
	temp = np.zeros(flatten.shape, dtype=np.complex64)
	for i, ele in enumerate(flatten):
		temp[i] = cmath.sqrt(ele)  # 程序里有两次开平方，第一次开平方在这里，用于构造numpy complex型的矩阵，cmath.sqrt()允许对负数开方，但是貌似一次只能操作一个数
	return mad(temp ** 0.5) ** 0.25  # 第二次开平方及计算完mad之后的0.25次方


def mad(vec):
	"""
	Mean Absolute Deviation
	:param array: numpy array
	:return: float
	"""
	return np.mean(np.abs(vec - np.mean(vec)))


if __name__ == '__main__':
	ref = cv2.imread('ref.png')
	dist = cv2.imread('dist.png')
	print('Q: ',MDSI(ref, dist))
