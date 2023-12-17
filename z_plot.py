import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# サンプルデータをNumPy配列に変換
data = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
])


# 新しいデータを追加
new_data = np.array([4, 5, 6, 7, 8, 9])

# dataに新しいデータを追加
data = np.concatenate([data, [new_data]])

# 各ベクトルを抽出
x, y, z, u, v, w = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

# 3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ベクトルをプロット
ax.quiver(x, y, z, u, v, w, length=1, normalize=True)

# ラベルの設定
ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')

# xの値域を設定
ax.set_xlim(0, 424)

# グラフを表示
#plt.show()

# サンプルの配列
my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

u = 9
v = 7
print(np.sqrt(u**2 + v**2))


