import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_points(initial_points, final_points):
    # ベクトルを計算
    vectors = final_points - initial_points

    # 3Dグラフの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初期座標をプロット
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], c='r', marker='o', label='Initial Points')

    # 終点座標をプロット
    ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], c='g', marker='o', label='Final Points')

    # ベクトルを表示
    for i in range(len(vectors)):
        ax.quiver(initial_points[i, 0], initial_points[i, 1], initial_points[i, 2], vectors[i, 0], vectors[i, 1], vectors[i, 2],
                  color='b', label=f'Vector {i + 1}')

    # グラフの設定
    ax.set_xlabel('X軸')
    ax.set_ylabel('Y軸')
    ax.set_zlabel('Z軸')
    ax.set_title('3つの点のベクトル変化')
    ax.legend()

    # グラフを表示
    plt.show()

# tからt+1のデータ
initial_points_t1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
final_points_t1 = np.array([[4, 5, 6], [6, 7, 8], [7, 8, 9]])

# t+1からt+2のデータ
initial_points_t2 = np.array([[4, 5, 6], [6, 7, 8], [7, 8, 9]])
final_points_t2 = np.array([[7, 7, 7], [8, 8, 8], [9, 9, 9]])

# tからt+1のグラフを表示
plot_points(initial_points_t1, final_points_t1)
plot_points(initial_points_t2, final_points_t2)
plt.show()

# グラフをリセット
plt.close()

# t+1からt+2のグラフを表示
plot_points(initial_points_t2, final_points_t2)
