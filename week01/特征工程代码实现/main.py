import subprocess
import sys

if __name__ == "__main__":
    python = sys.executable  # 使用当前解释器运行子文件，避免环境不一致
    files = [
        "00_generate_data.py",   # 1. 先生成房价数据 → housing_data.xlsx
        "01_minmax_scaler.py",   # 2. MinMax归一化
        "02_standard_scaler.py", # 3. Z-Score标准化
        "03_log_transform.py",   # 4. 对数变换
        "04_label_encoder.py",   # 5. 标签编码
        "05_onehot_encoder.py",  # 6. 独热编码
        "06_pca.py",             # 7. PCA降维
        "07_lda.py",             # 8. LDA降维
        "08_tsne.py",            # 9. t-SNE降维
    ]

    for f in files:
        print(f"\n========== 正在运行：{f} ==========")
        # subprocess 传参数列表，避免 cmd 引号解析错误
        result = subprocess.run([python, f])
        if result.returncode != 0:
            print(f"!! {f} 运行失败，退出码 {result.returncode}")

    print("\n全部特征工程代码运行完毕！")
