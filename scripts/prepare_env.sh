#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 定义变量
third_pkgs_dir="${PROJECT_ROOT}/third_pkgs"
url="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip"
zip_file_name="${third_pkgs_dir}/libtorch.zip"

# 创建 third_pkgs 目录
if [ ! -d "$third_pkgs_dir" ]; then
    mkdir -p "$third_pkgs_dir"
    echo "已创建目录: $third_pkgs_dir" [[1]]
else
    echo "目录已存在: $third_pkgs_dir"
fi

# 下载 ZIP 文件
echo "开始下载文件: $url"
wget -O "$zip_file_name" "$url"
if [ $? -eq 0 ]; then
    echo "文件已下载并保存为: $zip_file_name"
else
    echo "下载失败，请检查网络或URL是否正确。"
    exit 1
fi

# 解压 ZIP 文件
echo "正在解压文件..."
unzip "$zip_file_name" -d "$third_pkgs_dir"
if [ $? -eq 0 ]; then
    echo "文件已成功解压到: $third_pkgs_dir"
else
    echo "解压失败，请检查ZIP文件是否完整。"
    exit 1
fi

# 清理 ZIP 文件（可选）
echo "正在清理临时文件..."
rm -f "$zip_file_name"
if [ $? -eq 0 ]; then
    echo "已删除临时文件: $zip_file_name"
else
    echo "删除临时文件失败，请手动清理。"
fi

echo "操作完成！"