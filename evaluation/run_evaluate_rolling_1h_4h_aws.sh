#!/bin/bash

# ============================================================
# AWS S3 Rolling Evaluation Script
# 从 AWS S3 加载模型并进行滚动窗口评估
# ============================================================

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# AWS 配置 (可以通过环境变量覆盖)
# ============================================================

# S3 Bucket 名称
export AWS_BUCKET_NAME="${AWS_BUCKET_NAME:-your-bucket-name}"

# AWS 区域
export AWS_REGION="${AWS_REGION:-us-east-1}"

# 模型在 S3 上的前缀路径
export MODEL_S3_PREFIX="${MODEL_S3_PREFIX:-models/XGBoost-Crypto-Market-Trend-Prediction}"

# 是否使用本地缓存 (true/false)
export USE_LOCAL_CACHE="${USE_LOCAL_CACHE:-true}"

# 本地缓存目录 (当 USE_LOCAL_CACHE=true 时使用)
export LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-XGBoost-Crypto-Market-Trend-Prediction/models}"

# ============================================================
# 评估参数
# ============================================================

# 评估的加密货币符号 (空格分隔)
SYMBOLS="${SYMBOLS:-BTCUSDT ETHUSDT SOLUSDT}"

# 数据获取数量
LIMIT="${LIMIT:-20000}"

# 回看K线数量 (用于训练)
LOOKBACK="${LOOKBACK:-3000}"

# 预测未来多少小时
HORIZON="${HORIZON:-4}"

# 每次评估之间跳过的K线数 (避免数据重叠)
STEP="${STEP:-24}"

# ============================================================
# 打印配置
# ============================================================

echo "============================================================"
echo "AWS S3 Rolling Evaluation Configuration"
echo "============================================================"
echo "AWS Bucket:      $AWS_BUCKET_NAME"
echo "AWS Region:     $AWS_REGION"
echo "Model S3 Prefix: $MODEL_S3_PREFIX"
echo "Use Local Cache: $USE_LOCAL_CACHE"
echo "Local Cache Dir: $LOCAL_CACHE_DIR"
echo "------------------------------------------------------------"
echo "Symbols:        $SYMBOLS"
echo "Limit:          $LIMIT"
echo "Lookback:       $LOOKBACK"
echo "Horizon:        $HORIZON hours"
echo "Step:           $STEP"
echo "============================================================"
echo ""

# ============================================================
# 检查依赖
# ============================================================

echo "Checking dependencies..."

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 检查 boto3
python -c "import boto3" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: boto3 not installed. Will use local cache only."
    echo "To install: pip install boto3"
fi

# 检查所需模块
python -c "import xgboost" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: xgboost not installed"
    exit 1
fi

python -c "import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pandas not installed"
    exit 1
fi

echo "Dependencies OK"
echo ""

# ============================================================
# 运行评估
# ============================================================

echo "Starting AWS S3 Rolling Evaluation..."
echo ""

# 使用 Python 运行评估脚本
python evaluate_rolling_1h_4h_aws.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Evaluation completed successfully!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Evaluation failed with exit code: $EXIT_CODE"
    echo "============================================================"
fi

exit $EXIT_CODE
