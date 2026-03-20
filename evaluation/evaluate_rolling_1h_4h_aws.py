"""
Rolling Window Model Evaluation - AWS S3 版本
用与训练相同的方式：
- 从 AWS S3 获取模型
- 获取20000根数据
- 用过去N根训练
- 预测未来4小时
- 确保训练/测试不重合
"""

import sys
import os
import tempfile
import shutil

# AWS 配置 - 请根据实际情况修改
AWS_BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME', 'your-bucket-name')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
MODEL_S3_PREFIX = os.environ.get('MODEL_S3_PREFIX', 'models/XGBoost-Crypto-Market-Trend-Prediction')

# 是否使用本地缓存
USE_LOCAL_CACHE = os.environ.get('USE_LOCAL_CACHE', 'true').lower() == 'true'
LOCAL_CACHE_DIR = os.environ.get('LOCAL_CACHE_DIR', 'XGBoost-Crypto-Market-Trend-Prediction/models')

sys.path.insert(0, 'XGBoost-Crypto-Market-Trend-Prediction')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
)
import joblib
from xgboost import XGBClassifier

# 尝试导入 boto3
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not available, will use local cache only")

from src.data_fetcher import fetch_klines
from src.feature_engineering import build_features


def download_model_from_s3(symbol: str, temp_dir: str) -> str:
    """
    从 AWS S3 下载模型文件到临时目录
    """
    if not BOTO3_AVAILABLE:
        # 如果 boto3 不可用，尝试使用本地缓存
        local_path = f"{LOCAL_CACHE_DIR}/{symbol}.joblib"
        if os.path.exists(local_path):
            print(f"  Using local cache: {local_path}")
            return local_path
        raise Exception(f"boto3 not available and local cache not found: {local_path}")
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    s3_key = f"{MODEL_S3_PREFIX}/{symbol}.joblib"
    local_path = os.path.join(temp_dir, f"{symbol}.joblib")
    
    try:
        print(f"  Downloading s3://{AWS_BUCKET_NAME}/{s3_key} ...")
        s3_client.download_file(AWS_BUCKET_NAME, s3_key, local_path)
        print(f"  Downloaded to: {local_path}")
        return local_path
    except ClientError as e:
        print(f"  S3 download failed: {e}")
        # 尝试本地缓存
        local_cache = f"{LOCAL_CACHE_DIR}/{symbol}.joblib"
        if os.path.exists(local_cache):
            print(f"  Falling back to local cache: {local_cache}")
            return local_cache
        raise


def get_model_payload(symbol: str, temp_dir: str = None):
    """
    获取模型 payload，优先从 S3 下载，其次使用本地缓存
    """
    # 如果使用本地缓存且文件存在，直接使用
    if USE_LOCAL_CACHE:
        local_path = f"{LOCAL_CACHE_DIR}/{symbol}.joblib"
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return joblib.load(local_path)
    
    # 尝试从 S3 下载
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    model_path = download_model_from_s3(symbol, temp_dir)
    return joblib.load(model_path)


def rolling_evaluate_optimized(
    symbol,
    limit=20000,
    horizon=4,
    lookback=3000,  # 用过去3000根（约125天）训练
    step=24,        # 每24小时预测一次（避免数据重叠）
    temp_dir=None,
):
    """
    滚动窗口评估 - 确保训练/测试不重合
    """
    print(f"\n{'='*60}")
    print(f"Rolling: {symbol} | Lookback: {lookback} | Horizon: {horizon}")
    print('='*60)
    
    # 创建临时目录用于存放 S3 下载的模型
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # 获取20000根数据
    df = fetch_klines(symbol, interval='1h', limit=limit)
    print(f"Raw data: {len(df)} rows")
    print(f"Period: {df.index[0]} ~ {df.index[-1]}")
    
    # 构建特征
    df_feat = build_features(df, horizon=horizon)
    n = len(df_feat)
    print(f"After features: {n} rows")
    print(f"Period: {df_feat.index[0]} ~ {df_feat.index[-1]}")
    
    if n < lookback + 100:
        print(f"ERROR: Not enough data")
        return None
    
    # 从 S3 或本地加载模型获取特征列
    try:
        payload = get_model_payload(symbol, temp_dir)
        feature_cols = payload.get('feature_columns', [])
        print(f"Using {len(feature_cols)} features from trained model (S3)")
    except Exception as e:
        print(f"Warning: Failed to load model from S3: {e}")
        # 如果加载失败，使用默认特征
        FEATURE_COLUMNS = [
            "sma_7", "sma_14", "sma_21", "sma_50",
            "ema_7", "ema_14", "ema_21", "ema_50",
            "rsi_14",
            "macd_line", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_pct_b", "bb_width",
            "atr_14",
            "stoch_k", "stoch_d",
            "obv",
            "return_1h", "return_4h", "return_24h",
            "body_ratio", "upper_wick", "lower_wick", "hl_spread",
            "volume_ma_7", "volume_ma_14", "rel_volume", "taker_buy_ratio",
            "roc_3", "roc_6", "roc_12", "roc_24",
        ]
        feature_cols = [c for c in FEATURE_COLUMNS if c in df_feat.columns]
        print(f"Using {len(feature_cols)} default features")
    
    # 滚动窗口评估
    # 每次预测后，跳过step个样本，确保不重合
    results = []
    
    # 开始索引：需要足够的历史数据
    start_idx = lookback + horizon + 100
    
    for i in range(start_idx, n, step):
        # 训练数据：[i - lookback - horizon : i - horizon]
        # 测试数据：i - horizon (预测这个时间点之后4小时的涨跌)
        
        train_end = i - horizon
        train_start = train_end - lookback
        test_idx = i - horizon
        
        if train_start < 0 or test_idx < 0:
            continue
        
        train = df_feat.iloc[train_start:train_end]
        test = df_feat.iloc[test_idx:test_idx+1]
        
        if len(train) < lookback * 0.8:
            continue
        
        # 确保训练集有两个类别
        unique_targets = np.unique(train['target'].values)
        if len(unique_targets) < 2:
            continue
        
        # 确保特征列存在
        available_features = [c for c in feature_cols if c in train.columns]
        if len(available_features) < 5:
            continue
        
        try:
            X_train = train[available_features].values
            y_train = train['target'].values
            X_test = test[available_features].values.reshape(1, -1)
            y_test = test['target'].values[0]
            
            # 训练模型
            model = XGBClassifier(
                n_estimators=634,  # 使用优化版模型的参数
                max_depth=6,
                learning_rate=0.0135,
                subsample=0.71,
                colsample_bytree=0.69,
                min_child_weight=12.6,
                gamma=0.04,
                reg_alpha=1.67,
                reg_lambda=6.53,
                random_state=42,
                verbosity=0,
                scale_pos_weight=0.94,
            )
            model.fit(X_train, y_train)
            
            # 预测
            prob = model.predict_proba(X_test)[0, 1]
            pred = int(prob >= 0.5)
            
            results.append({
                'train_start': train.index[0],
                'train_end': train.index[-1],
                'test_time': test.index[0],
                'true': y_test,
                'pred': pred,
                'proba': prob,
            })
        except Exception as e:
            continue
    
    if not results:
        print("No valid predictions!")
        return None
    
    # 统计
    df_res = pd.DataFrame(results)
    y_true = df_res['true'].values
    y_pred = df_res['pred'].values
    y_proba = df_res['proba'].values
    
    print(f"\n--- Results ---")
    print(f"Total predictions: {len(results)}")
    print(f"First prediction:")
    print(f"  Train: {df_res['train_start'].iloc[0]} ~ {df_res['train_end'].iloc[0]}")
    print(f"  Test: {df_res['test_time'].iloc[0]}")
    print(f"Last prediction:")
    print(f"  Train: {df_res['train_start'].iloc[-1]} ~ {df_res['train_end'].iloc[-1]}")
    print(f"  Test: {df_res['test_time'].iloc[-1]}")
    
    # 指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Confusion Matrix ---")
    print(f"            Predicted")
    print(f"            Down    Up")
    print(f"Actual Down  {tn:4d}  {fp:4d}")
    print(f"Actual Up    {fn:4d}  {tp:4d}")
    
    print(f"\n--- Metrics ---")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  F1:                {f1:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"  ROC-AUC:           {auc:.4f}")
    
    print(f"\n--- Baseline ---")
    print(f"  Random:    0.5000")
    print(f"  Always Up: {np.mean(y_true):.4f}")
    
    return {
        'symbol': symbol,
        'lookback': lookback,
        'horizon': horizon,
        'n': len(results),
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': auc,
    }


# Run
if __name__ == "__main__":
    import tempfile
    
    print("\n" + "#"*70)
    print("# ROLLING WINDOW EVALUATION - AWS S3 VERSION")
    print("# 用20000根数据，滚动窗口评估，从S3加载模型")
    print("#"*70)
    
    # AWS 配置
    print(f"\nAWS Configuration:")
    print(f"  Bucket: {AWS_BUCKET_NAME}")
    print(f"  Region: {AWS_REGION}")
    print(f"  Model Prefix: {MODEL_S3_PREFIX}")
    print(f"  Use Local Cache: {USE_LOCAL_CACHE}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        results_list = []
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            result = rolling_evaluate_optimized(
                symbol=symbol,
                limit=20000,     # 获取20000根
                horizon=4,       # 预测未来4小时
                lookback=3000,   # 用过去3000根训练（约125天）
                step=24,         # 每24小时预测一次
                temp_dir=temp_dir,
            )
            if result:
                results_list.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Symbol':<12} {'N':>5} {'Acc':>7} {'BalAcc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
        print("-"*70)
        
        for r in results_list:
            print(f"{r['symbol']:<12} {r['n']:>5} {r['accuracy']:>7.4f} "
                  f"{r['balanced_accuracy']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['roc_auc']:>7.4f}")
        
        if results_list:
            avg_acc = np.mean([r['accuracy'] for r in results_list])
            avg_auc = np.mean([r['roc_auc'] for r in results_list])
            print("-"*70)
            print(f"{'Average':<12} {'':<5} {avg_acc:>7.4f} {'':<7} {'':<7} {'':<7} {'':<7} {avg_auc:>7.4f}")
    finally:
        # 清理临时目录
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
