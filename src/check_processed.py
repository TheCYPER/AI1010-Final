# src/check_processed.py
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from config import TrainConfig
from preprocess import infer_column_types, build_preprocessor
from sklearn.compose import ColumnTransformer


def describe_matrix(name, X):
    """打印矩阵的基本信息（形状/稀疏度/NaN/Inf/类型）"""
    if sp.issparse(X):
        nnz = X.nnz
        total = X.shape[0] * X.shape[1]
        sparsity = 1 - nnz / total if total > 0 else np.nan
        X_csr = X.tocsr()
        # 稀疏矩阵的 NaN/Inf 检查需要转稠密或抽样；这里先抽样几行
        sample_rows = min(1000, X.shape[0])
        idx = np.random.choice(X.shape[0], sample_rows, replace=False) if X.shape[0] > 0 else []
        X_sample = X_csr[idx].toarray() if sample_rows > 0 else np.empty((0, X.shape[1]))
        n_nan = np.isnan(X_sample).sum()
        n_inf = np.isinf(X_sample).sum()
        print(f"[{name}] shape={X.shape}, sparse=True, sparsity≈{sparsity:.4f}, "
              f"sample_nan={n_nan}, sample_inf={n_inf}, dtype={X.dtype}")
    else:
        n_nan = np.isnan(X).sum() if np.issubdtype(X.dtype, np.number) else np.nan
        n_inf = np.isinf(X).sum() if np.issubdtype(X.dtype, np.number) else np.nan
        print(f"[{name}] shape={X.shape}, sparse=False, nan={n_nan}, inf={n_inf}, dtype={X.dtype}")


def try_get_feature_names(preprocessor, input_cols):
    """
    尝试从 ColumnTransformer 获取展开后的特征名。
    对于不支持的子变换（如 FunctionTransformer、自定义类），生成占位名。
    """
    names = None
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        pass

    # 回退：按分支逐个处理
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        # drop 的分支会是 'drop'；remainder 可能在 transformers_ 之外
        if trans == "drop":
            continue

        # 规范化列名列表
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            cols_list = list(cols)
        elif isinstance(cols, slice):
            # 极少场景使用；这里简化处理
            cols_list = [f"slice_{cols.start}_{cols.stop}"]
        else:
            cols_list = [str(cols)]

        # 尝试子变换的特征名
        try:
            sub = trans
            # 子流水线则取其最后一步
            if hasattr(trans, "named_steps"):
                sub = list(trans.named_steps.values())[-1]
            if hasattr(trans, "get_feature_names_out"):
                sub_names = trans.get_feature_names_out(cols_list)
                feature_names.extend([f"{name}__{n}" for n in sub_names])
                continue
        except Exception:
            pass

        # 如果拿不到，就用占位名（根据变换后的维度估算）
        # 先构造一个最小 batch 来估长（取第一行）
        # 注意：我们在 fit 后才能 transform
        try:
            dummy = preprocessor._transformer_to_input_indices.get(name, None)
        except Exception:
            dummy = None

        # 保守：占位 1 个或按输入列数量占位
        # 最稳妥的方式是后面用处理后的矩阵列数进行对齐；这里留到 main 里做。
        feature_names.extend([f"{name}__{c}" for c in cols_list])

    return feature_names


def top_variance_features(X, feature_names, k=20):
    """返回方差 Top-K 的特征及其方差（仅数值）"""
    if sp.issparse(X):
        Xd = X.toarray()
    else:
        Xd = X
    # 可能存在非数值 dtype；转换下
    Xd = Xd.astype(float, copy=False)
    var = np.nanvar(Xd, axis=0)
    idx = np.argsort(-var)[:min(k, Xd.shape[1])]
    out = []
    for i in idx:
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        out.append((i, name, var[i]))
    return out, var


def zero_or_near_constant(var, eps=1e-12):
    """零方差或近似常量的列索引集合"""
    return np.where(np.isnan(var) | (var <= eps))[0]


def sample_dataframe_view(X, feature_names, n=5, max_cols=50):
    """把处理后的矩阵转成小样本 DataFrame（避免超大内存）"""
    if sp.issparse(X):
        Xd = X[:n].toarray()
    else:
        Xd = np.asarray(X[:n])
    cols = feature_names[:Xd.shape[1]] if feature_names else [f"f{i}" for i in range(Xd.shape[1])]
    if len(cols) > max_cols:
        # 只展示前 max_cols 列，防止打印过长
        cols = cols[:max_cols]
        Xd = Xd[:, :max_cols]
    df_view = pd.DataFrame(Xd, columns=cols)
    return df_view


def main():
    parser = argparse.ArgumentParser(description="Check processed dataset after ColumnTransformer.")
    parser.add_argument("--config", type=str, default=None, help="(unused placeholder) keep default TrainConfig()")
    args = parser.parse_args()

    cfg = TrainConfig()

    # 1) 读取数据与列类型
    df = pd.read_csv(cfg.paths.train_csv)
    target = cfg.cols.target

    if cfg.cols.numeric is None or cfg.cols.categorical is None:
        num_cols, cat_cols = infer_column_types(df, target)
    else:
        num_cols, cat_cols = cfg.cols.numeric, cfg.cols.categorical

    X = df.drop(columns=[target])
    y = df[target]

    # 2) 划分数据
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=cfg.test_size, random_state=42, stratify=y
    )

    # 3) 预处理器
    pre = build_preprocessor(num_cols, cat_cols)

    # 4) 拟合与变换（注意：如果 pipeline 里有目标编码，必须传 y）
    X_tr_t = pre.fit_transform(X_tr, y_tr)
    X_va_t = pre.transform(X_va)

    print("\n=== Processed Dataset Summary ===")
    describe_matrix("train_transformed", X_tr_t)
    describe_matrix("valid_transformed", X_va_t)

    # 5) 特征名（尽力获取）
    feat_names = try_get_feature_names(pre, input_cols=X_tr.columns)
    # 如果长度与列数不一致，用占位补齐
    n_feats = X_tr_t.shape[1]
    if len(feat_names) != n_feats:
        feat_names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in range(n_feats)]

    print(f"\n#features after preprocessing: {n_feats}")
    print("First 30 feature names:")
    print(feat_names[:30])

    # 6) 每个分支的输出维度估算（通过差分统计）
    print("\n=== Branch Output (rough) ===")
    try:
        for name, trans, cols in pre.transformers_:
            if trans == "drop":
                continue
            # 子变换在 clone 后可单独 fit/transform；我们用单列/多列的子 ColumnTransformer 不易直接估维度，
            # 这里以“transform单分支”近似估算其列数：
            #   构造一个仅包含该分支的小 ColumnTransformer，并对训练集变换求列数。
            ct = ColumnTransformer(transformers=[(name, trans, cols)], remainder="drop")
            try:
                Xt_branch = ct.fit_transform(X_tr, y_tr)
                out_dim = Xt_branch.shape[1]
            except Exception:
                out_dim = None
            print(f"- {name:15s} -> out_dim={out_dim}")
    except Exception:
        print("(Skip detailed per-branch dimension due to transformer constraints.)")

    # 7) 零方差/近似常量、Top 方差
    print("\n=== Variance Diagnostics (train) ===")
    if sp.issparse(X_tr_t):
        Xd = X_tr_t.toarray()
    else:
        Xd = np.asarray(X_tr_t)
    Xd = Xd.astype(float, copy=False)

    top_k, var = top_variance_features(Xd, feat_names, k=20)
    zero_idx = zero_or_near_constant(var, eps=1e-12)

    print(f"Zero / near-constant features: {len(zero_idx)}")
    if len(zero_idx) > 0:
        print("Examples:", [feat_names[i] for i in zero_idx[:15]])

    print("\nTop-20 features by variance:")
    for i, name, v in top_k:
        print(f"{i:>5}  {name:<40} var={v:.6f}")

    # 8) 样本视图
    print("\n=== Head of transformed (first 5 rows, up to 50 cols) ===")
    df_view = sample_dataframe_view(X_tr_t, feat_names, n=5, max_cols=50)
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(df_view)

    print("\n[OK] Processed dataset checks completed.")


if __name__ == "__main__":
    main()
