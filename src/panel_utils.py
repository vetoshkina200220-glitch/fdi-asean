"""
Вспомогательные функции для панельного анализа ПИИ в странах ASEAN.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS


# ─── Цветовая палитра групп признаков ───────────────────────────────────────

GROUP_COLORS = {
    "ВВП":              "#2196F3",
    "Деглобализация":   "#FF9800",
    "Декарбонизация":   "#4CAF50",
    "Диджитализация":   "#9C27B0",
    "Демография":       "#F44336",
    "Долг":             "#795548",
    "Инфляция":         "#FF5722",
    "Процентная ставка":"#607D8B",
}

# Сопоставление признак → группа
FEATURE_GROUPS = {
    "GDP_growth":       "ВВП",
    "Trade_openness":   "Деглобализация",
    "Tariff_rate":      "Деглобализация",
    "Trade_in_services":"Деглобализация",
    "Renewable_energy": "Декарбонизация",
    "Energy_intensity": "Декарбонизация",
    "CO2_per_capita":   "Декарбонизация",
    "Internet_users":   "Диджитализация",
    "Secure_servers":   "Диджитализация",
    "Pop_growth":       "Демография",
    "Edu_expenditure":  "Демография",
    "Labor_force":      "Демография",
    "Gov_Debt":         "Долг",
    "Ext_Debt":         "Долг",
    "Debt_Service":     "Долг",
    "Inflation":        "Инфляция",
    "Real_interest":    "Процентная ставка",
}


# ─── Подготовка панельных данных ─────────────────────────────────────────────

def prepare_panel(df: pd.DataFrame, entity_col: str = "Country", time_col: str = "Year") -> pd.DataFrame:
    """Устанавливает MultiIndex (entity, time) для linearmodels."""
    df = df.copy()
    df[entity_col] = df[entity_col].astype("category")
    df = df.set_index([entity_col, time_col])
    return df


# ─── Тест Хаусмана (FE vs RE) ────────────────────────────────────────────────

def hausman_test(fe_model, re_model) -> dict:
    """
    Тест Хаусмана: H0 = RE состоятелен (нет корреляции эффектов со случайными).
    p < 0.05 → предпочесть Fixed Effects.

    Параметры
    ----------
    fe_model : результат PanelOLS(...).fit()
    re_model : результат RandomEffects(...).fit()

    Возвращает
    ----------
    dict с ключами: statistic, p_value, df, conclusion
    """
    fe_params = fe_model.params
    re_params = re_model.params

    # Берём общие коэффициенты (исключаем константу)
    common = fe_params.index.intersection(re_params.index)
    b_fe = fe_params[common]
    b_re = re_params[common]
    diff = b_fe - b_re

    # Ковариационные матрицы
    cov_fe = fe_model.cov.loc[common, common]
    cov_re = re_model.cov.loc[common, common]
    cov_diff = cov_fe - cov_re

    # χ² статистика
    try:
        cov_diff_inv = np.linalg.pinv(cov_diff.values)
        stat = float(diff.values @ cov_diff_inv @ diff.values)
    except np.linalg.LinAlgError:
        stat = np.nan

    df = len(common)
    p_value = 1 - stats.chi2.cdf(stat, df) if not np.isnan(stat) else np.nan
    conclusion = "Fixed Effects (FE)" if p_value < 0.05 else "Random Effects (RE)"

    return {"statistic": round(stat, 3), "p_value": round(p_value, 4),
            "df": df, "conclusion": conclusion}


# ─── Расчёт VIF ──────────────────────────────────────────────────────────────

def calc_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет Variance Inflation Factor для каждого признака.
    VIF > 5 — умеренная мультиколлинеарность, VIF > 10 — критическая.
    """
    X_clean = X.dropna()
    vif_data = pd.DataFrame({
        "feature": X_clean.columns,
        "VIF": [variance_inflation_factor(X_clean.values, i)
                for i in range(X_clean.shape[1])]
    })
    vif_data["flag"] = vif_data["VIF"].apply(
        lambda v: "OK" if v < 5 else ("умеренная" if v < 10 else "критическая")
    )
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# ─── Coefficient Plot (Forest Plot) ──────────────────────────────────────────

def plot_coef(result, feature_groups: dict = None, title: str = "Стандартизованные коэффициенты OLS",
              save_path: str = None, alpha: float = 0.05):
    """
    Горизонтальный forest plot стандартизованных коэффициентов.

    Параметры
    ----------
    result     : результат PanelOLS / OLS .fit()
    feature_groups : dict признак→группа (используется для раскраски)
    title      : заголовок графика
    save_path  : путь для сохранения (None = не сохранять)
    alpha      : порог значимости (по умолчанию 0.05)
    """
    params = result.params.copy()
    ci_low = result.conf_int()["lower"]
    ci_high = result.conf_int()["upper"]
    pvals = result.pvalues

    # Исключаем константу и фиктивные переменные стран
    mask = ~params.index.str.startswith(("Intercept", "EntityEffects", "Country"))
    params = params[mask]
    ci_low = ci_low[mask]
    ci_high = ci_high[mask]
    pvals = pvals[mask]

    # Сортируем по значению коэффициента
    order = params.sort_values().index
    params = params[order]
    ci_low = ci_low[order]
    ci_high = ci_high[order]
    pvals = pvals[order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.55)))

    for i, feat in enumerate(params.index):
        group = feature_groups.get(feat, "Прочее") if feature_groups else "Прочее"
        color = GROUP_COLORS.get(group, "#9E9E9E")
        significant = pvals[feat] < alpha

        ax.errorbar(params[feat], i,
                    xerr=[[params[feat] - ci_low[feat]], [ci_high[feat] - params[feat]]],
                    fmt="o", color=color, ecolor=color,
                    markersize=8 if significant else 5,
                    alpha=1.0 if significant else 0.45,
                    capsize=3, linewidth=1.5)

        # Метка незначимых
        if not significant:
            ax.text(ci_high[feat] + 0.01, i, "ns", va="center", fontsize=8, color="gray")

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params.index, fontsize=9)
    ax.set_xlabel("Стандартизованный коэффициент β", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    # Легенда по группам
    if feature_groups:
        present_groups = {feature_groups.get(f, "Прочее") for f in params.index}
        handles = [mpatches.Patch(color=GROUP_COLORS.get(g, "#9E9E9E"), label=g)
                   for g in GROUP_COLORS if g in present_groups]
        ax.legend(handles=handles, loc="lower right", fontsize=8, title="Группа")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ─── Инкрементальный R² по группам ──────────────────────────────────────────

def incremental_r2(panel_df: pd.DataFrame, target: str,
                   groups: dict, base_features: list = None) -> pd.DataFrame:
    """
    Вычисляет Δ R² при добавлении каждой группы признаков к базовой модели (FE OLS).
    Также вычисляет R² leave-one-group-out (исключение группы из полной модели).

    Параметры
    ----------
    panel_df       : DataFrame с MultiIndex (entity, time), стандартизованные признаки
    target         : название целевой переменной
    groups         : dict группа → [список признаков]
    base_features  : список базовых контрольных признаков (None = пустая база)

    Возвращает
    ----------
    DataFrame: group / delta_r2_add / r2_without / delta_r2_drop
    """
    base_features = base_features or []
    all_features = [f for feats in groups.values() for f in feats]
    full_features = list(set(base_features + all_features))

    def fit_r2(features):
        if not features:
            return 0.0
        clean = panel_df[[target] + features].dropna()
        try:
            mod = PanelOLS(clean[target], clean[features],
                           entity_effects=True, drop_absorbed=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            return res.rsquared_within
        except Exception:
            return np.nan

    r2_base = fit_r2(base_features)
    r2_full = fit_r2(full_features)

    rows = []
    for group, feats in groups.items():
        feats_present = [f for f in feats if f in panel_df.columns]
        if not feats_present:
            continue

        # Δ R² при добавлении группы к базовой модели
        r2_with = fit_r2(base_features + feats_present)
        delta_add = r2_with - r2_base

        # R² без этой группы (leave-one-group-out)
        remaining = [f for f in full_features if f not in feats_present]
        r2_without = fit_r2(remaining)
        delta_drop = r2_full - r2_without

        rows.append({
            "Группа":           group,
            "ΔR² (добавление)": round(delta_add, 4),
            "R² без группы":    round(r2_without, 4),
            "ΔR² (исключение)": round(delta_drop, 4),
        })

    return pd.DataFrame(rows).sort_values("ΔR² (добавление)", ascending=False).reset_index(drop=True)


# ─── F-тест совместной значимости группы ────────────────────────────────────

def group_f_test(panel_df: pd.DataFrame, target: str,
                 all_features: list, group_features: list) -> dict:
    """
    F-тест совместной значимости группы признаков (H0: все β группы = 0).

    Реализован через сравнение R² полной и ограниченной FE-моделей:
        F = ((R²_full - R²_restr) / k) / ((1 - R²_full) / (N - K - 1))
    где k = кол-во признаков группы, K = кол-во признаков полной модели.

    Параметры
    ----------
    panel_df      : DataFrame с MultiIndex (entity, time)
    target        : целевая переменная
    all_features  : список всех признаков полной модели
    group_features: признаки тестируемой группы

    Возвращает
    ----------
    dict: F_stat, p_value, df_num, df_den, significant
    """
    present = [f for f in group_features if f in all_features]
    if not present:
        return {"F_stat": np.nan, "p_value": np.nan, "df_num": 0, "df_den": 0, "significant": False}

    restricted = [f for f in all_features if f not in present]

    def fit_r2_within(features):
        clean = panel_df[[target] + features].dropna()
        if len(clean) < len(features) + 3 or not features:
            return np.nan, 0
        try:
            mod = PanelOLS(clean[target], clean[features],
                           entity_effects=True, drop_absorbed=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            return res.rsquared_within, res.nobs
        except Exception:
            return np.nan, 0

    r2_full, n_obs = fit_r2_within(all_features)
    r2_restr, _ = fit_r2_within(restricted) if restricted else (0.0, 0)

    if np.isnan(r2_full) or np.isnan(r2_restr):
        return {"F_stat": np.nan, "p_value": np.nan, "df_num": len(present), "df_den": 0, "significant": False}

    k = len(present)
    K = len(all_features)
    df_den = max(n_obs - K - 1, 1)

    denom = (1 - r2_full) / df_den
    if denom <= 0:
        return {"F_stat": np.nan, "p_value": np.nan, "df_num": k, "df_den": df_den, "significant": False}

    f_stat = ((r2_full - r2_restr) / k) / denom
    p_val = 1 - stats.f.cdf(f_stat, k, df_den)

    return {
        "F_stat":      round(f_stat, 3),
        "p_value":     round(p_val, 4),
        "df_num":      k,
        "df_den":      int(df_den),
        "significant": bool(p_val < 0.05),
    }


# ─── Вспомогательная печать таблицы результатов регрессии ───────────────────

def regression_table(result, title: str = "Результаты регрессии") -> pd.DataFrame:
    """
    Форматирует результаты PanelOLS/OLS в читаемую таблицу.
    Добавляет метки значимости: *** p<0.01, ** p<0.05, * p<0.1.
    """
    params = result.params
    se = result.std_errors
    tstat = result.tstats
    pval = result.pvalues

    # Исключаем эффекты стран
    mask = ~params.index.str.startswith(("EntityEffects", "Country"))
    df = pd.DataFrame({
        "Коэффициент β": params[mask].round(4),
        "Std. Error":    se[mask].round(4),
        "t-stat":        tstat[mask].round(3),
        "p-value":       pval[mask].round(4),
    })

    def stars(p):
        if p < 0.01:  return "***"
        if p < 0.05:  return "**"
        if p < 0.10:  return "*"
        return ""

    df["Знач."] = df["p-value"].apply(stars)
    df.index.name = "Признак"
    return df
