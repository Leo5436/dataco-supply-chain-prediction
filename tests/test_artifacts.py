import pathlib

import joblib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sample_data_contract():
    """樣本資料必須含有下游需要的欄位，且標籤只有 0/1"""
    df = pd.read_csv(ROOT / 'data' / 'sample_orders.csv', encoding='latin-1')
    assert len(df) > 0
    for col in ['Late_delivery_risk', 'Days for shipment (scheduled)', 'Shipping Mode']:
        assert col in df.columns, f'缺少必要欄位: {col}'
    assert set(df['Late_delivery_risk'].unique()) <= {0, 1}


def test_no_leakage_in_features():
    """確認 data leakage 欄位沒有偷偷混進訓練特徵"""
    cols = joblib.load(ROOT / 'feature_columns.pkl')
    assert 'Days for shipping (real)' not in cols
    assert not any('Delivery Status' in c for c in cols)


def test_model_and_columns_are_consistent():
    """模型期待的特徵數，必須等於存下來的欄位數"""
    model = joblib.load(ROOT / 'xgb_late_delivery.pkl')
    cols = joblib.load(ROOT / 'feature_columns.pkl')
    assert model.n_features_in_ == len(cols)


def test_model_can_predict_like_app_does():
    """模擬 app.py 的輸入方式，確認能產出合法機率"""
    model = joblib.load(ROOT / 'xgb_late_delivery.pkl')
    cols = joblib.load(ROOT / 'feature_columns.pkl')
    X = pd.DataFrame(0, index=[0], columns=cols)
    prob = model.predict_proba(X)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_threshold_is_sane():
    """Section 7 決定的門檻必須是合法機率值"""
    threshold = float(joblib.load(ROOT / 'threshold.pkl'))
    assert 0.0 < threshold < 1.0
