-- Source records: the result docstrings in experiments/exp_base/exp*.py and
-- experiments/exp_dx_outlook/exp*.py, and the completed result figures under
-- experiments/exp_dx_outlook_lightgbm/results/, reviewed on 2026-08-09.
-- This query materializes the comparison rows used in artifact.json.
WITH experiment_summary AS (
    SELECT 'base' AS family, 'exp04-D' AS experiment, '業界統合' AS method,
           0.6702 AS nested_oof_f1, 0.6710 AS mean_f1, 0.0434 AS std_f1
    UNION ALL SELECT 'base', 'exp02 weight=2.0', 'クラス重み2.0', 0.6701, 0.6700, 0.0705
    UNION ALL SELECT 'base', 'exp03', '重複除外', 0.6667, 0.6676, 0.0473
    UNION ALL SELECT 'base', 'exp05 top20', '上位20特徴量', 0.6579, 0.6580, 0.0415
    UNION ALL SELECT 'base', 'exp06-G', '3軸すべて', 0.6577, 0.6551, 0.0421
    UNION ALL SELECT 'base', 'exp01', 'ベースライン', 0.6562, 0.6562, 0.0402
    UNION ALL SELECT 'text', 'exp03', '名詞のみ', 0.5815, 0.5876, 0.0605
    UNION ALL SELECT 'text', 'exp06', '全品詞', 0.5740, 0.5801, 0.0415
    UNION ALL SELECT 'text', 'exp01', '文字N-gram', 0.5692, 0.5735, 0.0666
    UNION ALL SELECT 'text', 'exp04', '名詞＋動詞', 0.5683, 0.5717, 0.0513
    UNION ALL SELECT 'text', 'exp02', '品詞別ブロック', 0.5664, 0.5722, 0.0571
    UNION ALL SELECT 'text', 'exp05', '形容詞まで', 0.5578, 0.5628, 0.0442
    UNION ALL SELECT 'text-lightgbm', 'exp04', '名詞＋動詞', 0.5369, 0.5340, 0.0530
    UNION ALL SELECT 'text-lightgbm', 'exp03', '名詞のみ', 0.5051, 0.5038, 0.0281
)
SELECT family, experiment, method, nested_oof_f1, mean_f1, std_f1
FROM experiment_summary
ORDER BY family, nested_oof_f1 DESC;
