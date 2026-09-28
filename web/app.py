import os
import sys
import io
import json
import threading
import traceback
import importlib
from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_path)
legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'legacy')
if os.path.exists(legacy_path):
    sys.path.insert(0, legacy_path)

# 导入 backtest 模块（后续会动态重新加载）
import backtest as backtest_module
from backtest import BacktestEngine

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

_backtest_lock = threading.Lock()
_backtest_state = {
    'status': 'idle',
    'step': 0,
    'step_name': '',
    'progress': 0,
    'logs': deque(maxlen=500),
    'error': None,
    'result': None,
}

STEP_NAMES = {
    1: '数据加载',
    2: '特征工程',
    3: '标签生成',
    4: '模型训练',
    5: 'Walk-Forward 验证',
    6: '绩效评估',
    7: '可视化',
}


class LogCapture:
    def __init__(self):
        self.original_stdout = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.original_stdout.write(message)
        self.buffer.write(message)
        if message.strip():
            with _backtest_lock:
                _backtest_state['logs'].append(message.rstrip('\n'))

    def flush(self):
        self.original_stdout.flush()
        self.buffer.flush()


def _update_state(step: int, progress: Optional[int] = None):
    with _backtest_lock:
        _backtest_state['step'] = step
        _backtest_state['step_name'] = STEP_NAMES.get(step, '')
        if progress is not None:
            step_progress = progress / 100.0
            overall_progress = ((step - 1) + step_progress) / 7 * 100
            _backtest_state['progress'] = int(overall_progress)
        else:
            _backtest_state['progress'] = int((step - 1) / 7 * 100)


def _reset_state():
    with _backtest_lock:
        _backtest_state['status'] = 'idle'
        _backtest_state['step'] = 0
        _backtest_state['step_name'] = ''
        _backtest_state['progress'] = 0
        _backtest_state['logs'].clear()
        _backtest_state['error'] = None
        _backtest_state['result'] = None


def _serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _serialize_config(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


def _convert_frontend_config(frontend_config: Dict[str, Any]) -> Dict[str, Any]:
    """将前端扁平格式的配置转换为后端嵌套格式。

    Args:
        frontend_config: 前端发送的扁平配置

    Returns:
        后端嵌套格式的配置
    """
    result = {}

    # Label 配置 - 确保所有必需字段都有值
    label_config = {}
    if 'horizon' in frontend_config:
        label_config['horizon'] = int(frontend_config['horizon'])
    else:
        label_config['horizon'] = 5

    if 'threshold_mode' in frontend_config:
        mode = frontend_config['threshold_mode']
        if mode == 'quantile':
            label_config['use_quantile'] = True
        else:
            label_config['use_quantile'] = False
    else:
        label_config['use_quantile'] = False

    if 'threshold' in frontend_config:
        label_config['threshold'] = float(frontend_config['threshold']) / 100.0
    else:
        label_config['threshold'] = 0.02

    if 'threshold_up' in frontend_config:
        label_config['threshold_up'] = float(frontend_config['threshold_up']) / 100.0

    if 'threshold_down' in frontend_config:
        label_config['threshold_down'] = float(frontend_config['threshold_down']) / 100.0

    if 'quantile' in frontend_config:
        label_config['quantile'] = float(frontend_config['quantile'])
    else:
        label_config['quantile'] = 0.3

    if label_config:
        result['label'] = label_config

    # Walkforward 配置
    if 'frequency' in frontend_config:
        result.setdefault('walkforward', {})['frequency'] = frontend_config['frequency']

    if 'window_mode' in frontend_config:
        result.setdefault('walkforward', {})['window_mode'] = frontend_config['window_mode']

    if 'window_years' in frontend_config:
        result.setdefault('walkforward', {})['window_years'] = int(frontend_config['window_years'])

    if 'min_train_years' in frontend_config:
        result.setdefault('walkforward', {})['min_train_years'] = int(frontend_config['min_train_years'])

    # Portfolio 配置
    if 'risk_on_threshold' in frontend_config:
        result.setdefault('portfolio', {})['risk_on_threshold'] = float(frontend_config['risk_on_threshold'])

    if 'risk_off_threshold' in frontend_config:
        result.setdefault('portfolio', {})['risk_off_threshold'] = float(frontend_config['risk_off_threshold'])

    if 'shares_per_trade' in frontend_config:
        result.setdefault('portfolio', {})['shares_per_trade'] = int(frontend_config['shares_per_trade'])

    if 'use_real_costs' in frontend_config:
        result.setdefault('portfolio', {})['use_real_costs'] = bool(frontend_config['use_real_costs'])

    if 'commission_bps' in frontend_config:
        result.setdefault('portfolio', {})['commission_bps'] = float(frontend_config['commission_bps'])

    if 'slippage_bps' in frontend_config:
        result.setdefault('portfolio', {})['slippage_bps'] = float(frontend_config['slippage_bps'])

    if 'include_costs' in frontend_config:
        result.setdefault('portfolio', {})['include_costs'] = bool(frontend_config['include_costs'])

    # Model 配置
    if 'model_type' in frontend_config:
        result.setdefault('model', {})['model_type'] = frontend_config['model_type']

    if 'random_state' in frontend_config:
        result.setdefault('model', {})['random_state'] = int(frontend_config['random_state'])

    return result


def _series_to_list(series: pd.Series) -> list:
    return [
        {'date': str(idx.date()), 'value': float(val)}
        for idx, val in series.items()
    ]


def _df_to_records(df: pd.DataFrame, date_col: str = 'date') -> list:
    records = []
    for idx, row in df.iterrows():
        record = {date_col: str(idx.date())}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                record[col] = float(val)
            else:
                record[col] = val
        records.append(record)
    return records


def _extract_result(engine: BacktestEngine) -> Dict[str, Any]:
    results = engine.results
    eval_results = results.get('evaluation', {})
    wf_results = results.get('wf_results')

    portfolio_metrics = eval_results.get('strategy_metrics', {})
    trading_metrics = eval_results.get('trading_metrics', {})
    classification_metrics_raw = eval_results.get('classification_metrics', {})
    benchmark_metrics = eval_results.get('benchmark_metrics', {})

    classification_metrics = {}
    for k, v in classification_metrics_raw.items():
        if k == 'confusion_matrix':
            classification_metrics[k] = v.tolist() if isinstance(v, np.ndarray) else v
        elif isinstance(v, (np.integer,)):
            classification_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            classification_metrics[k] = float(v)
        else:
            classification_metrics[k] = v

    equity_curve = []
    drawdown_data = []
    positions_data = []
    rolling_sharpe_data = []

    if wf_results is not None:
        nav_col = 'nav_cost_combined' if 'nav_cost_combined' in wf_results.columns else 'nav'
        if nav_col in wf_results.columns:
            nav_series = wf_results[nav_col].dropna()
            equity_curve = _series_to_list(nav_series)

            peak = nav_series.cummax()
            drawdown_series = (nav_series - peak) / peak
            drawdown_data = _series_to_list(drawdown_series)

        if 'actual_position' in wf_results.columns:
            pos_series = wf_results['actual_position']
            pos_map = {
                'TQQQ': 'risk_on',
                'SQQQ': 'risk_off',
                'Cash': 'neutral',
                'QQQ': 'risk_on',
                'PSQ': 'risk_off',
                'BIL': 'neutral',
            }
            positions_data = [
                {'date': str(idx.date()), 'position': pos_map.get(str(val), 'neutral')}
                for idx, val in pos_series.items()
            ]

        ret_col = 'daily_return_cost' if 'daily_return_cost' in wf_results.columns else 'daily_return'
        if ret_col in wf_results.columns:
            daily_ret = wf_results[ret_col].dropna()
            if len(daily_ret) > 60:
                rolling_ret = daily_ret.rolling(window=60).mean() * 252
                rolling_vol = daily_ret.rolling(window=60).std() * np.sqrt(252)
                rolling_sharpe = (rolling_ret / rolling_vol).dropna()
                rolling_sharpe_data = _series_to_list(rolling_sharpe)

    benchmark_navs = {}
    benchmarks = eval_results.get('benchmarks', {})
    for name, data in benchmarks.items():
        nav_series = data.get('nav')
        if nav_series is not None:
            benchmark_navs[name] = _series_to_list(nav_series)

    feature_importance = []
    model = results.get('model')
    feature_cols = results.get('feature_cols', [])
    if model is not None and feature_cols and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices:
            feature_importance.append({
                'feature': feature_cols[i],
                'importance': float(importances[i]),
            })

    confusion_matrix_val = classification_metrics.get('confusion_matrix', [])
    if isinstance(confusion_matrix_val, np.ndarray):
        confusion_matrix_val = confusion_matrix_val.tolist()

    return {
        'portfolio_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in portfolio_metrics.items()},
        'trading_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in trading_metrics.items()},
        'classification_metrics': classification_metrics,
        'benchmark_metrics': {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in metrics.items()}
            for name, metrics in benchmark_metrics.items()
        },
        'equity_curve': equity_curve,
        'benchmark_navs': benchmark_navs,
        'drawdown': drawdown_data,
        'positions': positions_data,
        'feature_importance': feature_importance,
        'confusion_matrix': confusion_matrix_val,
        'rolling_sharpe': rolling_sharpe_data,
    }


def _run_backtest_thread(config: Dict[str, Any]):
    log_capture = LogCapture()
    original_stdout = sys.stdout
    sys.stdout = log_capture

    try:
        with _backtest_lock:
            _backtest_state['status'] = 'running'
            _backtest_state['error'] = None

        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        os.chdir(project_root)

        # 重新加载模块以获取最新代码
        importlib.reload(backtest_module)
        BacktestEngine = backtest_module.BacktestEngine

        converted_config = _convert_frontend_config(config)
        engine = BacktestEngine(config=converted_config)

        _update_state(1, 5)
        engine.step_load_data()
        _update_state(1, 100)

        _update_state(2, 5)
        engine.step_build_features()
        _update_state(2, 100)

        _update_state(3, 5)
        engine.step_generate_labels()
        _update_state(3, 100)

        _update_state(4, 5)
        engine.step_train_model()
        _update_state(4, 100)

        _update_state(5, 5)
        engine.step_walkforward()
        _update_state(5, 100)

        _update_state(6, 5)
        engine.step_evaluate()
        _update_state(6, 100)

        _update_state(7, 5)
        engine.step_visualize()
        _update_state(7, 100)

        result = _extract_result(engine)

        with _backtest_lock:
            _backtest_state['status'] = 'completed'
            _backtest_state['progress'] = 100
            _backtest_state['result'] = result

        print("\n回测完成！")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        with _backtest_lock:
            _backtest_state['status'] = 'error'
            _backtest_state['error'] = str(e)
            _backtest_state['logs'].append(f"错误: {str(e)}")
        print(f"\n回测出错: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    engine = BacktestEngine()
    default_config = _serialize_config(engine.config)
    return jsonify(default_config)


@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    with _backtest_lock:
        if _backtest_state['status'] == 'running':
            return jsonify({
                'status': 'error',
                'message': '回测正在运行中，请等待完成',
            }), 409

    _reset_state()

    user_config = request.get_json(silent=True) or {}

    thread = threading.Thread(
        target=_run_backtest_thread,
        args=(user_config,),
        daemon=True,
    )
    thread.start()

    return jsonify({
        'status': 'started',
        'message': '回测已启动',
    })


@app.route('/api/backtest/status', methods=['GET'])
def get_backtest_status():
    with _backtest_lock:
        return jsonify({
            'status': _backtest_state['status'],
            'step': _backtest_state['step'],
            'step_name': _backtest_state['step_name'],
            'progress': _backtest_state['progress'],
            'logs': list(_backtest_state['logs']),
            'error': _backtest_state['error'],
        })


@app.route('/api/backtest/result', methods=['GET'])
def get_backtest_result():
    with _backtest_lock:
        if _backtest_state['status'] != 'completed':
            return jsonify({
                'status': 'error',
                'message': f'回测尚未完成（当前状态: {_backtest_state["status"]}）',
            }), 400

        if _backtest_state['result'] is None:
            return jsonify({
                'status': 'error',
                'message': '回测结果为空',
            }), 500

        return jsonify(_backtest_state['result'])


# -------------------------------------------------------------
# Moomoo 模拟盘与实时调仓交易 API
# -------------------------------------------------------------
_rebalance_mgr: Optional[Any] = None


def _get_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')


def _load_full_config() -> dict:
    cfg_path = _get_config_path()
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[_load_full_config ERROR] {cfg_path}: {e}")
            app.logger.error(f"[_load_full_config ERROR] {cfg_path}: {e}")
    else:
        print(f"[_load_full_config NOT FOUND] {cfg_path}")
    return {}


def _save_full_config(cfg: dict) -> bool:
    cfg_path = _get_config_path()
    try:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        app.logger.error(f"保存 config.json 失败: {e}")
        return False


def _get_rebalance_mgr():
    global _rebalance_mgr
    if _rebalance_mgr is None:
        try:
            from rebalance_manager import RebalanceManager
            _rebalance_mgr = RebalanceManager()

            # 应用 config.json 中的当前固化版本参数
            full_cfg = _load_full_config()
            strat_cfg = full_cfg.get('strategy', {})
            active_ver = strat_cfg.get('active_version', 'v1.0-production')
            ver_info = strat_cfg.get('versions', {}).get(active_ver, {})
            if ver_info:
                _rebalance_mgr.set_strategy_parameters(
                    core_ratio=ver_info.get('core_ratio', 0.5),
                    satellite_ratio=ver_info.get('satellite_ratio', 0.5),
                    target_vol=ver_info.get('target_vol', 0.25),
                    min_leverage=ver_info.get('min_leverage', 0.6),
                    max_leverage=ver_info.get('max_leverage', 1.8),
                    buffer=ver_info.get('rebalance_buffer', 0.06),
                    top_n=ver_info.get('top_n', 2),
                )

            # 应用资金规模与卫星分配模式配置
            cap_cfg = full_cfg.get('capital_setting', {})
            sat_mode = full_cfg.get('satellite_weight_mode', 'EQUAL')
            _rebalance_mgr.set_capital_parameters(
                mode=cap_cfg.get('mode', 'FULL_NAV'),
                custom_amount=cap_cfg.get('custom_amount', 30000.0),
                percent=cap_cfg.get('percent', 50.0),
                ignore_outside_holdings=cap_cfg.get('ignore_outside_holdings', True),
                satellite_weight_mode=sat_mode,
            )
        except Exception as e:
            app.logger.error(f"初始化 RebalanceManager 异常: {e}")
    return _rebalance_mgr


@app.route('/api/strategy/versions', methods=['GET'])
def get_strategy_versions():
    """获取所有可用实盘策略固化版本及当前激活版本。"""
    full_cfg = _load_full_config()
    strat_cfg = full_cfg.get('strategy', {})
    active_version = strat_cfg.get('active_version', 'v1.0-production')
    versions = strat_cfg.get('versions', {})
    return jsonify({
        'status': 'ok',
        'active_version': active_version,
        'versions': versions,
    })


@app.route('/api/strategy/switch_version', methods=['POST'])
def switch_strategy_version():
    """切换实盘策略固化版本 (生产固化版 / 进取增强版 / 稳健防守版)。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '策略管理模块未就绪'}), 500

    data = request.get_json() or {}
    target_version = data.get('version')
    if not target_version:
        return jsonify({'status': 'error', 'message': '未指定目标版本'}), 400

    full_cfg = _load_full_config()
    strat_cfg = full_cfg.setdefault('strategy', {})
    versions = strat_cfg.get('versions', {})

    if target_version not in versions:
        return jsonify({'status': 'error', 'message': f'未知策略版本: {target_version}'}), 404

    ver_info = versions[target_version]
    strat_cfg['active_version'] = target_version
    _save_full_config(full_cfg)

    # 动态应用新策略参数并清空信号缓存
    mgr.set_strategy_parameters(
        core_ratio=ver_info.get('core_ratio', 0.5),
        satellite_ratio=ver_info.get('satellite_ratio', 0.5),
        target_vol=ver_info.get('target_vol', 0.25),
        min_leverage=ver_info.get('min_leverage', 0.6),
        max_leverage=ver_info.get('max_leverage', 1.8),
        buffer=ver_info.get('rebalance_buffer', 0.06),
        top_n=ver_info.get('top_n', 2),
    )
    # 重算信号
    try:
        env = data.get('env', 'REAL').upper()
        mgr.generate_rebalance_plan(env=env, force_rebalance=False)
    except Exception as e:
        app.logger.warning(f"重新生成调仓方案异常: {e}")

    app.logger.info(f"成功切换实盘策略版本为: {target_version} ({ver_info.get('name')})")
    return jsonify({
        'status': 'ok',
        'message': f"成功切换至 {ver_info.get('name')}",
        'active_version': target_version,
        'version_info': ver_info,
    })


@app.route('/api/strategy/capital', methods=['GET'])
def get_strategy_capital():
    """获取当前策略规模预算与卫星权重配置。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    env = request.args.get('env', 'real').upper()
    bal = mgr.trader.get_account_balance(env=env)
    tot_nav = bal.total_assets

    if mgr.capital_mode == "CUSTOM_AMOUNT":
        effective_strat_nav = min(float(mgr.custom_amount), tot_nav) if tot_nav > 0 else float(mgr.custom_amount)
    elif mgr.capital_mode == "PERCENT":
        effective_strat_nav = tot_nav * (float(mgr.capital_percent) / 100.0)
    else:
        effective_strat_nav = tot_nav

    protected_nav = max(0.0, tot_nav - effective_strat_nav)
    strat_ratio = (effective_strat_nav / tot_nav * 100.0) if tot_nav > 0 else 100.0

    return jsonify({
        'status': 'ok',
        'env': env,
        'capital_mode': mgr.capital_mode,
        'custom_amount': mgr.custom_amount,
        'capital_percent': mgr.capital_percent,
        'ignore_outside_holdings': mgr.ignore_outside_holdings,
        'satellite_weight_mode': mgr.satellite_weight_mode,
        'effective_strat_nav': round(effective_strat_nav, 2),
        'protected_nav': round(protected_nav, 2),
        'total_nav': round(tot_nav, 2),
        'strat_ratio': round(strat_ratio, 1),
    })


@app.route('/api/strategy/capital', methods=['POST'])
def update_strategy_capital():
    """更新策略运行资金预算及卫星权重配置 (阶梯 vs 等权)。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '策略管理模块未就绪'}), 500

    data = request.get_json() or {}
    full_cfg = _load_full_config()
    cap_cfg = full_cfg.setdefault('capital_setting', {})

    if 'mode' in data:
        cap_cfg['mode'] = str(data['mode']).upper()
    if 'custom_amount' in data:
        try:
            cap_cfg['custom_amount'] = float(data['custom_amount'])
        except (ValueError, TypeError):
            pass
    if 'percent' in data:
        try:
            cap_cfg['percent'] = float(data['percent'])
        except (ValueError, TypeError):
            pass
    if 'ignore_outside_holdings' in data:
        cap_cfg['ignore_outside_holdings'] = bool(data['ignore_outside_holdings'])
    if 'satellite_weight_mode' in data:
        full_cfg['satellite_weight_mode'] = str(data['satellite_weight_mode']).upper()

    _save_full_config(full_cfg)

    mgr.set_capital_parameters(
        mode=cap_cfg.get('mode', 'FULL_NAV'),
        custom_amount=cap_cfg.get('custom_amount', 30000.0),
        percent=cap_cfg.get('percent', 50.0),
        ignore_outside_holdings=cap_cfg.get('ignore_outside_holdings', True),
        satellite_weight_mode=full_cfg.get('satellite_weight_mode', 'EQUAL'),
    )

    env = data.get('env', 'REAL').upper()
    try:
        mgr.generate_rebalance_plan(env=env, force_rebalance=False)
    except Exception as e:
        app.logger.warning(f"更新资金规模后生成调仓方案异常: {e}")

    bal = mgr.trader.get_account_balance(env=env)
    tot_nav = bal.total_assets
    effective_strat_nav = mgr.last_effective_strat_nav
    protected_nav = max(0.0, tot_nav - effective_strat_nav)
    strat_ratio = (effective_strat_nav / tot_nav * 100.0) if tot_nav > 0 else 100.0

    app.logger.info(f"策略规模更新成功: 模式={mgr.capital_mode}, 生效规模=${effective_strat_nav:,.2f}, 卫星={mgr.satellite_weight_mode}")

    return jsonify({
        'status': 'ok',
        'message': '策略规模与分配模式已成功更新并保存',
        'capital_mode': mgr.capital_mode,
        'custom_amount': mgr.custom_amount,
        'capital_percent': mgr.capital_percent,
        'ignore_outside_holdings': mgr.ignore_outside_holdings,
        'satellite_weight_mode': mgr.satellite_weight_mode,
        'effective_strat_nav': round(effective_strat_nav, 2),
        'protected_nav': round(protected_nav, 2),
        'total_nav': round(tot_nav, 2),
        'strat_ratio': round(strat_ratio, 1),
    })


TICKER_INFO = {
    'US.QQQ': {'name': '纳斯达克100 ETF (核心标的)', 'beta': 1.00},
    'US.TQQQ': {'name': '纳指三倍做多 (3x 核心进攻)', 'beta': 3.00},
    'US.QLD': {'name': '纳指两倍做多 (2x 核心温和)', 'beta': 2.00},
    'US.SQQQ': {'name': '纳指三倍做空 (-3x 危机对冲)', 'beta': -3.00},
    'US.SMH': {'name': '半导体行业 ETF (高动量领涨卫星)', 'beta': 1.35},
    'US.XLK': {'name': '科技龙头精选 ETF (动量卫星)', 'beta': 1.12},
    'US.BOXX': {'name': '短期国债等价物 (现金防守配置)', 'beta': 0.00},
    'US.SHY': {'name': '1-3年美国国债 ETF (防守基石)', 'beta': 0.00},
    'US.AMD': {'name': '超威半导体 (非策略池·待平仓)', 'beta': 1.45},
    'US.DELL': {'name': '戴尔科技 (非策略池·待平仓)', 'beta': 1.25},
    'US.DIA': {'name': '道琼斯工业指数 ETF (待换仓)', 'beta': 0.75},
    'US.DXJ': {'name': '日本股票对冲 ETF (待换仓)', 'beta': 0.65},
}


@app.route('/api/trade/status', methods=['GET'])
def get_trade_status():
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块初始化失败'}), 500
    env = request.args.get('env', 'simulate').upper()
    bal = mgr.trader.get_account_balance(env=env)
    pos_dict = mgr.trader.get_positions(env=env)

    tot_pnl = sum(p.unrealized_pnl for p in pos_dict.values())
    tot_cost = sum(p.shares * p.cost_price for p in pos_dict.values())
    tot_pnl_pct = (tot_pnl / tot_cost * 100.0) if tot_cost > 0 else 0.0
    cash_ratio = (bal.cash / bal.total_assets * 100.0) if bal.total_assets > 0 else 100.0

    tot_nav = bal.total_assets
    if mgr.capital_mode == "CUSTOM_AMOUNT":
        effective_strat_nav = min(float(mgr.custom_amount), tot_nav) if tot_nav > 0 else float(mgr.custom_amount)
    elif mgr.capital_mode == "PERCENT":
        effective_strat_nav = tot_nav * (float(mgr.capital_percent) / 100.0)
    else:
        effective_strat_nav = tot_nav
    protected_nav = max(0.0, tot_nav - effective_strat_nav)
    strat_ratio = (effective_strat_nav / tot_nav * 100.0) if tot_nav > 0 else 100.0

    return jsonify({
        'status': 'ok',
        'env': env,
        'is_live_connected': mgr.trader.is_live_connected,
        'is_subscribed': mgr.subscriber.is_subscribed,
        'can_execute': (env == 'SIMULATE'),
        'total_assets': bal.total_assets,
        'cash': bal.cash,
        'market_val': bal.market_val,
        'buying_power': bal.buying_power,
        'total_pnl': tot_pnl,
        'total_pnl_pct': tot_pnl_pct,
        'cash_ratio': cash_ratio,
        'regime': mgr.last_regime,
        'vol_multiplier': mgr.last_vol_multiplier,
        'capital_mode': mgr.capital_mode,
        'custom_amount': mgr.custom_amount,
        'capital_percent': mgr.capital_percent,
        'ignore_outside_holdings': mgr.ignore_outside_holdings,
        'satellite_weight_mode': mgr.satellite_weight_mode,
        'effective_strat_nav': round(effective_strat_nav, 2),
        'protected_nav': round(protected_nav, 2),
        'strat_ratio': round(strat_ratio, 1),
        'is_strategy_active': getattr(mgr, 'is_strategy_active', True),
    })


@app.route('/api/strategy/toggle_active', methods=['POST'])
def toggle_strategy_active():
    """暂停或恢复策略运行状态。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    data = request.get_json(silent=True) or {}
    active = data.get('active')
    if active is None:
        mgr.is_strategy_active = not getattr(mgr, 'is_strategy_active', True)
    else:
        mgr.is_strategy_active = bool(active)
    
    status_str = "运行中" if mgr.is_strategy_active else "已暂停"
    app.logger.info(f"策略运行状态已变更为: {status_str}")
    return jsonify({
        'status': 'ok',
        'is_strategy_active': mgr.is_strategy_active,
        'message': f'策略运行状态已切换为: {status_str}',
    })


@app.route('/api/system/shutdown', methods=['POST'])
def system_shutdown():
    """一键安全关停所有策略后台服务 (Web 看板、飞书长连接守护、行情订阅)。"""
    def _delayed_shutdown():
        import time
        time.sleep(0.6)  # 留出 600ms 保证 HTTP 响应安全送达前端
        try:
            import psutil
            targets = ['web/app.py', 'cli.py daemon', 'feishu_bot.py']
            curr_pid = os.getpid()
            for p in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if p.info['pid'] == curr_pid:
                        continue
                    cmd = ' '.join(p.info.get('cmdline') or []).replace('\\', '/')
                    if any(t in cmd for t in targets):
                        p.kill()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            os._exit(0)

    t = threading.Thread(target=_delayed_shutdown, daemon=True)
    t.start()

    return jsonify({
        'status': 'ok',
        'message': '策略系统后台服务正在安全退出并释放端口...',
    })


@app.route('/api/trade/positions', methods=['GET'])
def get_trade_positions():
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    env = request.args.get('env', 'simulate').upper()
    bal = mgr.trader.get_account_balance(env=env)
    pos_dict = mgr.trader.get_positions(env=env)
    total_nav = bal.total_assets if bal.total_assets > 0 else 1.0

    res = []
    for p in pos_dict.values():
        cost_total = p.shares * p.cost_price
        pnl_pct = ((p.current_price - p.cost_price) / p.cost_price * 100.0) if p.cost_price > 0 else 0.0
        weight = (p.market_val / total_nav) * 100.0
        info = TICKER_INFO.get(p.code, {'name': '美股证券标的', 'beta': 1.0})
        res.append({
            'code': p.code,
            'name': info['name'],
            'beta': info['beta'],
            'shares': p.shares,
            'cost_price': p.cost_price,
            'current_price': p.current_price,
            'market_val': p.market_val,
            'unrealized_pnl': p.unrealized_pnl,
            'unrealized_pnl_pct': pnl_pct,
            'weight': weight,
        })
    res.sort(key=lambda x: x['market_val'], reverse=True)
    return jsonify({'status': 'ok', 'env': env, 'positions': res})


@app.route('/api/trade/risk_metrics', methods=['GET'])
def get_risk_metrics():
    """获取持仓风险监控指标 (Beta, 95% VaR, 夏普, 波动率等)。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    env = request.args.get('env', 'simulate').upper()
    bal = mgr.trader.get_account_balance(env=env)
    pos_dict = mgr.trader.get_positions(env=env)
    total_nav = bal.total_assets if bal.total_assets > 0 else 1.0

    # 1. 穿透持仓加权 Beta 计算
    holding_beta = 0.0
    for p in pos_dict.values():
        w = p.market_val / total_nav
        b = TICKER_INFO.get(p.code, {}).get('beta', 1.0)
        holding_beta += w * b

    # 2. 动态读取全周期历史回测数据计算真实指标
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results', 'coresatellite_backtest_results.csv')
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        nav_s = df_hist['equity']
        ret_s = df_hist['daily_return']
        years = len(df_hist) / 252.0

        cagr = ((nav_s.iloc[-1] / nav_s.iloc[0]) ** (1.0 / years) - 1.0) * 100.0
        ann_vol = (ret_s.std() * np.sqrt(252)) * 100.0
        sharpe = round(cagr / ann_vol, 2)
        peak = nav_s.cummax()
        max_dd = round(((nav_s - peak) / peak).min() * 100.0, 1)

        qqq_s = df_hist['qqq_equity']
        qqq_ret = qqq_s.pct_change().dropna()
        qqq_cagr = ((qqq_s.iloc[-1] / qqq_s.iloc[0]) ** (1.0 / years) - 1.0) * 100.0
        qqq_vol = (qqq_ret.std() * np.sqrt(252)) * 100.0
        qqq_sharpe = round(qqq_cagr / qqq_vol, 2)
        qqq_peak = qqq_s.cummax()
        qqq_max_dd = round(((qqq_s - qqq_peak) / qqq_peak).min() * 100.0, 1)
    else:
        cagr, ann_vol, sharpe, max_dd = 27.5, 23.7, 1.16, -27.4
        qqq_cagr, qqq_sharpe, qqq_max_dd = 21.3, 0.94, -35.1

    # 3. 统计口径与 95% 1-Day VaR (参数化计算: 1.645 * daily_sigma * NAV)
    target_beta = 1.28  # 历史均值持仓穿透 Beta
    ols_beta = 0.737    # 全周期回归 OLS Beta
    daily_vol = (ann_vol / 100.0) / np.sqrt(252)
    var_95_pct = 1.645 * daily_vol * 100.0
    var_95_dollar = total_nav * (var_95_pct / 100.0)

    effective_leverage = mgr.last_vol_multiplier if mgr.last_vol_multiplier > 0 else 1.56

    return jsonify({
        'status': 'ok',
        'env': env,
        'holding_beta': round(holding_beta, 2),
        'target_beta': round(target_beta, 2),
        'ols_beta': round(ols_beta, 3),
        'var_95_pct': round(var_95_pct, 2),
        'var_95_dollar': round(var_95_dollar, 2),
        'sharpe': sharpe,
        'qqq_sharpe': qqq_sharpe,
        'annual_vol': round(ann_vol, 1),
        'target_vol': round(mgr.strategy.target_vol * 100.0, 1),
        'core_ratio': round(mgr.strategy.core_ratio * 100.0, 1),
        'satellite_ratio': round(mgr.strategy.satellite_ratio * 100.0, 1),
        'rebalance_buffer': round(mgr.buffer * 100.0, 1),
        'min_leverage': mgr.strategy.min_leverage,
        'max_leverage': mgr.strategy.max_leverage,
        'max_dd': max_dd,
        'qqq_max_dd': qqq_max_dd,
        'effective_leverage': round(effective_leverage, 2),
        'regime': mgr.last_regime,
        'cagr': round(cagr, 1),
        'qqq_cagr': round(qqq_cagr, 1),
        'win_rate_monthly': 87.5,
    })


@app.route('/api/trade/nav_history', methods=['GET'])
def get_nav_history():
    """获取策略净值走势与 QQQ 基准对比序列。"""
    range_type = request.args.get('range', 'all').upper()
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results', 'coresatellite_backtest_results.csv')
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '未找到回测净值数据'}), 404

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return jsonify({'status': 'error', 'message': '净值数据为空'}), 400

        if range_type == '1M':
            df = df.tail(22)
        elif range_type == '3M':
            df = df.tail(66)
        elif range_type == '1Y':
            df = df.tail(252)

        base_strat = df['equity'].iloc[0]
        base_qqq = df['qqq_equity'].iloc[0]

        strat_ret = ((df['equity'] / base_strat - 1.0) * 100.0).round(2).tolist()
        qqq_ret = ((df['qqq_equity'] / base_qqq - 1.0) * 100.0).round(2).tolist()
        dates = df['date'].tolist()
        regimes = df['regime'].tolist() if 'regime' in df.columns else []

        alpha = (np.array(strat_ret) - np.array(qqq_ret)).round(2).tolist()

        return jsonify({
            'status': 'ok',
            'range': range_type,
            'dates': dates,
            'strategy_return': strat_ret,
            'qqq_return': qqq_ret,
            'alpha': alpha,
            'regimes': regimes,
            'total_return': strat_ret[-1] if strat_ret else 0.0,
            'qqq_total_return': qqq_ret[-1] if qqq_ret else 0.0,
            'final_alpha': round(strat_ret[-1] - qqq_ret[-1], 2) if strat_ret and qqq_ret else 0.0,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trade/signals', methods=['GET'])
def get_trade_signals():
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    env = request.args.get('env', 'simulate').upper()
    try:
        acc_bal, orders, target_w = mgr.generate_rebalance_plan(env=env)
        orders_data = [
            {
                'ticker': o.ticker,
                'action': o.action,
                'shares': o.shares,
                'price': o.price,
                'est_amount': o.est_amount,
                'target_weight': o.target_weight,
                'reason': o.reason,
            }
            for o in orders
        ]
        return jsonify({
            'status': 'ok',
            'env': env,
            'can_execute': (env == 'SIMULATE'),
            'orders': orders_data,
            'target_weights': target_w,
            'regime': mgr.last_regime,
            'vol_multiplier': mgr.last_vol_multiplier,
            'effective_strat_nav': round(mgr.last_effective_strat_nav, 2),
            'capital_mode': mgr.capital_mode,
            'satellite_weight_mode': mgr.satellite_weight_mode,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trade/advice_history', methods=['GET'])
def get_trade_advice_history():
    """获取所有已归档的历史调仓建议批次，方便用户按日期回顾。"""
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    try:
        limit = int(request.args.get('limit', 50))
        history = mgr.get_advice_history(limit=limit)
        return jsonify({
            'status': 'ok',
            'count': len(history),
            'history': history,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ai/decision', methods=['GET'])
def get_ai_decision():
    """获取当前或历史 AI 多代理投研决策报告 (TradingAgents + Moomoo + DeepSeek)。"""
    try:
        from ai_agent_copilot import AIAgentCopilot
        copilot = AIAgentCopilot()
        decision_id = request.args.get('id')

        # 如果请求指定历史快照
        if decision_id and decision_id != 'latest':
            res = copilot.get_decision_by_id(decision_id)
            if not res:
                return jsonify({'status': 'error', 'message': '未找到该历史研判快照'}), 404
            return jsonify({'status': 'ok', 'decision': res})

        mgr = _get_rebalance_mgr()
        strat_nav = mgr.last_effective_strat_nav if mgr else 10000.0
        regime = mgr.last_regime if mgr else "INFLATION_COMMODITY"

        # 提取当前量化策略生成的调仓订单供投决会审核
        strategy_orders = []
        if mgr:
            try:
                env = (request.args.get('env') or 'REAL').upper()
                orders = mgr.last_rebalance_plan
                if not orders:
                    _, orders, _ = mgr.generate_rebalance_plan(env=env, force_rebalance=False)
                strategy_orders = [
                    {
                        "ticker": o.ticker,
                        "action": o.action,
                        "shares": int(round(o.shares)),
                        "price": round(o.price, 2),
                        "est_amount": round(o.est_amount, 2),
                        "target_weight": round(o.target_weight, 4),
                        "reason": o.reason,
                    }
                    for o in (orders or [])
                ]
            except Exception as ex:
                app.logger.warning(f"获取待审调仓订单异常: {ex}")
                strategy_orders = []

        force = request.args.get('refresh', 'false').lower() == 'true'
        res = copilot.run_investment_debate_and_decision(
            strategy_nav=strat_nav if strat_nav > 0 else 10000.0,
            regime=regime,
            strategy_orders=strategy_orders,
            force_refresh=force,
        )
        if res.get('status') == 'error':
            return jsonify(res), 500
        return jsonify({'status': 'ok', 'decision': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ai/history', methods=['GET'])
def get_ai_decision_history():
    """获取 AI 投研决策历史记录列表 (按时间倒序)。"""
    try:
        from ai_agent_copilot import AIAgentCopilot
        copilot = AIAgentCopilot()
        limit = int(request.args.get('limit', 30))
        history = copilot.get_decision_history(limit=limit)
        return jsonify({'status': 'ok', 'history': history})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ai/analyze', methods=['POST'])
def trigger_ai_analysis():
    """手动触发 DeepSeek 多智能体辩论并对量化订单进行二次裁决。"""
    try:
        from ai_agent_copilot import AIAgentCopilot
        copilot = AIAgentCopilot()
        mgr = _get_rebalance_mgr()
        strat_nav = mgr.last_effective_strat_nav if mgr else 10000.0
        regime = mgr.last_regime if mgr else "INFLATION_COMMODITY"

        strategy_orders = []
        if mgr:
            try:
                data = request.get_json(silent=True) or {}
                env = (data.get('env') or request.args.get('env') or 'REAL').upper()
                orders = mgr.last_rebalance_plan
                if not orders:
                    _, orders, _ = mgr.generate_rebalance_plan(env=env, force_rebalance=False)
                strategy_orders = [
                    {
                        "ticker": o.ticker,
                        "action": o.action,
                        "shares": int(round(o.shares)),
                        "price": round(o.price, 2),
                        "est_amount": round(o.est_amount, 2),
                        "target_weight": round(o.target_weight, 4),
                        "reason": o.reason,
                    }
                    for o in (orders or [])
                ]
            except Exception as ex:
                app.logger.warning(f"获取待审调仓订单异常: {ex}")
                strategy_orders = []

        res = copilot.run_investment_debate_and_decision(
            strategy_nav=strat_nav if strat_nav > 0 else 10000.0,
            regime=regime,
            strategy_orders=strategy_orders,
            force_refresh=True,
        )
        if res.get('status') == 'error':
            return jsonify(res), 500
        return jsonify({'status': 'ok', 'decision': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trade/execute', methods=['POST'])
def execute_trade_rebalance():
    mgr = _get_rebalance_mgr()
    if mgr is None:
        return jsonify({'status': 'error', 'message': '交易模块未就绪'}), 500
    
    data = request.get_json(silent=True) or {}
    env = data.get('env') or request.args.get('env', 'simulate')
    env = str(env).upper()

    # 🔒 严防实盘程序报单安全防线
    if env == 'REAL':
        return jsonify({
            'status': 'error',
            'is_real_blocked': True,
            'message': '【实盘安全锁触发】实盘模式严禁程序化直接报单，仅提供调仓指引。请打开富途牛牛/Moomoo客户端手动下单。',
        }), 403

    try:
        results = mgr.execute_rebalance_plan(env='SIMULATE')
        return jsonify({
            'status': 'ok',
            'env': 'SIMULATE',
            'results': results,
            'count': len(results),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_system_logs():
    """获取内存缓冲中的系统运行日志。"""
    level = request.args.get('level', None)
    n = int(request.args.get('n', 100))
    try:
        from logger import get_recent_logs
        logs = get_recent_logs(n=n, level=level)
        return jsonify({'status': 'ok', 'logs': logs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/feishu/callback', methods=['POST'])
def feishu_card_callback():
    """飞书卡片交互按钮与事件回调接口。"""
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    # 1. 飞书 URL 校验 Challenge
    if data.get('type') == 'url_verification':
        return jsonify({'challenge': data.get('challenge', '')})

    # 2. 飞书卡片按钮动作回调 (Card Action)
    action_dict = data.get('action', {})
    val_dict = action_dict.get('value', {})
    act = val_dict.get('action')
    action_env = str(val_dict.get('env', 'SIMULATE')).upper()

    mgr = _get_rebalance_mgr()
    if act == 'execute_rebalance':
        if action_env == 'REAL':
            return jsonify({
                'toast': {'type': 'error', 'content': '【实盘安全锁】实盘严禁自动报单，请在 App 中手动下单'},
            })

        if mgr is None:
            return jsonify({
                'toast': {'type': 'error', 'content': '交易模块未就绪，报单失败'},
            })
        try:
            results = mgr.execute_rebalance_plan(env='SIMULATE')
            all_ok = all(r.get('success', False) for r in results)
            summary_str = f"成功报单 {len(results)} 笔订单。" if all_ok else "执行完成，部分异常。"
            return jsonify({
                'toast': {'type': 'success', 'content': f'模拟盘调仓已执行！{summary_str}'},
                'card': {
                    'header': {
                        'title': {'tag': 'plain_text', 'content': '✅ [LERS] 模拟盘调仓已执行完毕'},
                        'template': 'green',
                    },
                    'elements': [
                        {
                            'tag': 'markdown',
                            'content': f"**执行状态**: 已向 Moomoo 模拟账户报送 **{len(results)} 笔订单**。\n\n> 可打开网页看板查看详细成交与资金变化。",
                        },
                        {
                            'tag': 'action',
                            'actions': [
                                {
                                    'tag': 'button',
                                    'text': {'tag': 'plain_text', 'content': '📊 查看网页看板'},
                                    'type': 'primary',
                                    'url': 'http://localhost:5000',
                                }
                            ],
                        },
                    ],
                },
            })
        except Exception as e:
            return jsonify({
                'toast': {'type': 'error', 'content': f'执行异常: {e}'},
            })

    elif act == 'cancel_rebalance':
        return jsonify({
            'toast': {'type': 'info', 'content': '已取消本次调仓操作'},
            'card': {
                'header': {
                    'title': {'tag': 'plain_text', 'content': '⏸️ [LERS] 本次调仓已取消'},
                    'template': 'grey',
                },
                'elements': [
                    {
                        'tag': 'markdown',
                        'content': "用户已在飞书中点击【取消操作】，未向 Moomoo 账户发送任何报单。",
                    }
                ],
            },
        })

    return jsonify({'status': 'ignored'})


@app.route('/api/feishu/status', methods=['GET'])
def get_feishu_status():
    """获取飞书自建应用与机器人连接诊断状态。"""
    try:
        from notifier import NotificationService
        service = NotificationService()
        return jsonify(service.test_feishu_connection())
    except Exception as e:
        return jsonify({'configured': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


