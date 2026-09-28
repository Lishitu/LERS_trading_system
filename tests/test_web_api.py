"""Web API unit tests to ensure endpoints return valid responses."""

import pytest
import sys
import os

# Ensure web module is findable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'<!DOCTYPE html>' in rv.data or b'<html' in rv.data


def test_api_config(client):
    rv = client.get('/api/config')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert isinstance(json_data, dict)
    assert 'data' in json_data


def test_trade_api_endpoints(client):
    # Test trade status
    rv_status = client.get('/api/trade/status')
    assert rv_status.status_code == 200
    status_data = rv_status.get_json()
    assert status_data['status'] == 'ok'
    assert 'total_assets' in status_data
    assert 'cash' in status_data

    # Test trade positions
    rv_pos = client.get('/api/trade/positions')
    assert rv_pos.status_code == 200
    pos_data = rv_pos.get_json()
    assert pos_data['status'] == 'ok'
    assert isinstance(pos_data['positions'], list)


def test_system_logs_api(client):
    rv = client.get('/api/logs?n=10')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['status'] == 'ok'
    assert isinstance(data['logs'], list)


def test_feishu_callback_url_verification(client):
    rv = client.post('/api/feishu/callback', json={
        'type': 'url_verification',
        'challenge': 'test_challenge_token_123',
    })
    assert rv.status_code == 200
    data = rv.get_json()
    assert data.get('challenge') == 'test_challenge_token_123'


def test_feishu_callback_cancel_action(client):
    rv = client.post('/api/feishu/callback', json={
        'action': {
            'value': {'action': 'cancel_rebalance'},
            'tag': 'button',
        }
    })
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'toast' in data
    assert 'card' in data
    assert '取消' in data['toast']['content']


def test_trade_api_real_mode_guard(client):
    # 1. 查询实盘状态
    rv_status = client.get('/api/trade/status?env=real')
    assert rv_status.status_code == 200
    data_status = rv_status.get_json()
    assert data_status['env'] == 'REAL'
    assert data_status['can_execute'] is False

    # 2. 尝试实盘报单 -> 必须被 403 阻断
    rv_exec = client.post('/api/trade/execute?env=real')
    assert rv_exec.status_code == 403
    data_exec = rv_exec.get_json()
    assert data_exec['is_real_blocked'] is True
    assert '实盘安全锁' in data_exec['message']

    # 3. 飞书回调实盘报单阻断
    rv_feishu = client.post('/api/feishu/callback', json={
        'action': {
            'value': {'action': 'execute_rebalance', 'env': 'REAL'},
            'tag': 'button',
        }
    })
    assert rv_feishu.status_code == 200
    data_feishu = rv_feishu.get_json()
    assert '实盘' in data_feishu['toast']['content']



