# -*- coding: utf-8 -*-
"""LERS 量化系统标准发布版打包脚本 (含全量隐私脱敏与规范化结构校验)。"""

import os
import zipfile
import datetime
import shutil

root_dir = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(root_dir, 'dist')
os.makedirs(dist_dir, exist_ok=True)

version = '1.0.1'
timestamp = datetime.datetime.now().strftime('%Y%m%d')
zip_filename = f'LERS_Trading_System_v{version}_{timestamp}.zip'
zip_filepath = os.path.join(dist_dir, zip_filename)
zip_root_alias = os.path.join(root_dir, 'LERS_Trading_System_latest.zip')
dest1 = r'C:\Users\lishi\Desktop\LERS_Trading_System_v1.0.1.zip'
dest2 = r'C:\Users\lishi\OneDrive\桌面\LERS_Trading_System_v1.0.1.zip'

exclude_dirs = {'.git', '.pytest_cache', '.trae', '__pycache__', 'dist', '.idea', '.vscode'}
exclude_exts = {'.pyc', '.pyo', '.pyd', '.log'}

# 读取脱敏模板作为 zip 包内的 config.json
example_cfg_path = os.path.join(root_dir, 'config', 'config.example.json')
with open(example_cfg_path, 'r', encoding='utf-8') as f:
    sanitized_config_str = f.read()

count = 0
total_uncompressed = 0

print(f"[PACKAGE] 开始构建 LERS v{version} 脱敏发布包...")

with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith('.')]
        rel_dir = os.path.relpath(dirpath, root_dir)

        # 调仓历史、AI诊断、建议历史与日志目录只保留 .gitkeep
        if rel_dir in [
            os.path.join('data', 'trading_records'), 'data/trading_records',
            os.path.join('data', 'advice_history'), 'data/advice_history',
            os.path.join('data', 'ai_analysis'), 'data/ai_analysis',
            'logs',
        ]:
            arcname = os.path.join('LERS_Trading_System', rel_dir, '.gitkeep')
            zipf.writestr(arcname, '')
            count += 1
            continue

        for f in filenames:
            if f.endswith(tuple(exclude_exts)) and f != '.gitkeep':
                continue
            if f.endswith('.zip') or f.startswith('test_') or f.startswith('verify_') or f == 'package_release.py':
                continue

            abs_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(abs_path, root_dir)
            arcname = os.path.join('LERS_Trading_System', rel_path)

            # 注入脱敏 config.json
            if rel_path in [os.path.join('config', 'config.json'), 'config/config.json']:
                zipf.writestr(arcname, sanitized_config_str)
                count += 1
                total_uncompressed += len(sanitized_config_str)
                continue

            # 注入脱敏 apikey.txt (保护用户的真实 API Key 不被泄露)
            if rel_path in [os.path.join('config', 'apikey.txt'), 'config/apikey.txt']:
                sanitized_apikey = "# 请在此填入您的 DeepSeek API Key (例如: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\n"
                zipf.writestr(arcname, sanitized_apikey)
                count += 1
                total_uncompressed += len(sanitized_apikey)
                continue

            zipf.write(abs_path, arcname)
            count += 1
            total_uncompressed += os.path.getsize(abs_path)

shutil.copy2(zip_filepath, zip_root_alias)
if os.path.exists(r'C:\Users\lishi\Desktop'):
    try:
        shutil.copy2(zip_filepath, dest1)
        print(f"[COPY] 成功复制到桌面: {dest1}")
    except Exception as e:
        pass

if os.path.exists(r'C:\Users\lishi\OneDrive\桌面'):
    try:
        shutil.copy2(zip_filepath, dest2)
        print(f"[COPY] 成功复制到 OneDrive 桌面: {dest2}")
    except Exception as e:
        pass

compressed_size = os.path.getsize(zip_filepath)
print(f"[DONE] 发布包打包完成！文件数: {count}, 压缩后体积: {compressed_size / (1024*1024):.2f} MB")
print(f"       目标归档: {zip_filepath}")
