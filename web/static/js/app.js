const DEFAULT_CONFIG = {
  horizon: 5,
  thresholdMode: 'symmetric',
  threshold: 2,
  thresholdUp: 2,
  thresholdDown: 2,
  quantile: 0.3,
  frequency: 'quarterly',
  windowMode: 'expanding',
  windowYears: 5,
  minTrainYears: 5,
  riskOnThreshold: 0.6,
  riskOffThreshold: 0.6,
  sharesPerTrade: 100,
  costMode: 'real',
  simpleCostBps: 15,
  includeCosts: true,
  modelType: 'lightgbm',
  randomState: 42,
};

let statusPollTimer = null;
let equityChart = null;
let drawdownChart = null;
let positionChart = null;
let featureImportanceChart = null;
let confusionMatrixChart = null;
let rollingSharpeChart = null;
let currentBenchmarkData = null;

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  bindEvents();
  initCollapsibleSections();
  loadDefaultConfig();
});

function initCollapsibleSections() {
  const sectionHeaders = document.querySelectorAll('.param-section-header');
  sectionHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const section = header.closest('.param-section');
      section.classList.toggle('collapsed');
    });
  });
}

function initCharts() {
  const equityDom = document.getElementById('equityChart');
  const drawdownDom = document.getElementById('drawdownChart');
  const positionDom = document.getElementById('positionChart');
  const featureImpDom = document.getElementById('featureImportanceChart');
  const confusionMatDom = document.getElementById('confusionMatrixChart');
  const rollingSharpeDom = document.getElementById('rollingSharpeChart');

  if (equityDom && typeof echarts !== 'undefined') {
    equityChart = echarts.init(equityDom);
    equityChart.setOption({
      title: { text: '', left: 'center' },
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: [] },
      yAxis: { type: 'value' },
      series: [{ name: '净值', type: 'line', data: [] }],
    });
  }

  if (drawdownDom && typeof echarts !== 'undefined') {
    drawdownChart = echarts.init(drawdownDom);
    drawdownChart.setOption({
      title: { text: '', left: 'center' },
      tooltip: { trigger: 'axis', formatter: '{b}<br/>回撤: {c}%' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: [] },
      yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
      series: [{ name: '回撤', type: 'line', areaStyle: { color: 'rgba(239, 68, 68, 0.2)' }, lineStyle: { color: '#ef4444' }, data: [] }],
    });
  }

  if (positionDom && typeof echarts !== 'undefined') {
    positionChart = echarts.init(positionDom);
    positionChart.setOption({
      title: { text: '', left: 'center' },
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: [] },
      yAxis: { type: 'category', data: ['Risk-Off', 'Neutral', 'Risk-On'], min: 0, max: 2 },
      series: [{ name: '持仓', type: 'line', step: 'middle', data: [] }],
    });
  }

  if (featureImpDom && typeof echarts !== 'undefined') {
    featureImportanceChart = echarts.init(featureImpDom);
    featureImportanceChart.setOption({
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: '{b}: {c}'
      },
      grid: { left: '3%', right: '8%', bottom: '3%', top: '3%', containLabel: true },
      xAxis: { type: 'value' },
      yAxis: { type: 'category', data: [], axisLabel: { fontSize: 11 } },
      series: [{
        name: '重要性',
        type: 'bar',
        data: [],
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#93c5fd' },
            { offset: 1, color: '#3b82f6' }
          ])
        }
      }]
    });
  }

  if (confusionMatDom && typeof echarts !== 'undefined') {
    confusionMatrixChart = echarts.init(confusionMatDom);
    confusionMatrixChart.setOption({
      tooltip: {
        position: 'top',
        formatter: function(params) {
          return `真实: ${params.value[1]}<br/>预测: ${params.value[0]}<br/>数量: ${params.value[2]}`;
        }
      },
      grid: { left: '15%', right: '10%', bottom: '15%', top: '5%' },
      xAxis: {
        type: 'category',
        data: ['Risk-Off', 'Neutral', 'Risk-On'],
        position: 'bottom',
        axisLabel: { fontSize: 11 }
      },
      yAxis: {
        type: 'category',
        data: ['Risk-Off', 'Neutral', 'Risk-On'],
        axisLabel: { fontSize: 11 }
      },
      visualMap: {
        min: 0,
        max: 100,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: ['#eff6ff', '#3b82f6']
        }
      },
      series: [{
        name: '混淆矩阵',
        type: 'heatmap',
        data: [],
        label: {
          show: true,
          fontSize: 14,
          fontWeight: 'bold'
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.2)'
          }
        }
      }]
    });
  }

  if (rollingSharpeDom && typeof echarts !== 'undefined') {
    rollingSharpeChart = echarts.init(rollingSharpeDom);
    rollingSharpeChart.setOption({
      tooltip: { trigger: 'axis', formatter: '{b}<br/>夏普比率: {c}' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: [] },
      yAxis: { type: 'value' },
      series: [{
        name: '滚动夏普',
        type: 'line',
        data: [],
        smooth: true,
        lineStyle: { color: '#8b5cf6' },
        areaStyle: { color: 'rgba(139, 92, 246, 0.1)' },
        markLine: {
          data: [{ yAxis: 0, lineStyle: { color: '#ef4444', type: 'dashed' } }],
          symbol: 'none'
        }
      }]
    });
  }

  window.addEventListener('resize', () => {
    equityChart && equityChart.resize();
    drawdownChart && drawdownChart.resize();
    positionChart && positionChart.resize();
    featureImportanceChart && featureImportanceChart.resize();
    confusionMatrixChart && confusionMatrixChart.resize();
    rollingSharpeChart && rollingSharpeChart.resize();
  });
}

function bindEvents() {
  const thresholdMode = document.getElementById('thresholdMode');
  thresholdMode.addEventListener('change', handleThresholdModeChange);

  const windowMode = document.getElementById('windowMode');
  windowMode.addEventListener('change', handleWindowModeChange);

  const costMode = document.getElementById('costMode');
  if (costMode) {
    costMode.addEventListener('change', handleCostModeChange);
  }

  const runBtn = document.getElementById('runBtn');
  runBtn.addEventListener('click', handleRunBacktest);

  const resetBtn = document.getElementById('resetBtn');
  resetBtn.addEventListener('click', handleReset);

  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', handleChartTabChange);
  });

  const metricTabBtns = document.querySelectorAll('.metric-tab-btn');
  metricTabBtns.forEach(btn => {
    btn.addEventListener('click', handleMetricTabChange);
  });

  const benchmarkSelect = document.getElementById('benchmarkSelect');
  if (benchmarkSelect) {
    benchmarkSelect.addEventListener('change', handleBenchmarkChange);
  }

  const helpBtn = document.getElementById('helpBtn');
  if (helpBtn) {
    helpBtn.addEventListener('click', handleHelp);
  }
}

function handleHelp() {
  const helpContent = `LERS - 杠杆ETF状态切换系统

使用说明：
1. 在左侧配置参数（标签参数、Walk-Forward、组合参数、模型参数）
2. 点击"运行回测"按钮开始回测
3. 顶部状态栏显示运行进度
4. 绩效指标区域展示核心收益风险指标
5. 图表区域包含净值曲线、回撤、持仓等可视化
6. 运行日志实时显示回测进度

参数说明：
- 预测周期：预测未来N天的收益率
- 阈值模式：标签生成的阈值策略
- Walk-Forward：滚动验证的频率和窗口
- 组合参数：Risk-On/Off阈值和交易成本
- 模型参数：机器学习模型配置

提示：点击参数区块标题可折叠/展开该区块。`;
  
  alert(helpContent);
}

function handleThresholdModeChange(e) {
  const mode = e.target.value;
  const thresholdGroup = document.getElementById('thresholdGroup');
  const thresholdUpGroup = document.getElementById('thresholdUpGroup');
  const thresholdDownGroup = document.getElementById('thresholdDownGroup');
  const quantileGroup = document.getElementById('quantileGroup');

  thresholdGroup.classList.add('hidden');
  thresholdUpGroup.classList.add('hidden');
  thresholdDownGroup.classList.add('hidden');
  quantileGroup.classList.add('hidden');

  if (mode === 'symmetric') {
    thresholdGroup.classList.remove('hidden');
  } else if (mode === 'asymmetric') {
    thresholdUpGroup.classList.remove('hidden');
    thresholdDownGroup.classList.remove('hidden');
  } else if (mode === 'quantile') {
    quantileGroup.classList.remove('hidden');
  }
}

function handleWindowModeChange(e) {
  const mode = e.target.value;
  const windowYearsInput = document.getElementById('windowYears');

  if (mode === 'rolling') {
    windowYearsInput.disabled = false;
  } else {
    windowYearsInput.disabled = true;
  }
}

function handleCostModeChange(e) {
  const mode = e.target.value;
  const simpleCostGroup = document.getElementById('simpleCostGroup');

  if (mode === 'simple') {
    simpleCostGroup.classList.remove('hidden');
  } else {
    simpleCostGroup.classList.add('hidden');
  }
}

function handleChartTabChange(e) {
  const tabName = e.target.dataset.tab;
  
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  e.target.classList.add('active');

  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  const targetContent = document.getElementById(tabName + 'Charts');
  if (targetContent) {
    targetContent.classList.add('active');
  }

  setTimeout(() => {
    if (tabName === 'core') {
      equityChart && equityChart.resize();
      drawdownChart && drawdownChart.resize();
      positionChart && positionChart.resize();
    } else if (tabName === 'analysis') {
      featureImportanceChart && featureImportanceChart.resize();
      confusionMatrixChart && confusionMatrixChart.resize();
      rollingSharpeChart && rollingSharpeChart.resize();
    }
  }, 50);
}

function handleMetricTabChange(e) {
  const tabName = e.target.dataset.metricTab;
  
  document.querySelectorAll('.metric-tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  e.target.classList.add('active');

  document.querySelectorAll('.metrics-tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  const targetContent = document.getElementById(tabName + 'Metrics');
  if (targetContent) {
    targetContent.classList.add('active');
  }
}

function handleBenchmarkChange(e) {
  const benchmarkName = e.target.value;
  if (!benchmarkName || !currentBenchmarkData) {
    return;
  }
  
  const benchmarkMetrics = currentBenchmarkData[benchmarkName];
  if (benchmarkMetrics) {
    renderBenchmarkMetrics(benchmarkMetrics);
  }
}

function collectConfig() {
  const costMode = document.getElementById('costMode').value;
  const simpleCostBps = parseFloat(document.getElementById('simpleCostBps').value) || 15;

  return {
    horizon: parseFloat(document.getElementById('horizon').value),
    threshold_mode: document.getElementById('thresholdMode').value,
    threshold: parseFloat(document.getElementById('threshold').value),
    threshold_up: parseFloat(document.getElementById('thresholdUp').value),
    threshold_down: parseFloat(document.getElementById('thresholdDown').value),
    quantile: parseFloat(document.getElementById('quantile').value),
    frequency: document.getElementById('frequency').value,
    window_mode: document.getElementById('windowMode').value,
    window_years: parseFloat(document.getElementById('windowYears').value),
    min_train_years: parseFloat(document.getElementById('minTrainYears').value),
    risk_on_threshold: parseFloat(document.getElementById('riskOnThreshold').value),
    risk_off_threshold: parseFloat(document.getElementById('riskOffThreshold').value),
    shares_per_trade: parseInt(document.getElementById('sharesPerTrade').value),
    use_real_costs: costMode === 'real',
    commission_bps: simpleCostBps / 2,
    slippage_bps: simpleCostBps / 2,
    include_costs: document.getElementById('includeCosts').checked,
    model_type: document.getElementById('modelType').value,
    random_state: parseInt(document.getElementById('randomState').value),
  };
}

async function handleRunBacktest() {
  const runBtn = document.getElementById('runBtn');
  const btnText = runBtn.querySelector('.btn-text');
  
  runBtn.disabled = true;
  runBtn.classList.add('btn-loading');
  btnText.textContent = '运行中...';

  clearLog();
  appendLog('正在启动回测...');

  try {
    const config = collectConfig();
    const response = await fetch('/api/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    const data = await response.json();

    if (data.status === 'started' || response.ok) {
      updateStatus('running');
      startStatusPolling();
    } else {
      appendLog('启动失败: ' + (data.message || '未知错误'));
      updateStatus('error');
      runBtn.disabled = false;
      runBtn.classList.remove('btn-loading');
      btnText.textContent = '▶ 运行回测';
    }
  } catch (error) {
    appendLog('请求错误: ' + error.message);
    updateStatus('error');
    runBtn.disabled = false;
    runBtn.classList.remove('btn-loading');
    btnText.textContent = '▶ 运行回测';
  }
}

function handleReset() {
  document.getElementById('horizon').value = DEFAULT_CONFIG.horizon;
  document.getElementById('thresholdMode').value = DEFAULT_CONFIG.thresholdMode;
  document.getElementById('threshold').value = DEFAULT_CONFIG.threshold;
  document.getElementById('thresholdUp').value = DEFAULT_CONFIG.thresholdUp;
  document.getElementById('thresholdDown').value = DEFAULT_CONFIG.thresholdDown;
  document.getElementById('quantile').value = DEFAULT_CONFIG.quantile;
  document.getElementById('frequency').value = DEFAULT_CONFIG.frequency;
  document.getElementById('windowMode').value = DEFAULT_CONFIG.windowMode;
  document.getElementById('windowYears').value = DEFAULT_CONFIG.windowYears;
  document.getElementById('minTrainYears').value = DEFAULT_CONFIG.minTrainYears;
  document.getElementById('riskOnThreshold').value = DEFAULT_CONFIG.riskOnThreshold;
  document.getElementById('riskOffThreshold').value = DEFAULT_CONFIG.riskOffThreshold;
  document.getElementById('sharesPerTrade').value = DEFAULT_CONFIG.sharesPerTrade;
  document.getElementById('costMode').value = DEFAULT_CONFIG.costMode;
  document.getElementById('simpleCostBps').value = DEFAULT_CONFIG.simpleCostBps;
  document.getElementById('includeCosts').checked = DEFAULT_CONFIG.includeCosts;
  document.getElementById('modelType').value = DEFAULT_CONFIG.modelType;
  document.getElementById('randomState').value = DEFAULT_CONFIG.randomState;

  const thresholdMode = document.getElementById('thresholdMode');
  handleThresholdModeChange({ target: { value: thresholdMode.value } });

  const windowMode = document.getElementById('windowMode');
  handleWindowModeChange({ target: { value: windowMode.value } });

  const costMode = document.getElementById('costMode');
  if (costMode) {
    handleCostModeChange({ target: { value: costMode.value } });
  }

  updateStatus('idle');
  updateProgress(0);
}

async function loadDefaultConfig() {
  try {
    const response = await fetch('/api/config');
    const config = await response.json();

    if (config.label && config.label.horizon !== undefined) {
      document.getElementById('horizon').value = config.label.horizon;
    }
    if (config.label && config.label.threshold_mode) {
      document.getElementById('thresholdMode').value = config.label.threshold_mode;
      handleThresholdModeChange({ target: { value: config.label.threshold_mode } });
    }
    if (config.label && config.label.threshold !== undefined) {
      document.getElementById('threshold').value = config.label.threshold;
    }
    if (config.label && config.label.threshold_up !== undefined) {
      document.getElementById('thresholdUp').value = config.label.threshold_up;
    }
    if (config.label && config.label.threshold_down !== undefined) {
      document.getElementById('thresholdDown').value = config.label.threshold_down;
    }
    if (config.label && config.label.quantile !== undefined) {
      document.getElementById('quantile').value = config.label.quantile;
    }

    if (config.walkforward && config.walkforward.frequency) {
      document.getElementById('frequency').value = config.walkforward.frequency;
    }
    if (config.walkforward && config.walkforward.window_mode) {
      document.getElementById('windowMode').value = config.walkforward.window_mode;
      handleWindowModeChange({ target: { value: config.walkforward.window_mode } });
    }
    if (config.walkforward && config.walkforward.window_years !== undefined) {
      document.getElementById('windowYears').value = config.walkforward.window_years;
    }
    if (config.walkforward && config.walkforward.min_train_years !== undefined) {
      document.getElementById('minTrainYears').value = config.walkforward.min_train_years;
    }

    if (config.portfolio && config.portfolio.risk_on_threshold !== undefined) {
      document.getElementById('riskOnThreshold').value = config.portfolio.risk_on_threshold;
    }
    if (config.portfolio && config.portfolio.risk_off_threshold !== undefined) {
      document.getElementById('riskOffThreshold').value = config.portfolio.risk_off_threshold;
    }
    if (config.portfolio && config.portfolio.shares_per_trade !== undefined) {
      document.getElementById('sharesPerTrade').value = config.portfolio.shares_per_trade;
    }
    if (config.portfolio && config.portfolio.use_real_costs !== undefined) {
      const costModeValue = config.portfolio.use_real_costs ? 'real' : 'simple';
      document.getElementById('costMode').value = costModeValue;
      handleCostModeChange({ target: { value: costModeValue } });
    }
    if (config.portfolio && config.portfolio.include_costs !== undefined) {
      document.getElementById('includeCosts').checked = config.portfolio.include_costs;
    }

    if (config.model && config.model.model_type) {
      document.getElementById('modelType').value = config.model.model_type;
    }
    if (config.model && config.model.random_state !== undefined) {
      document.getElementById('randomState').value = config.model.random_state;
    }
  } catch (error) {
    console.warn('加载默认配置失败，使用内置默认值:', error);
  }
}

function startStatusPolling() {
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
  }

  statusPollTimer = setInterval(async () => {
    try {
      const response = await fetch('/api/backtest/status');
      const data = await response.json();

      updateProgress(data.progress);

      if (data.logs && data.logs.length > 0) {
        const currentLog = document.getElementById('logOutput').textContent;
        if (currentLog === '等待回测启动...') {
          clearLog();
        }
        data.logs.forEach(log => {
          appendLog(log);
        });
      }

      if (data.status === 'running') {
        updateStatus('running', data.step_name);
      } else if (data.status === 'completed') {
        updateStatus('completed');
        stopStatusPolling();
        loadResults();
        const runBtn = document.getElementById('runBtn');
        const btnText = runBtn.querySelector('.btn-text');
        runBtn.disabled = false;
        runBtn.classList.remove('btn-loading');
        btnText.textContent = '▶ 运行回测';
      } else if (data.status === 'error') {
        updateStatus('error');
        stopStatusPolling();
        if (data.error) {
          appendLog('\n错误: ' + data.error);
        }
        const runBtn = document.getElementById('runBtn');
        const btnText = runBtn.querySelector('.btn-text');
        runBtn.disabled = false;
        runBtn.classList.remove('btn-loading');
        btnText.textContent = '▶ 运行回测';
      }
    } catch (error) {
      console.error('状态轮询错误:', error);
    }
  }, 1000);
}

function stopStatusPolling() {
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
}

function updateStatus(status, stepName = '') {
  const statusDot = document.getElementById('btStatusDot') || document.querySelector('.status-dot');
  const statusText = document.getElementById('btStatusText') || document.querySelector('.status-text');

  if (!statusDot || !statusText) return;

  statusDot.className = 'status-dot';

  let text = '';
  switch (status) {
    case 'idle':
      statusDot.classList.add('status-idle');
      text = '回测引擎待命';
      break;
    case 'running':
      statusDot.classList.add('status-running');
      text = stepName ? `回测运行中 - ${stepName}` : '回测运行中';
      break;
    case 'completed':
      statusDot.classList.add('status-completed');
      text = '回测已完成';
      break;
    case 'error':
      statusDot.classList.add('status-error');
      text = '回测异常';
      break;
    default:
      statusDot.classList.add('status-idle');
      text = status;
  }

  statusText.textContent = text;
}

function updateProgress(progress) {
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');

  progressBar.style.width = `${progress}%`;
  progressText.textContent = `${progress}%`;
}

function appendLog(message) {
  const logOutput = document.getElementById('logOutput');
  const logContainer = document.getElementById('logContainer');

  if (logOutput.textContent === '等待回测启动...') {
    logOutput.textContent = '';
  }

  logOutput.textContent += message + '\n';
  logContainer.scrollTop = logContainer.scrollHeight;
}

function clearLog() {
  const logOutput = document.getElementById('logOutput');
  logOutput.textContent = '';
}

const DISPLAY_METRICS = [
  { key: 'total_return', label: '总收益率', format: v => `${(v * 100).toFixed(2)}%`, isPositive: v => v >= 0 },
  { key: 'annual_return', label: '年化收益率', format: v => `${(v * 100).toFixed(2)}%`, isPositive: v => v >= 0 },
  { key: 'sharpe_ratio', label: '夏普比率', format: v => v.toFixed(3) },
  { key: 'max_drawdown', label: '最大回撤', format: v => `${(v * 100).toFixed(2)}%`, isPositive: v => false },
  { key: 'annual_volatility', label: '年化波动率', format: v => `${(v * 100).toFixed(2)}%` },
  { key: 'calmar_ratio', label: '卡玛比率', format: v => v.toFixed(3) },
];

function renderMetrics(metrics) {
  const metricsGrid = document.querySelector('#strategyMetrics .metrics-grid');
  metricsGrid.innerHTML = '';

  if (!metrics || Object.keys(metrics).length === 0) {
    return;
  }

  const allMetrics = { ...metrics.portfolio_metrics, ...metrics.trading_metrics };

  DISPLAY_METRICS.forEach(metric => {
    const value = allMetrics[metric.key];
    if (value === undefined || value === null) return;

    const card = document.createElement('div');
    card.className = 'metric-card';

    const label = document.createElement('div');
    label.className = 'metric-label';
    label.textContent = metric.label;

    const valueEl = document.createElement('div');
    valueEl.className = 'metric-value';
    valueEl.textContent = metric.format(value);

    if (metric.isPositive !== undefined) {
      valueEl.classList.add(metric.isPositive(value) ? 'positive' : 'negative');
    }

    card.appendChild(label);
    card.appendChild(valueEl);
    metricsGrid.appendChild(card);
  });
}

function renderBenchmarkMetrics(benchmarkMetrics) {
  const metricsGrid = document.getElementById('benchmarkMetricsGrid');
  metricsGrid.innerHTML = '';

  if (!benchmarkMetrics || Object.keys(benchmarkMetrics).length === 0) {
    return;
  }

  DISPLAY_METRICS.forEach(metric => {
    const value = benchmarkMetrics[metric.key];
    if (value === undefined || value === null) return;

    const card = document.createElement('div');
    card.className = 'metric-card';

    const label = document.createElement('div');
    label.className = 'metric-label';
    label.textContent = metric.label;

    const valueEl = document.createElement('div');
    valueEl.className = 'metric-value';
    valueEl.textContent = metric.format(value);

    if (metric.isPositive !== undefined) {
      valueEl.classList.add(metric.isPositive(value) ? 'positive' : 'negative');
    }

    card.appendChild(label);
    card.appendChild(valueEl);
    metricsGrid.appendChild(card);
  });
}

function populateBenchmarkSelect(benchmarkMetrics) {
  const select = document.getElementById('benchmarkSelect');
  if (!select || !benchmarkMetrics) return;

  select.innerHTML = '<option value="">-- 请选择 --</option>';
  
  Object.keys(benchmarkMetrics).forEach(name => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
}

async function loadResults() {
  try {
    const response = await fetch('/api/backtest/result');
    const data = await response.json();

    if (data.status === 'error') {
      appendLog('加载结果失败: ' + data.message);
      return;
    }

    renderMetrics(data);
    updateCharts(data);

    if (data.benchmark_metrics && Object.keys(data.benchmark_metrics).length > 0) {
      currentBenchmarkData = data.benchmark_metrics;
      populateBenchmarkSelect(data.benchmark_metrics);
    }
  } catch (error) {
    appendLog('加载结果错误: ' + error.message);
  }
}

function updateCharts(data) {
  if (equityChart && data.equity_curve) {
    const dates = data.equity_curve.map(item => item.date);
    const values = data.equity_curve.map(item => item.value);

    const series = [{ name: '策略净值', type: 'line', data: values, smooth: true, lineStyle: { color: '#3b82f6' }, areaStyle: { color: 'rgba(59, 130, 246, 0.1)' } }];

    if (data.benchmark_navs) {
      Object.entries(data.benchmark_navs).forEach(([name, navData]) => {
        series.push({
          name: name,
          type: 'line',
          data: navData.map(item => item.value),
          smooth: true,
          lineStyle: { type: 'dashed' },
        });
      });
    }

    equityChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: series.map(s => s.name), top: 0 },
      xAxis: { type: 'category', boundaryGap: false, data: dates },
      yAxis: { type: 'value' },
      series: series,
    });
  }

  if (drawdownChart && data.drawdown) {
    const dates = data.drawdown.map(item => item.date);
    const values = data.drawdown.map(item => (item.value * 100).toFixed(2));

    drawdownChart.setOption({
      tooltip: { trigger: 'axis', formatter: '{b}<br/>回撤: {c}%' },
      xAxis: { type: 'category', boundaryGap: false, data: dates },
      yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
      series: [{ name: '回撤', type: 'line', data: values, areaStyle: { color: 'rgba(239, 68, 68, 0.2)' }, lineStyle: { color: '#ef4444' }, smooth: true }],
    });
  }

  if (positionChart && data.positions) {
    const dates = data.positions.map(item => item.date);
    const posMap = {
      'risk_off': 0, 'riskoff': 0, 'RiskOff': 0, 0: 0,
      'SQQQ': 0, 'sqqq': 0,
      'neutral': 1, 'Neutral': 1, 1: 1,
      'Cash': 1, 'cash': 1,
      'risk_on': 2, 'riskon': 2, 'RiskOn': 2, 2: 2,
      'TQQQ': 2, 'tqqq': 2, 'QQQ': 2, 'qqq': 2,
    };
    const values = data.positions.map(item => posMap[item.position] !== undefined ? posMap[item.position] : 1);

    positionChart.setOption({
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const posLabels = ['Risk-Off', 'Neutral', 'Risk-On'];
          const val = params[0].value;
          return `${params[0].name}<br/>持仓: ${posLabels[val] || 'Unknown'}`;
        }
      },
      xAxis: { type: 'category', boundaryGap: false, data: dates },
      yAxis: { type: 'category', data: ['Risk-Off', 'Neutral', 'Risk-On'], min: 0, max: 2 },
      series: [{
        name: '持仓',
        type: 'line',
        step: 'middle',
        data: values,
        lineStyle: { color: '#10b981', width: 2 },
        symbol: 'none',
      }],
    });
  }

  if (featureImportanceChart && data.feature_importance && data.feature_importance.length > 0) {
    const top20 = data.feature_importance.slice(0, 20).reverse();
    const features = top20.map(item => item.feature);
    const importances = top20.map(item => item.importance);

    featureImportanceChart.setOption({
      yAxis: { type: 'category', data: features, axisLabel: { fontSize: 11 } },
      series: [{
        name: '重要性',
        type: 'bar',
        data: importances,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#93c5fd' },
            { offset: 1, color: '#3b82f6' }
          ])
        }
      }]
    });
  }

  if (confusionMatrixChart && data.confusion_matrix && data.confusion_matrix.length > 0) {
    const cm = data.confusion_matrix;
    const heatmapData = [];
    let maxVal = 0;

    for (let i = 0; i < cm.length; i++) {
      for (let j = 0; j < cm[i].length; j++) {
        heatmapData.push([j, i, cm[i][j]]);
        if (cm[i][j] > maxVal) maxVal = cm[i][j];
      }
    }

    confusionMatrixChart.setOption({
      visualMap: {
        min: 0,
        max: maxVal > 0 ? maxVal : 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: ['#eff6ff', '#3b82f6']
        }
      },
      series: [{
        name: '混淆矩阵',
        type: 'heatmap',
        data: heatmapData,
        label: {
          show: true,
          fontSize: 14,
          fontWeight: 'bold'
        }
      }]
    });
  }

  if (rollingSharpeChart && data.rolling_sharpe && data.rolling_sharpe.length > 0) {
    const dates = data.rolling_sharpe.map(item => item.date);
    const values = data.rolling_sharpe.map(item => item.value !== null ? parseFloat(item.value.toFixed(3)) : null);

    rollingSharpeChart.setOption({
      xAxis: { type: 'category', boundaryGap: false, data: dates },
      yAxis: { type: 'value' },
      series: [{
        name: '滚动夏普',
        type: 'line',
        data: values,
        smooth: true,
        lineStyle: { color: '#8b5cf6' },
        areaStyle: { color: 'rgba(139, 92, 246, 0.1)' },
        markLine: {
          data: [{ yAxis: 0, lineStyle: { color: '#ef4444', type: 'dashed' } }],
          symbol: 'none'
        }
      }]
    });
  }
}

/* ========================================================
   Moomoo 模拟与实盘实时调仓看板控制器
   ======================================================== */
document.addEventListener('DOMContentLoaded', () => {
  const navBacktestBtn = document.getElementById('navBacktestBtn');
  const navLiveTradeBtn = document.getElementById('navLiveTradeBtn');
  const backtestView = document.getElementById('backtestView');
  const liveTradeView = document.getElementById('liveTradeView');

  const envSimulateBtn = document.getElementById('envSimulateBtn');
  const envRealBtn = document.getElementById('envRealBtn');
  const envModeBadge = document.getElementById('envModeBadge');
  const realSecurityBanner = document.getElementById('realSecurityBanner');

  const refreshLiveBtn = document.getElementById('refreshLiveBtn');
  const copyOrdersBtn = document.getElementById('copyOrdersBtn');
  const oneClickExecuteBtn = document.getElementById('oneClickExecuteBtn');
  const clearTerminalBtn = document.getElementById('clearTerminalBtn');
  const toggleStrategyBtn = document.getElementById('toggleStrategyBtn');
  const shutdownSystemBtn = document.getElementById('shutdownSystemBtn');
  const topShutdownBtn = document.getElementById('topShutdownBtn');
  const shutdownOverlay = document.getElementById('shutdownOverlay');

  let currentLiveEnv = 'real';
  let currentNavRange = 'ALL';
  let currentOrdersList = [];
  let logPollInterval = null;
  let liveEquityChart = null;

  // 初始化净值图表
  function initLiveEquityChart() {
    const dom = document.getElementById('liveEquityChart');
    if (!dom || typeof echarts === 'undefined') return;
    if (!liveEquityChart) {
      liveEquityChart = echarts.init(dom);
      window.addEventListener('resize', () => liveEquityChart && liveEquityChart.resize());
    }
  }

  // 1. 顶部主视图切换
  const liveStatusIndicator = document.getElementById('liveStatusIndicator');
  const backtestStatusIndicator = document.getElementById('backtestStatusIndicator');
  const btProgressContainer = document.getElementById('btProgressContainer');
  const progressText = document.getElementById('progressText');

  if (navBacktestBtn && navLiveTradeBtn) {
    navBacktestBtn.addEventListener('click', () => {
      navBacktestBtn.classList.add('active');
      navLiveTradeBtn.classList.remove('active');
      backtestView.classList.remove('hidden');
      liveTradeView.classList.add('hidden');

      const appContainer = document.getElementById('appContainer');
      if (appContainer) appContainer.classList.remove('live-mode');

      if (liveStatusIndicator) liveStatusIndicator.classList.add('hidden');
      if (backtestStatusIndicator) backtestStatusIndicator.classList.remove('hidden');
      if (btProgressContainer) btProgressContainer.classList.remove('hidden');
      if (progressText) progressText.classList.remove('hidden');

      if (logPollInterval) clearInterval(logPollInterval);

      setTimeout(() => {
        equityChart && equityChart.resize();
        drawdownChart && drawdownChart.resize();
        positionChart && positionChart.resize();
        featureImportanceChart && featureImportanceChart.resize();
        confusionMatrixChart && confusionMatrixChart.resize();
        rollingSharpeChart && rollingSharpeChart.resize();
      }, 100);
    });

    navLiveTradeBtn.addEventListener('click', () => {
      navLiveTradeBtn.classList.add('active');
      navBacktestBtn.classList.remove('active');
      backtestView.classList.add('hidden');
      liveTradeView.classList.remove('hidden');

      const appContainer = document.getElementById('appContainer');
      if (appContainer) appContainer.classList.add('live-mode');

      if (liveStatusIndicator) liveStatusIndicator.classList.remove('hidden');
      if (backtestStatusIndicator) backtestStatusIndicator.classList.add('hidden');
      if (btProgressContainer) btProgressContainer.classList.add('hidden');
      if (progressText) progressText.classList.add('hidden');

      initLiveEquityChart();
      loadLiveTradeData();
      startLiveLogPolling();
      setTimeout(() => liveEquityChart && liveEquityChart.resize(), 100);
    });
  }

  // 2. 模拟盘 vs 实盘 环境切换
  function switchLiveEnvironment(env) {
    currentLiveEnv = env;
    const isReal = (env === 'real');

    if (envSimulateBtn && envRealBtn) {
      if (isReal) {
        envRealBtn.classList.add('active');
        envSimulateBtn.classList.remove('active');
      } else {
        envSimulateBtn.classList.add('active');
        envRealBtn.classList.remove('active');
      }
    }

    if (realSecurityBanner) {
      realSecurityBanner.classList.toggle('hidden', !isReal);
    }
    if (copyOrdersBtn) {
      copyOrdersBtn.classList.toggle('hidden', !isReal);
    }

    if (envModeBadge) {
      if (isReal) {
        envModeBadge.className = 'badge badge-amber';
        envModeBadge.innerText = '🔒 美股实盘环境: 只读监控模式 (安全锁生效)';
      } else {
        envModeBadge.className = 'badge badge-purple';
        envModeBadge.innerText = '⚡ 模拟盘环境: 支持一键模拟报单';
      }
    }

    // 动态调整表格与卡片标题
    const liveNavTitle = document.getElementById('liveNavTitle');
    const posTableTitle = document.getElementById('posTableTitle');
    const ordersTableTitle = document.getElementById('ordersTableTitle');
    if (liveNavTitle) liveNavTitle.innerText = isReal ? '账户总资产 (Total NAV)' : '模拟账户总资产 (Total NAV)';
    if (posTableTitle) posTableTitle.innerText = isReal ? '📦 美股实盘账户实际持仓明细' : '📦 模拟账户实际持仓明细';
    if (ordersTableTitle) ordersTableTitle.innerText = isReal ? '📋 实盘调仓指引清单 (手动下单)' : '📋 待执行调仓交易清单 (计划订单)';

    // 安全锁控制执行按钮
    if (oneClickExecuteBtn) {
      if (isReal) {
        oneClickExecuteBtn.disabled = true;
        oneClickExecuteBtn.classList.add('btn-locked');
        oneClickExecuteBtn.title = '实盘安全锁已生效：严禁程序直接下单，请根据清单在富途牛牛/Moomoo客户端手动下单。';
        oneClickExecuteBtn.innerText = '🔒 实盘安全锁 (仅手动下单)';
      } else {
        oneClickExecuteBtn.disabled = false;
        oneClickExecuteBtn.classList.remove('btn-locked');
        oneClickExecuteBtn.title = '向 Moomoo 模拟账户报单';
        oneClickExecuteBtn.innerText = '🚀 一键执行调仓';
      }
    }

    loadLiveTradeData();
    loadCapitalSettings();
  }

  if (envSimulateBtn) {
    envSimulateBtn.addEventListener('click', () => switchLiveEnvironment('simulate'));
  }
  if (envRealBtn) {
    envRealBtn.addEventListener('click', () => switchLiveEnvironment('real'));
  }

  if (refreshLiveBtn) {
    refreshLiveBtn.addEventListener('click', () => {
      loadLiveTradeData();
    });
  }

  // 策略固化版本切换
  const strategyVersionSelect = document.getElementById('strategyVersionSelect');
  const versionTagBadge = document.getElementById('versionTagBadge');
  const versionDescText = document.getElementById('versionDescText');

  async function loadStrategyVersions() {
    try {
      const resp = await fetch('/api/strategy/versions');
      const data = await resp.json();
      if (data.status === 'ok' && data.versions) {
        if (strategyVersionSelect) {
          strategyVersionSelect.innerHTML = Object.entries(data.versions).map(([k, v]) => {
            const isSelected = (k === data.active_version) ? 'selected' : '';
            return `<option value="${k}" ${isSelected}>${v.name}</option>`;
          }).join('');
        }
        updateStrategyVersionInfo(data.active_version, data.versions[data.active_version]);
      }
    } catch (e) {
      console.error('加载策略版本配置异常:', e);
    }
  }

  function updateStrategyVersionInfo(versionKey, info) {
    if (!info) return;
    if (versionTagBadge) {
      versionTagBadge.innerText = info.tag || '已生效';
      if (versionKey === 'v1.0-production') {
        versionTagBadge.className = 'badge badge-purple';
      } else if (versionKey === 'v1.0-aggressive') {
        versionTagBadge.className = 'badge badge-amber';
      } else {
        versionTagBadge.className = 'badge badge-blue';
      }
    }
    if (versionDescText) {
      versionDescText.innerText = `${info.description} (历史夏普 ${info.sharpe} | 最大回撤 ${info.max_dd}% | CAGR ${info.cagr}%)`;
    }
  }

  if (strategyVersionSelect) {
    strategyVersionSelect.addEventListener('change', async () => {
      const chosenVer = strategyVersionSelect.value;
      try {
        const resp = await fetch('/api/strategy/switch_version', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version: chosenVer, env: currentLiveEnv })
        });
        const res = await resp.json();
        if (res.status === 'ok') {
          updateStrategyVersionInfo(chosenVer, res.version_info);
          loadLiveTradeData();
          const term = document.getElementById('liveTerminalOutput');
          if (term) {
            term.innerHTML += `<div class="terminal-line warning">[${new Date().toLocaleTimeString()}] [STRATEGY] 已成功切换实盘策略固化版本: ${res.version_info.name}</div>`;
            term.scrollTop = term.scrollHeight;
          }
        } else {
          alert('切换策略版本失败: ' + (res.message || '未知错误'));
        }
      } catch (err) {
        alert('网络请求异常: ' + err.message);
      }
    });
  }

  // 策略规模与持仓分配设置控制器
  const capitalModeSelect = document.getElementById('capitalModeSelect');
  const capitalAmountWrapper = document.getElementById('capitalAmountWrapper');
  const capitalPercentWrapper = document.getElementById('capitalPercentWrapper');
  const capitalCustomAmount = document.getElementById('capitalCustomAmount');
  const capitalPercent = document.getElementById('capitalPercent');
  const ignoreOutsideHoldings = document.getElementById('ignoreOutsideHoldings');
  const satelliteWeightModeSelect = document.getElementById('satelliteWeightModeSelect');
  const applyCapitalBtn = document.getElementById('applyCapitalBtn');

  function updateCapitalInputVisibility(mode) {
    if (mode === 'CUSTOM_AMOUNT') {
      if (capitalAmountWrapper) capitalAmountWrapper.classList.remove('hidden');
      if (capitalPercentWrapper) capitalPercentWrapper.classList.add('hidden');
    } else if (mode === 'PERCENT') {
      if (capitalAmountWrapper) capitalAmountWrapper.classList.add('hidden');
      if (capitalPercentWrapper) capitalPercentWrapper.classList.remove('hidden');
    } else {
      if (capitalAmountWrapper) capitalAmountWrapper.classList.add('hidden');
      if (capitalPercentWrapper) capitalPercentWrapper.classList.add('hidden');
    }
  }

  if (capitalModeSelect) {
    capitalModeSelect.addEventListener('change', () => {
      updateCapitalInputVisibility(capitalModeSelect.value);
    });
  }

  async function loadCapitalSettings() {
    try {
      const resp = await fetch(`/api/strategy/capital?env=${currentLiveEnv}`);
      const data = await resp.json();
      if (data.status === 'ok') {
        if (capitalModeSelect) capitalModeSelect.value = data.capital_mode;
        if (capitalCustomAmount) capitalCustomAmount.value = data.custom_amount;
        if (capitalPercent) capitalPercent.value = data.capital_percent;
        if (ignoreOutsideHoldings) ignoreOutsideHoldings.checked = !!data.ignore_outside_holdings;
        if (satelliteWeightModeSelect) satelliteWeightModeSelect.value = data.satellite_weight_mode;
        updateCapitalInputVisibility(data.capital_mode);

        const badgeStratNav = document.getElementById('badgeStratNav');
        const badgeStratRatio = document.getElementById('badgeStratRatio');
        const badgeProtectedNav = document.getElementById('badgeProtectedNav');
        if (badgeStratNav) badgeStratNav.innerText = `$${data.effective_strat_nav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (badgeStratRatio) badgeStratRatio.innerText = `${data.strat_ratio}%`;
        if (badgeProtectedNav) badgeProtectedNav.innerText = `$${data.protected_nav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      }
    } catch (e) {
      console.error('加载资金规模配置异常:', e);
    }
  }

  if (applyCapitalBtn) {
    applyCapitalBtn.addEventListener('click', async () => {
      const mode = capitalModeSelect ? capitalModeSelect.value : 'FULL_NAV';
      const custom_amount = capitalCustomAmount ? parseFloat(capitalCustomAmount.value) : 30000.0;
      const percent = capitalPercent ? parseFloat(capitalPercent.value) : 50.0;
      const ignore_outside = ignoreOutsideHoldings ? ignoreOutsideHoldings.checked : true;
      const sat_mode = satelliteWeightModeSelect ? satelliteWeightModeSelect.value : 'EQUAL';

      applyCapitalBtn.disabled = true;
      applyCapitalBtn.innerText = '保存中...';

      try {
        const resp = await fetch('/api/strategy/capital', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: mode,
            custom_amount: custom_amount,
            percent: percent,
            ignore_outside_holdings: ignore_outside,
            satellite_weight_mode: sat_mode,
            env: currentLiveEnv,
          })
        });
        const res = await resp.json();
        if (res.status === 'ok') {
          loadLiveTradeData();
          const term = document.getElementById('liveTerminalOutput');
          if (term) {
            term.innerHTML += `<div class="terminal-line warning">[${new Date().toLocaleTimeString()}] [CAPITAL] 策略规模已调整: 模式=${res.capital_mode}, 有效规模=$${res.effective_strat_nav.toLocaleString('en-US', { minimumFractionDigits: 2 })}, 卫星分配=${res.satellite_weight_mode}</div>`;
            term.scrollTop = term.scrollHeight;
          }
          alert(`✅ 策略规模设置已成功保存并重新计算调仓信号！\n当前策略有效运行资金: $${res.effective_strat_nav.toLocaleString('en-US', { minimumFractionDigits: 2 })}\n卫星持仓分配模式: ${res.satellite_weight_mode}`);
        } else {
          alert('设置失败: ' + (res.message || '未知错误'));
        }
      } catch (err) {
        alert('网络请求异常: ' + err.message);
      } finally {
        applyCapitalBtn.disabled = false;
        applyCapitalBtn.innerText = '💾 应用配置';
      }
    });
  }

  // 净值时间范围切换
  const rangePills = document.querySelectorAll('#navRangePills .range-pill-btn');
  rangePills.forEach(btn => {
    btn.addEventListener('click', () => {
      rangePills.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentNavRange = btn.dataset.range || 'ALL';
      loadLiveNavHistory(currentNavRange);
    });
  });

  // 一键复制调仓清单 (实盘模式)
  if (copyOrdersBtn) {
    copyOrdersBtn.addEventListener('click', () => {
      const ordersToCopy = (currentlyDisplayedOrders && currentlyDisplayedOrders.length > 0) ? currentlyDisplayedOrders : currentOrdersList;
      if (!ordersToCopy || ordersToCopy.length === 0) {
        alert('当前持仓在缓冲区内或无调仓交易。');
        return;
      }
      const isHist = historyAdviceSelect && historyAdviceSelect.value !== 'live';
      let copyText = `【LERS 核心-卫星策略 实盘调仓建议清单${isHist ? ' (历史回顾批次)' : ''}】\n`;
      copyText += `模式版本: ${isHist ? currentlyDisplayedTitle : '实时最新计算建议'}\n`;
      copyText += `评估状态: ${document.getElementById('riskRegimeVal')?.innerText || 'BULL_GROWTH'}\n`;
      copyText += `生成时间: ${new Date().toLocaleString()}\n`;
      copyText += `-------------------------------------------\n`;
      ordersToCopy.forEach((o, idx) => {
        const actionText = o.action === 'BUY' ? '买入' : '卖出';
        const shares = (o.shares !== undefined && o.shares !== null) ? Math.round(Number(o.shares)).toLocaleString() : '0';
        const price = (o.price !== undefined && o.price !== null) ? Number(o.price).toFixed(2) : '0.00';
        const estAmt = (o.est_amount !== undefined && o.est_amount !== null) ? Number(o.est_amount).toFixed(2) : '0.00';
        const tw = (o.target_weight !== undefined && o.target_weight !== null) ? (Number(o.target_weight) * 100).toFixed(1) : '0.0';
        copyText += `${idx + 1}. [${actionText}] ${o.ticker}  ${shares}股  参考限价: $${price}  预估金额: $${estAmt} (目标权重: ${tw}%) - ${o.reason || '-'}\n`;
      });
      copyText += `-------------------------------------------\n`;
      copyText += `🔒 实盘安全提醒：请在富途牛牛/Moomoo App 按照先卖后买顺序手动报单。`;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(copyText).then(() => {
          alert('✅ 实盘调仓清单已复制到剪贴板！\n您可以直接粘贴发送到手机或在富途牛牛 App 中对照下单。');
        }).catch(() => {
          prompt('请手动全选复制以下调仓清单：', copyText);
        });
      } else {
        prompt('请手动全选复制以下调仓清单：', copyText);
      }
    });
  }

  if (clearTerminalBtn) {
    clearTerminalBtn.addEventListener('click', () => {
      const term = document.getElementById('liveTerminalOutput');
      if (term) term.innerHTML = '<div class="terminal-line">[SYSTEM] 控制台已清屏...</div>';
    });
  }

  // 策略暂停 / 恢复状态更新与交互
  function updateStrategyActiveUI(isActive) {
    const liveStatusText = document.getElementById('liveStatusText');
    const liveStatusDot = document.querySelector('#liveStatusIndicator .status-dot');
    if (toggleStrategyBtn) {
      if (isActive) {
        toggleStrategyBtn.innerText = '⏸️ 暂停策略';
        toggleStrategyBtn.className = 'btn btn-secondary btn-sm';
        if (liveStatusText) {
          liveStatusText.innerText = '监控服务运行中';
          liveStatusText.style.color = '#10b981';
        }
        if (liveStatusDot) liveStatusDot.className = 'status-dot status-running';
      } else {
        toggleStrategyBtn.innerText = '▶️ 恢复策略';
        toggleStrategyBtn.className = 'btn btn-warning btn-sm';
        if (liveStatusText) {
          liveStatusText.innerText = '策略已暂停调度';
          liveStatusText.style.color = '#f59e0b';
        }
        if (liveStatusDot) liveStatusDot.className = 'status-dot status-idle';
      }
    }
  }

  if (toggleStrategyBtn) {
    toggleStrategyBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/strategy/toggle_active', { method: 'POST' });
        const res = await resp.json();
        if (res.status === 'ok') {
          updateStrategyActiveUI(res.is_strategy_active);
          const term = document.getElementById('liveTerminalOutput');
          if (term) {
            const statusStr = res.is_strategy_active ? '已恢复运行' : '已暂停调度';
            term.innerHTML += `<div class="terminal-line warning">[${new Date().toLocaleTimeString()}] [STRATEGY] 策略${statusStr}</div>`;
            term.scrollTop = term.scrollHeight;
          }
        }
      } catch (err) {
        alert('切换策略运行状态异常: ' + err.message);
      }
    });
  }

  // 一键安全关停所有后台服务
  async function handleSystemShutdown() {
    const ok = confirm(
      '确定要安全退出美股策略系统吗？\n\n' +
      '这将安全停止：\n' +
      '1. Web 监控看板服务 (端口 5000)\n' +
      '2. 飞书长连接监听与通知守护进程 (cli.py daemon)\n' +
      '3. 本地实时行情订阅连接\n\n' +
      '退出后彻底释放系统内存与端口资源，无残留脚本运行。\n' +
      '如需再次使用，随时双击桌面快捷方式即可重新启动。'
    );
    if (!ok) return;

    if (shutdownOverlay) {
      shutdownOverlay.classList.remove('hidden');
    }
    if (logPollInterval) {
      clearInterval(logPollInterval);
    }

    try {
      await fetch('/api/system/shutdown', { method: 'POST' });
    } catch (e) {
      // 服务退出导致连接断开属正常现象
    }
  }

  if (shutdownSystemBtn) {
    shutdownSystemBtn.addEventListener('click', handleSystemShutdown);
  }
  if (topShutdownBtn) {
    topShutdownBtn.addEventListener('click', handleSystemShutdown);
  }

  // 一键执行调仓 (仅模拟盘)
  if (oneClickExecuteBtn) {
    oneClickExecuteBtn.addEventListener('click', async () => {
      if (currentLiveEnv === 'real') {
        alert('🔒 实盘安全防御已激活：\n本系统不支持且已物理锁定实盘程序化直接报单，请在官方客户端手动下单。');
        return;
      }

      if (!confirm('确认立即向 Moomoo 模拟账户报送调仓清单吗？\n（系统将遵循卖单优先回笼资金、后买入加仓原则）')) {
        return;
      }

      oneClickExecuteBtn.disabled = true;
      oneClickExecuteBtn.innerText = '⏳ 正在报单中...';

      try {
        const resp = await fetch('/api/trade/execute?env=simulate', { method: 'POST' });
        const data = await resp.json();

        if (data.status === 'ok') {
          alert(`✅ 调仓报单完成！共提交 ${data.count} 笔订单。`);
          loadLiveTradeData();
        } else {
          alert(`❌ 报单异常: ${data.message || '未知错误'}`);
        }
      } catch (err) {
        alert(`❌ 网络请求失败: ${err.message}`);
      } finally {
        oneClickExecuteBtn.disabled = false;
        oneClickExecuteBtn.innerText = '🚀 一键执行调仓';
      }
    });
  }

  // 净值走势加载与渲染
  async function loadLiveNavHistory(range = 'ALL') {
    initLiveEquityChart();
    if (!liveEquityChart) return;

    try {
      const resp = await fetch(`/api/trade/nav_history?range=${range}`);
      const data = await resp.json();
      if (data.status !== 'ok') return;

      const totalRetDom = document.getElementById('chartTotalReturn');
      const qqqRetDom = document.getElementById('chartQqqReturn');
      const alphaDom = document.getElementById('chartAlpha');
      if (totalRetDom) totalRetDom.innerText = `${data.total_return >= 0 ? '+' : ''}${data.total_return.toFixed(1)}%`;
      if (qqqRetDom) qqqRetDom.innerText = `${data.qqq_total_return >= 0 ? '+' : ''}${data.qqq_total_return.toFixed(1)}%`;
      if (alphaDom) alphaDom.innerText = `${data.final_alpha >= 0 ? '+' : ''}${data.final_alpha.toFixed(1)}%`;

      const option = {
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params) => {
            if (!params || params.length === 0) return '';
            const idx = params[0].dataIndex;
            const date = data.dates[idx] || '';
            const regime = data.regimes[idx] || '';
            let html = `<div style="font-weight:bold; margin-bottom:4px;">${date} <span style="font-size:11px; color:#a855f7;">[${regime}]</span></div>`;
            params.forEach(p => {
              const sign = p.value >= 0 ? '+' : '';
              html += `<div style="display:flex; justify-content:space-between; gap:16px;">
                <span>${p.marker} ${p.seriesName}:</span>
                <strong>${sign}${p.value.toFixed(2)}%</strong>
              </div>`;
            });
            return html;
          }
        },
        legend: {
          data: ['核心-卫星策略 (LERS)', 'QQQ 基准', '超额收益 (Alpha)'],
          top: 0,
          textStyle: { fontSize: 12, color: '#4b5563' }
        },
        grid: {
          left: '3%',
          right: '3%',
          bottom: '10%',
          top: '12%',
          containLabel: true
        },
        dataZoom: [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', bottom: '2%', height: 18 }
        ],
        xAxis: {
          type: 'category',
          data: data.dates,
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#d1d5db' } },
          axisLabel: { color: '#6b7280', fontSize: 11 }
        },
        yAxis: {
          type: 'value',
          axisLabel: { formatter: '{value}%', color: '#6b7280', fontSize: 11 },
          splitLine: { lineStyle: { color: '#f3f4f6' } }
        },
        series: [
          {
            name: '核心-卫星策略 (LERS)',
            type: 'line',
            data: data.strategy_return,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 2.5, color: '#10b981' },
            itemStyle: { color: '#10b981' },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(16, 185, 129, 0.28)' },
                { offset: 1, color: 'rgba(16, 185, 129, 0.02)' }
              ])
            }
          },
          {
            name: 'QQQ 基准',
            type: 'line',
            data: data.qqq_return,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 2, color: '#3b82f6' },
            itemStyle: { color: '#3b82f6' }
          },
          {
            name: '超额收益 (Alpha)',
            type: 'line',
            data: data.alpha,
            smooth: true,
            showSymbol: false,
            lineStyle: { width: 1.5, type: 'dashed', color: '#8b5cf6' },
            itemStyle: { color: '#8b5cf6' }
          }
        ]
      };

      liveEquityChart.setOption(option, true);
    } catch (err) {
      console.error('加载净值曲线异常:', err);
    }
  }

  // 风险指标加载
  async function loadLiveRiskMetrics() {
    try {
      const resp = await fetch(`/api/trade/risk_metrics?env=${currentLiveEnv}`);
      const data = await resp.json();
      if (data.status !== 'ok') return;

      const riskHoldingBeta = document.getElementById('riskHoldingBeta');
      const riskOlsBeta = document.getElementById('riskOlsBeta');
      const riskVarVal = document.getElementById('riskVarVal');
      const riskVarDollar = document.getElementById('riskVarDollar');
      const riskSharpeVal = document.getElementById('riskSharpeVal');
      const riskQqqSharpe = document.getElementById('riskQqqSharpe');
      const riskVolVal = document.getElementById('riskVolVal');
      const riskTargetVol = document.getElementById('riskTargetVol');
      const riskLeverageVal = document.getElementById('riskLeverageVal');
      const riskRegimeVal = document.getElementById('riskRegimeVal');
      const riskMaxDdVal = document.getElementById('riskMaxDdVal');
      const riskQqqMaxDd = document.getElementById('riskQqqMaxDd');

      if (riskHoldingBeta) riskHoldingBeta.innerText = `${data.holding_beta}x`;
      if (riskOlsBeta) riskOlsBeta.innerText = `${data.ols_beta}x`;
      if (riskVarVal) riskVarVal.innerText = `-${data.var_95_pct}%`;
      if (riskVarDollar) riskVarDollar.innerText = `-$${data.var_95_dollar.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      if (riskSharpeVal) riskSharpeVal.innerText = `${data.sharpe}`;
      if (riskQqqSharpe) riskQqqSharpe.innerText = `${data.qqq_sharpe}`;
      if (riskVolVal) riskVolVal.innerText = `${data.annual_vol}%`;
      if (riskTargetVol) riskTargetVol.innerText = `${data.target_vol.toFixed(1)}%`;
      if (riskLeverageVal) riskLeverageVal.innerText = `${data.effective_leverage}x`;
      if (riskRegimeVal) riskRegimeVal.innerText = data.regime;
      if (riskMaxDdVal) riskMaxDdVal.innerText = `${data.max_dd}%`;
      if (riskQqqMaxDd) riskQqqMaxDd.innerText = `${data.qqq_max_dd}%`;
    } catch (err) {
      console.error('加载风险指标异常:', err);
    }
  }

  // 加载实时状态与调仓数据
  async function loadLiveTradeData() {
    try {
      const envParam = `?env=${currentLiveEnv}`;
      const isReal = (currentLiveEnv === 'real');

      // 1. 资金账户概况
      const statusResp = await fetch(`/api/trade/status${envParam}`);
      const statusData = await statusResp.json();
      if (statusData.status === 'ok') {
        const liveLastSyncTime = document.getElementById('liveLastSyncTime');
        if (liveLastSyncTime) {
          liveLastSyncTime.innerText = `最近同步: ${new Date().toLocaleTimeString('zh-CN', { hour12: false })}`;
        }

        const connDot = document.getElementById('liveConnDot');
        const connText = document.getElementById('liveConnText');
        if (statusData.is_live_connected) {
          connDot.className = 'pulse-dot online';
          connText.innerText = isReal ? 'OpenD 美股实盘 (只读连接)' : 'OpenD 美股模拟盘 (已连接)';
        } else {
          connDot.className = 'pulse-dot offline';
          connText.innerText = '内建模拟撮合引擎 (Mock)';
        }

        const liveNavVal = document.getElementById('liveNavVal');
        const liveStratNavVal = document.getElementById('liveStratNavVal');
        const liveStratNavSub = document.getElementById('liveStratNavSub');
        const liveProtectedVal = document.getElementById('liveProtectedVal');
        const liveProtectedSub = document.getElementById('liveProtectedSub');
        const liveCashVal = document.getElementById('liveCashVal');
        const liveCashSub = document.getElementById('liveCashSub');
        const liveMktVal = document.getElementById('liveMktVal');
        const liveMktSub = document.getElementById('liveMktSub');
        const liveTotalPnlVal = document.getElementById('liveTotalPnlVal');
        const liveTotalPnlSub = document.getElementById('liveTotalPnlSub');

        const effStratNav = statusData.effective_strat_nav !== undefined ? statusData.effective_strat_nav : statusData.total_assets;
        const protNav = statusData.protected_nav !== undefined ? statusData.protected_nav : 0.0;
        const stratRatio = statusData.strat_ratio !== undefined ? statusData.strat_ratio : 100.0;

        if (liveNavVal) liveNavVal.innerText = `$${statusData.total_assets.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (liveStratNavVal) liveStratNavVal.innerText = `$${effStratNav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (liveStratNavSub) liveStratNavSub.innerText = `占总资产: ${stratRatio.toFixed(1)}% (${statusData.capital_mode || 'FULL_NAV'})`;
        if (liveProtectedVal) liveProtectedVal.innerText = `$${protNav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (liveProtectedSub) liveProtectedSub.innerText = `非策略自主资产/闲置现金 (${(100.0 - stratRatio).toFixed(1)}%)`;

        const badgeStratNav = document.getElementById('badgeStratNav');
        const badgeStratRatio = document.getElementById('badgeStratRatio');
        const badgeProtectedNav = document.getElementById('badgeProtectedNav');
        if (badgeStratNav) badgeStratNav.innerText = `$${effStratNav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (badgeStratRatio) badgeStratRatio.innerText = `${stratRatio.toFixed(1)}%`;
        if (badgeProtectedNav) badgeProtectedNav.innerText = `$${protNav.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

        if (liveCashVal) liveCashVal.innerText = `$${statusData.cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        if (liveCashSub) liveCashSub.innerText = `现金占比: ${statusData.cash_ratio ? statusData.cash_ratio.toFixed(1) : '0.0'}%`;
        if (liveMktVal) liveMktVal.innerText = `$${statusData.market_val.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

        if (liveTotalPnlVal) {
          const pnl = statusData.total_pnl || 0.0;
          const pnlPct = statusData.total_pnl_pct || 0.0;
          const sign = pnl >= 0 ? '+' : '';
          liveTotalPnlVal.className = `live-card-value ${pnl >= 0 ? 'text-green' : 'text-danger'}`;
          liveTotalPnlVal.innerText = `${sign}$${pnl.toFixed(2)}`;
          if (liveTotalPnlSub) liveTotalPnlSub.innerText = `浮动盈亏比: ${sign}${pnlPct.toFixed(2)}%`;
        }

        if (statusData.is_strategy_active !== undefined) {
          updateStrategyActiveUI(statusData.is_strategy_active);
        }
      }

      // 2. 风险指标
      loadLiveRiskMetrics();

      // 3. 净值走势
      loadLiveNavHistory(currentNavRange);

      // 4. 持仓明细 (9列)
      const posResp = await fetch(`/api/trade/positions${envParam}`);
      const posData = await posResp.json();
      const posTbody = document.getElementById('positionsTableBody');
      const posCountTag = document.getElementById('posCountTag');
      const liveMktSub = document.getElementById('liveMktSub');

      if (posData.status === 'ok' && posData.positions) {
        if (posCountTag) posCountTag.innerText = `${posData.positions.length} 标的`;
        if (liveMktSub) liveMktSub.innerText = `${posData.positions.length} 只标的持有中`;

        if (posData.positions.length === 0) {
          posTbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted">当前账户无持仓 (100% 现金就绪)</td></tr>';
        } else {
          posTbody.innerHTML = posData.positions.map(p => {
            const pnlClass = p.unrealized_pnl >= 0 ? 'text-green' : 'text-danger';
            const sign = p.unrealized_pnl >= 0 ? '+' : '';
            return `
              <tr>
                <td><strong>${p.code}</strong></td>
                <td><span style="font-size:12px; color:#475569;">${p.name || '-'}</span></td>
                <td>${p.shares.toLocaleString()} 股</td>
                <td>$${p.cost_price.toFixed(2)}</td>
                <td>$${p.current_price.toFixed(2)}</td>
                <td><strong>$${p.market_val.toLocaleString('en-US', { minimumFractionDigits: 2 })}</strong></td>
                <td class="${pnlClass}"><strong>${sign}$${p.unrealized_pnl.toFixed(2)}</strong> <small>(${sign}${p.unrealized_pnl_pct.toFixed(2)}%)</small></td>
                <td><span class="badge-beta">β ${p.beta.toFixed(2)}</span></td>
                <td><strong>${p.weight.toFixed(1)}%</strong></td>
              </tr>
            `;
          }).join('');
        }
      }

      // 5. 调仓清单与目标权重
      const sigResp = await fetch(`/api/trade/signals${envParam}`);
      const sigData = await sigResp.json();
      const targetWeightsContainer = document.getElementById('targetWeightsContainer');

      if (sigData.status === 'ok') {
        currentOrdersList = sigData.orders || [];

        // 目标权重条
        if (sigData.target_weights) {
          targetWeightsContainer.innerHTML = Object.entries(sigData.target_weights)
            .sort((a, b) => b[1] - a[1])
            .map(([ticker, w]) => {
              const pct = (w * 100).toFixed(1);
              return `
                <div class="target-weight-bar-item">
                  <div class="weight-bar-label">
                    <span>${ticker}</span>
                    <span>${pct}%</span>
                  </div>
                  <div class="weight-bar-track">
                    <div class="weight-bar-fill" style="width: ${Math.min(pct * 1.5, 100)}%"></div>
                  </div>
                </div>
              `;
            }).join('');
        }

        // 若当前未在历史回顾状态，渲染实时订单
        if (!historyAdviceSelect || historyAdviceSelect.value === 'live') {
          renderOrdersTable(currentOrdersList, false);
        }
        loadAdviceHistory();
      }
    } catch (err) {
      console.error('加载实盘调仓数据异常:', err);
    }
  }

  // 建议历史与订单渲染
  let cachedAdviceHistory = [];
  const historyAdviceSelect = document.getElementById('historyAdviceSelect');
  const historyReviewBanner = document.getElementById('historyReviewBanner');
  const historyReviewText = document.getElementById('historyReviewText');
  const backToLiveAdviceBtn = document.getElementById('backToLiveAdviceBtn');

  let currentlyDisplayedOrders = [];
  let currentlyDisplayedTitle = '实时最新计算建议';

  function renderOrdersTable(orders, isHistorical = false, histTitle = '') {
    const ordersTbody = document.getElementById('ordersTableBody');
    const orderCountTag = document.getElementById('orderCountTag');
    const isReal = (currentLiveEnv === 'real');

    currentlyDisplayedOrders = orders || [];
    currentlyDisplayedTitle = isHistorical ? `历史回顾批次 [${histTitle}]` : '实时最新计算建议';

    if (!ordersTbody || !orderCountTag) return;

    if (isHistorical) {
      orderCountTag.innerText = `${currentlyDisplayedOrders.length} 笔 (历史回顾)`;
    } else {
      orderCountTag.innerText = `${currentlyDisplayedOrders.length} 笔${isReal ? '建议' : '订单'}`;
    }

    if (!orders || orders.length === 0) {
      ordersTbody.innerHTML = isHistorical 
        ? '<tr><td colspan="7" class="text-center text-muted">该历史周期无调仓交易记录</td></tr>'
        : '<tr><td colspan="7" class="text-center text-muted">✅ 当前持仓在 6% 缓冲期内，无需调仓</td></tr>';
      return;
    }

    ordersTbody.innerHTML = orders.map(o => {
      const isBuy = o.action === 'BUY';
      const badgeClass = isBuy ? 'badge-buy' : 'badge-sell';
      const actionText = isReal ? (isBuy ? '🟢 建议买入' : '🔴 建议卖出') : (isBuy ? '🟢 买入' : '🔴 卖出');
      const shares = (o.shares !== undefined && o.shares !== null) ? Math.round(Number(o.shares)).toLocaleString() : '0';
      const price = (o.price !== undefined && o.price !== null) ? Number(o.price).toFixed(2) : '0.00';
      const estAmt = (o.est_amount !== undefined && o.est_amount !== null) ? Number(o.est_amount).toLocaleString('en-US', { minimumFractionDigits: 2 }) : '0.00';
      const weight = (o.target_weight !== undefined && o.target_weight !== null) ? (Number(o.target_weight) * 100).toFixed(1) : '0.0';
      return `
        <tr>
          <td><strong>${o.ticker}</strong></td>
          <td><span class="${badgeClass}">${actionText}</span></td>
          <td>${shares} 股</td>
          <td>$${price}</td>
          <td>$${estAmt}</td>
          <td>${weight}%</td>
          <td><span class="text-muted">${o.reason || '-'}</span></td>
        </tr>
      `;
    }).join('');
  }

  async function loadAdviceHistory() {
    if (!historyAdviceSelect) return;
    try {
      const resp = await fetch(`/api/trade/advice_history?env=${currentLiveEnv}&limit=30`);
      const data = await resp.json();
      if (data.status === 'ok' && data.history) {
        cachedAdviceHistory = data.history;
        const curVal = historyAdviceSelect.value;
        let html = '<option value="live">⚡ 当前最新实时建议</option>';
        data.history.forEach(item => {
          html += `<option value="${item.id}">📅 ${item.created_at} (${item.regime} | ${item.order_count}笔调仓)</option>`;
        });
        historyAdviceSelect.innerHTML = html;
        if (curVal && curVal !== 'live' && data.history.some(h => h.id === curVal)) {
          historyAdviceSelect.value = curVal;
        } else {
          historyAdviceSelect.value = 'live';
        }
      }
    } catch (e) {
      console.error('加载建议历史列表异常:', e);
    }
  }

  if (historyAdviceSelect) {
    historyAdviceSelect.addEventListener('change', () => {
      const selectedId = historyAdviceSelect.value;
      if (selectedId === 'live') {
        if (historyReviewBanner) historyReviewBanner.classList.add('hidden');
        renderOrdersTable(currentOrdersList, false);
      } else {
        const item = cachedAdviceHistory.find(h => h.id === selectedId);
        if (item) {
          if (historyReviewBanner) {
            historyReviewBanner.classList.remove('hidden');
            if (historyReviewText) {
              historyReviewText.innerText = `🔍 正在回顾历史建议 [${item.created_at}] · 宏观状态: ${item.regime} · 策略资金: $${item.strategy_nav.toLocaleString()} · 共 ${item.order_count} 笔订单`;
            }
          }
          renderOrdersTable(item.orders || [], true, item.created_at);
        }
      }
    });
  }

  if (backToLiveAdviceBtn) {
    backToLiveAdviceBtn.addEventListener('click', () => {
      if (historyAdviceSelect) {
        historyAdviceSelect.value = 'live';
        historyAdviceSelect.dispatchEvent(new Event('change'));
      }
    });
  }

  // TradingAgents AI 决策副驾驶 (DeepSeek 投决会)
  const reanalyzeAiBtn = document.getElementById('reanalyzeAiBtn');
  const aiHistorySelect = document.getElementById('aiHistorySelect');
  const aiGeneratedTime = document.getElementById('aiGeneratedTime');
  const aiVerdictBadge = document.getElementById('aiVerdictBadge');
  const aiBuildPct = document.getElementById('aiBuildPct');
  const aiConfidence = document.getElementById('aiConfidence');
  const aiStabilityComment = document.getElementById('aiStabilityComment');
  const aiMacroSummary = document.getElementById('aiMacroSummary');
  const aiCioConclusion = document.getElementById('aiCioConclusion');
  const aiBullThesis = document.getElementById('aiBullThesis');
  const aiBearThesis = document.getElementById('aiBearThesis');
  const aiCoreGuidance = document.getElementById('aiCoreGuidance');
  const aiSatelliteGuidance = document.getElementById('aiSatelliteGuidance');
  const aiOrderReviewBadge = document.getElementById('aiOrderReviewBadge');
  const aiOrderReviewSummary = document.getElementById('aiOrderReviewSummary');
  const aiOrderReviewTableBody = document.getElementById('aiOrderReviewTableBody');

  let cachedAiHistory = [];

  function renderAiDecision(d) {
    if (!d) return;
    if (aiGeneratedTime) aiGeneratedTime.innerText = `最新研判: ${d.generated_at || '刚刚'}`;
    if (aiVerdictBadge) {
      aiVerdictBadge.innerText = d.verdict_badge || '中立观望';
      if (d.action_type === 'BUILD_AGGRESSIVE' || (d.verdict_badge && d.verdict_badge.includes('积极'))) {
        aiVerdictBadge.className = 'ai-verdict-badge badge-success';
      } else if (d.action_type === 'DEFEND_TRIM' || (d.verdict_badge && d.verdict_badge.includes('减仓'))) {
        aiVerdictBadge.className = 'ai-verdict-badge badge-danger';
      } else {
        aiVerdictBadge.className = 'ai-verdict-badge badge-warning';
      }
    }
    if (aiBuildPct) {
      const pct = d.recommended_build_pct !== undefined ? d.recommended_build_pct : 30;
      aiBuildPct.innerText = `${pct}%`;
    }
    if (aiConfidence) {
      aiConfidence.innerText = `${d.confidence_score || 80}%`;
    }
    if (aiStabilityComment) {
      aiStabilityComment.innerText = d.stability_comment || '延续上期宏观研判基调，保持策略纪律与连贯性。';
    }
    if (aiMacroSummary) aiMacroSummary.innerText = d.macro_geopolitical_summary || '暂无宏观总结';
    if (aiCioConclusion) aiCioConclusion.innerText = d.cio_conclusion || '暂无CIO执行指南';
    if (aiBullThesis) aiBullThesis.innerText = d.bull_thesis || '暂无多方论据';
    if (aiBearThesis) aiBearThesis.innerText = d.bear_thesis || '暂为空方警示';
    if (aiCoreGuidance) aiCoreGuidance.innerText = d.core_allocation_guidance || '暂无底仓指导';
    if (aiSatelliteGuidance) aiSatelliteGuidance.innerText = d.satellite_allocation_guidance || '暂无卫星指导';

    // 渲染量化订单二次裁决看板
    if (aiOrderReviewSummary) {
      aiOrderReviewSummary.innerText = d.order_review_summary || '量化调仓计划二次风控裁决完毕';
    }
    if (aiOrderReviewTableBody) {
      if (d.order_reviews && d.order_reviews.length > 0) {
        if (aiOrderReviewBadge) aiOrderReviewBadge.innerText = `已审核 ${d.order_reviews.length} 笔订单`;
        aiOrderReviewTableBody.innerHTML = d.order_reviews.map(r => {
          let badgeCls = 'badge-review-approved';
          let statusText = '🟢 批准执行 (APPROVED)';
          const st = (r.review_status || '').toUpperCase();
          if (st.includes('TRIM')) {
            badgeCls = 'badge-review-trimmed';
            statusText = '🟡 建议缩减 (TRIMMED)';
          } else if (st.includes('REJECT')) {
            badgeCls = 'badge-review-rejected';
            statusText = '🔴 建议暂缓 (REJECTED)';
          } else if (st.includes('HEDGE')) {
            badgeCls = 'badge-review-hedged';
            statusText = '🟣 替换对冲 (HEDGED)';
          }

          const origAct = (r.strategy_action || '').toUpperCase() === 'BUY' ? '买入' : '卖出';
          const origShares = (r.strategy_shares !== undefined && r.strategy_shares !== null) ? Math.round(Number(r.strategy_shares)).toLocaleString() : '-';
          const adjShares = (r.adjusted_shares !== undefined && r.adjusted_shares !== null) ? Math.round(Number(r.adjusted_shares)).toLocaleString() : origShares;

          return `
            <tr>
              <td><strong>${r.ticker}</strong></td>
              <td><span class="badge ${r.strategy_action === 'BUY' ? 'badge-buy' : 'badge-sell'}">${origAct} ${origShares} 股</span></td>
              <td><span class="${badgeCls}">${statusText}</span></td>
              <td><strong>建议 ${origAct} ${adjShares} 股</strong></td>
              <td><span style="font-size: 11.5px; color: #334155; line-height: 1.4;">${r.audit_reason || '-'}</span></td>
            </tr>
          `;
        }).join('');
      } else {
        if (aiOrderReviewBadge) aiOrderReviewBadge.innerText = '0 笔订单待审';
        aiOrderReviewTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">✅ 当前持仓在 6% 缓冲期内，量化算法未触发调仓，投决会建议维持既有底仓与现金配置</td></tr>';
      }
    }
  }

  async function loadAiHistory() {
    if (!aiHistorySelect) return;
    try {
      const resp = await fetch('/api/ai/history?limit=30');
      const data = await resp.json();
      if (data.status === 'ok' && data.history) {
        cachedAiHistory = data.history;
        const curVal = aiHistorySelect.value;
        let html = '<option value="latest">⚡ 当前最新实时研判</option>';
        data.history.forEach(h => {
          html += `<option value="${h.id}">📅 ${h.generated_at} (${h.verdict_badge} | 建仓${h.recommended_build_pct}%)</option>`;
        });
        aiHistorySelect.innerHTML = html;
        if (curVal && curVal !== 'latest' && data.history.some(h => h.id === curVal)) {
          aiHistorySelect.value = curVal;
        } else {
          aiHistorySelect.value = 'latest';
        }
      }
    } catch (e) {
      console.error('加载AI研判历史异常:', e);
    }
  }

  if (aiHistorySelect) {
    aiHistorySelect.addEventListener('change', async () => {
      const selectedId = aiHistorySelect.value;
      if (selectedId === 'latest') {
        loadAiDecision(false);
      } else {
        try {
          const resp = await fetch(`/api/ai/decision?id=${selectedId}&env=${currentLiveEnv}`);
          const data = await resp.json();
          if (data.status === 'ok' && data.decision) {
            renderAiDecision(data.decision);
          }
        } catch (e) {
          alert('加载历史研判失败: ' + e.message);
        }
      }
    });
  }

  async function loadAiDecision(force = false) {
    if (force && reanalyzeAiBtn) {
      reanalyzeAiBtn.disabled = true;
      reanalyzeAiBtn.innerText = '🧠 投决会研判中 (约5-10秒)...';
    }
    try {
      let resp;
      if (force) {
        resp = await fetch('/api/ai/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ env: currentLiveEnv })
        });
      } else {
        resp = await fetch(`/api/ai/decision?env=${currentLiveEnv}`);
      }
      const data = await resp.json();
      const decision = data.decision || data;
      if (data.status === 'ok' && decision) {
        renderAiDecision(decision);
        loadAiHistory();
      } else if (data.status === 'no_analysis') {
        if (aiMacroSummary) aiMacroSummary.innerText = '尚未生成AI研判，点击右上角【重新运行投决会研判】即可启动分析。';
      } else {
        if (aiMacroSummary) aiMacroSummary.innerText = `研判提示: ${data.message || '请检查API Key配置'}`;
      }
    } catch (e) {
      console.error('加载AI决策副驾驶异常:', e);
      if (aiMacroSummary) aiMacroSummary.innerText = '无法连接AI决策副驾驶服务: ' + e.message;
    } finally {
      if (reanalyzeAiBtn) {
        reanalyzeAiBtn.disabled = false;
        reanalyzeAiBtn.innerText = '🧠 重新运行投决会研判';
      }
    }
  }

  if (reanalyzeAiBtn) {
    reanalyzeAiBtn.addEventListener('click', () => loadAiDecision(true));
  }

  // 6. 实时日志轮询
  function startLiveLogPolling() {
    if (logPollInterval) clearInterval(logPollInterval);

    const term = document.getElementById('liveTerminalOutput');
    const autoScroll = document.getElementById('autoScrollLog');

    const fetchLogs = async () => {
      try {
        const resp = await fetch('/api/logs?n=60');
        const data = await resp.json();
        if (data.status === 'ok' && data.logs && data.logs.length > 0) {
          term.innerHTML = data.logs.map(log => {
            let cls = '';
            if (log.level === 'ERROR') cls = 'error';
            else if (log.level === 'WARNING') cls = 'warning';
            else cls = 'info';
            return `<div class="terminal-line ${cls}">[${log.time}] [${log.level}] ${log.message}</div>`;
          }).join('');

          if (autoScroll && autoScroll.checked) {
            term.scrollTop = term.scrollHeight;
          }
        }
      } catch (e) {
        // silent fail
      }
    };

    fetchLogs();
    logPollInterval = setInterval(fetchLogs, 3000);
  }

  // 初始进入页面时默认激活美股实盘监控
  loadStrategyVersions();
  loadCapitalSettings();
  initLiveEquityChart();
  loadLiveTradeData();
  loadAdviceHistory();
  loadAiHistory();
  loadAiDecision(false);
  startLiveLogPolling();
});


