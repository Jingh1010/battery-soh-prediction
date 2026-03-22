"""
蓄电池健康状态(SOH)预测系统
基于Streamlit的Web应用
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="蓄电池SOH预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
<div class="main-header">
    <h1>🔋 蓄电池健康状态(SOH)预测系统</h1>
    <p>基于机器学习方法的电池健康状态预测与评估平台</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏
st.sidebar.header("⚙️ 系统配置")

# 生成模拟数据函数
@st.cache_data
def generate_battery_data(n_samples=1000, random_seed=42):
    """生成模拟的电池循环测试数据"""
    np.random.seed(random_seed)
    
    cycles = np.arange(1, n_samples + 1)
    
    # 真实SOH: 随着循环次数增加，SOH从100%逐渐下降
    true_soh = 100 - 0.028 * cycles + np.random.normal(0, 0.8, n_samples)
    true_soh = np.clip(true_soh, 60, 100)
    
    # 容量与SOH正相关
    capacity = 50 * (true_soh / 100) + np.random.normal(0, 0.6, n_samples)
    capacity = np.clip(capacity, 28, 50)
    
    # 电压随SOH下降略有下降
    voltage = 12.8 * (true_soh / 100) + np.random.normal(0, 0.08, n_samples)
    voltage = np.clip(voltage, 11.2, 12.9)
    
    # 电流与SOH负相关
    current = 20 + 6 * (1 - true_soh/100) + np.random.normal(0, 0.6, n_samples)
    current = np.clip(current, 18, 27)
    
    # 温度随SOH下降升高
    temperature = 25 + 4 * (1 - true_soh/100) + np.random.normal(0, 0.4, n_samples)
    temperature = np.clip(temperature, 24, 31)
    
    df = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temperature,
        'Capacity': capacity,
        'SOH': true_soh
    })
    return df

# 数据加载选项
st.sidebar.subheader("📁 数据源")
data_option = st.sidebar.radio(
    "选择数据源",
    ["使用模拟数据", "上传自定义数据"],
    help="可以选择使用系统生成的模拟数据，或上传自己的CSV文件"
)

if data_option == "上传自定义数据":
    uploaded_file = st.sidebar.file_uploader(
        "上传CSV文件（包含列：Cycle, Voltage, Current, Temperature, Capacity, SOH）",
        type=['csv']
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ 数据上传成功！")
    else:
        st.sidebar.info("请上传数据文件，或使用模拟数据")
        data = generate_battery_data(1000)
else:
    n_samples = st.sidebar.slider("样本数量", 500, 3000, 1000, 100)
    data = generate_battery_data(n_samples)
    st.sidebar.success(f"✅ 已生成{n_samples}条模拟数据")

# 显示数据预览
st.subheader("📊 数据集预览")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data.head(10), use_container_width=True)
with col2:
    st.write("**数据统计信息:**")
    st.dataframe(data.describe(), use_container_width=True)

# 特征工程函数
def add_features(df):
    """添加工程特征"""
    df_feat = df.copy()
    # 容量衰减率
    df_feat['Capacity_decay_rate'] = df_feat['Capacity'].pct_change().fillna(0)
    # 电压变化率
    df_feat['Voltage_change_rate'] = df_feat['Voltage'].pct_change().fillna(0)
    # 电压滚动均值（5周期）
    df_feat['Voltage_rolling_mean'] = df_feat['Voltage'].rolling(window=5, min_periods=1).mean()
    # 温度滚动均值
    df_feat['Temp_rolling_mean'] = df_feat['Temperature'].rolling(window=5, min_periods=1).mean()
    # 温度变化率
    df_feat['Temp_change_rate'] = df_feat['Temperature'].pct_change().fillna(0)
    # 电流变化率
    df_feat['Current_change_rate'] = df_feat['Current'].pct_change().fillna(0)
    # 容量与循环次数的比值
    df_feat['Capacity_per_cycle'] = df_feat['Capacity'] / (df_feat['Cycle'] + 1)
    return df_feat

# 数据预处理选项
st.sidebar.subheader("🔧 数据处理")
handle_outliers = st.sidebar.checkbox("处理异常值", value=True)
use_engineered = st.sidebar.checkbox("使用工程特征", value=True)

# 模型选择
st.sidebar.subheader("🤖 模型配置")
model_options = {
    "随机森林 (Random Forest)": RandomForestRegressor(random_state=42),
    "梯度提升 (Gradient Boosting)": GradientBoostingRegressor(random_state=42),
    "线性回归 (Linear Regression)": LinearRegression(),
    "支持向量机 (SVR)": SVR()
}
selected_model_name = st.sidebar.selectbox("选择机器学习模型", list(model_options.keys()))

# 超参数调优选项
do_tuning = st.sidebar.checkbox("超参数调优", value=False)

# 训练按钮
if st.sidebar.button("🚀 开始训练模型", type="primary", use_container_width=True):
    
    # 数据预处理
    with st.spinner("正在处理数据..."):
        processed_data = data.copy()
        
        # 处理缺失值
        processed_data = processed_data.dropna()
        
        # 处理异常值
        if handle_outliers:
            for col in ['Voltage', 'Current', 'Temperature', 'Capacity', 'SOH']:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data = processed_data[(processed_data[col] >= lower_bound) & 
                                                (processed_data[col] <= upper_bound)]
        
        # 添加特征
        processed_data = add_features(processed_data)
        processed_data = processed_data.dropna()
        
        # 特征选择
        base_features = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity']
        engineered_features = ['Capacity_decay_rate', 'Voltage_change_rate', 'Voltage_rolling_mean', 
                               'Temp_rolling_mean', 'Temp_change_rate', 'Current_change_rate', 'Capacity_per_cycle']
        
        if use_engineered:
            selected_features = base_features + engineered_features
        else:
            selected_features = base_features
        
        X = processed_data[selected_features]
        y = processed_data['SOH']
        
        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        st.info(f"数据划分完成 - 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    
    # 模型训练
    with st.spinner(f"正在训练 {selected_model_name} 模型..."):
        model = model_options[selected_model_name]
        
        if do_tuning and selected_model_name in ["随机森林 (Random Forest)", "梯度提升 (Gradient Boosting)"]:
            if selected_model_name == "随机森林 (Random Forest)":
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5]
                }
                grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                st.success(f"最佳参数: {grid_search.best_params_}")
            else:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1]
                }
                grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                st.success(f"最佳参数: {grid_search.best_params_}")
        else:
            model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
    
    # 显示结果
    st.subheader("📈 模型评估结果")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>训练集 R²</h3>
            <h2>{train_r2:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>验证集 R²</h3>
            <h2>{val_r2:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>测试集 R²</h3>
            <h2>{test_r2:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>测试集 MAE</h3>
            <h2>{test_mae:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 可视化
    st.subheader("📊 预测结果可视化")
    
    # 创建图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("真实值 vs 预测值", "预测误差分布", "SOH退化曲线对比", "特征重要性"),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 真实值 vs 预测值
    fig.add_trace(
        go.Scatter(x=y_test, y=y_test_pred, mode='markers',
                  marker=dict(color='steelblue', size=8, opacity=0.6),
                  name='预测点'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                  mode='lines', line=dict(color='red', dash='dash', width=2),
                  name='理想线'),
        row=1, col=1
    )
    fig.update_xaxes(title_text="真实 SOH (%)", row=1, col=1)
    fig.update_yaxes(title_text="预测 SOH (%)", row=1, col=1)
    
    # 误差分布
    errors = y_test - y_test_pred
    fig.add_trace(
        go.Histogram(x=errors, nbinsx=30, marker_color='coral', name='误差分布'),
        row=1, col=2
    )
    fig.update_xaxes(title_text="预测误差 (%)", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=1, col=2)
    
    # SOH退化曲线
    results_df = pd.DataFrame({
        'Cycle': processed_data.loc[X_test.index, 'Cycle'].values,
        '真实SOH': y_test.values,
        '预测SOH': y_test_pred
    }).sort_values('Cycle')
    
    fig.add_trace(
        go.Scatter(x=results_df['Cycle'], y=results_df['真实SOH'],
                  mode='lines', name='真实SOH', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=results_df['Cycle'], y=results_df['预测SOH'],
                  mode='lines', name='预测SOH', line=dict(color='orange', width=2, dash='dot')),
        row=2, col=1
    )
    fig.update_xaxes(title_text="循环次数", row=2, col=1)
    fig.update_yaxes(title_text="SOH (%)", row=2, col=1)
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            '特征': selected_features,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=True)
        
        fig.add_trace(
            go.Bar(x=importance_df['重要性'], y=importance_df['特征'],
                  orientation='h', marker_color='teal', name='特征重要性'),
            row=2, col=2
        )
        fig.update_xaxes(title_text="重要性", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, title_text="模型预测分析")
    st.plotly_chart(fig, use_container_width=True)
    
    # 保存模型到session
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['features'] = selected_features
    st.session_state['test_r2'] = test_r2
    st.session_state['test_mae'] = test_mae
    
    st.success("✅ 模型训练完成！现在可以使用下方预测功能。")

# 在线预测模块
st.subheader("🎯 在线预测")
st.markdown("输入电池参数，预测当前健康状态(SOH)")

col1, col2, col3 = st.columns(3)
with col1:
    pred_cycle = st.number_input("循环次数 (Cycle)", min_value=1, max_value=5000, value=500, step=10)
    pred_voltage = st.number_input("电压 (V)", min_value=10.0, max_value=15.0, value=12.5, step=0.1)
with col2:
    pred_current = st.number_input("电流 (A)", min_value=0.0, max_value=100.0, value=22.0, step=0.5)
    pred_temperature = st.number_input("温度 (℃)", min_value=-10.0, max_value=60.0, value=26.0, step=0.5)
with col3:
    pred_capacity = st.number_input("容量 (Ah)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)

if st.button("🔮 预测SOH", type="secondary", use_container_width=True):
    if 'model' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        features = st.session_state['features']
        
        # 构建输入数据
        input_dict = {
            'Cycle': pred_cycle,
            'Voltage': pred_voltage,
            'Current': pred_current,
            'Temperature': pred_temperature,
            'Capacity': pred_capacity,
            'Capacity_decay_rate': 0.0,
            'Voltage_change_rate': 0.0,
            'Voltage_rolling_mean': pred_voltage,
            'Temp_rolling_mean': pred_temperature,
            'Temp_change_rate': 0.0,
            'Current_change_rate': 0.0,
            'Capacity_per_cycle': pred_capacity / (pred_cycle + 1)
        }
        
        # 只保留模型使用的特征
        input_df = pd.DataFrame([{k: input_dict[k] for k in features}])
        input_scaled = scaler.transform(input_df)
        
        # 预测
        prediction = model.predict(input_scaled)[0]
        prediction = np.clip(prediction, 0, 100)
        
        # 显示结果
        if prediction >= 90:
            status = "🌟 优秀"
            color = "#4caf50"
            advice = "电池性能良好，继续正常使用"
        elif prediction >= 80:
            status = "✅ 良好"
            color = "#8bc34a"
            advice = "电池性能正常，定期检查即可"
        elif prediction >= 70:
            status = "⚠️ 注意"
            color = "#ff9800"
            advice = "电池性能下降，建议关注充放电情况"
        else:
            status = "🔴 需更换"
            color = "#f44336"
            advice = "电池性能严重下降，建议尽快更换"
        
        st.markdown(f"""
        <div class="prediction-result">
            <h2>预测结果</h2>
            <h1 style="font-size: 64px;">{prediction:.1f}%</h1>
            <h3>健康状态评级: {status}</h3>
            <p>{advice}</p>
            <hr>
            <small>模型测试集表现: R² = {st.session_state.get('test_r2', 0.9):.3f} | MAE = {st.session_state.get('test_mae', 2.0):.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 请先训练模型！点击左侧边栏的'开始训练模型'按钮。")

# 使用说明
with st.expander("📖 使用说明与模型解释"):
    st.markdown("""
    ### 🎯 系统功能说明
    
    1. **数据加载**: 支持模拟数据生成或上传自定义CSV文件
    2. **数据处理**: 自动处理缺失值、异常值，支持数据标准化
    3. **特征工程**: 自动提取容量衰减率、电压变化趋势、滚动均值等时序特征
    4. **模型训练**: 支持随机森林、梯度提升、线性回归、SVR等多种模型
    5. **超参数调优**: 可选网格搜索优化模型参数
    6. **在线预测**: 训练完成后可输入新参数实时预测SOH值
    
    ### 📊 SOH健康状态分级
    
    | SOH范围 | 等级 | 说明 | 建议 |
    |---------|------|------|------|
    | 90-100% | 🌟 优秀 | 电池性能良好，接近全新状态 | 正常使用 |
    | 80-90% | ✅ 良好 | 电池性能正常 | 定期检查 |
    | 70-80% | ⚠️ 注意 | 电池性能下降 | 关注充放电情况 |
    | <70% | 🔴 需更换 | 电池性能严重下降 | 建议尽快更换 |
    
    ### 🔬 特征说明
    
    | 特征 | 说明 | 与SOH的关系 |
    |------|------|-------------|
    | Cycle | 充放电循环次数 | 负相关，循环越多SOH越低 |
    | Capacity | 电池容量(Ah) | 正相关，容量下降表明老化 |
    | Voltage | 电池电压(V) | 正相关，内阻增加导致电压下降 |
    | Current | 电池电流(A) | 负相关，老化导致内阻增加 |
    | Temperature | 电池温度(℃) | 负相关，老化导致发热增加 |
    | 容量衰减率 | 容量变化速率 | 衰减越快，老化越严重 |
    | 电压变化率 | 电压变化趋势 | 波动越大，性能越不稳定 |
    
    ### 🚀 快速开始
    
    1. 选择数据源（模拟数据或上传CSV）
    2. 配置数据处理选项
    3. 选择机器学习模型
    4. 点击"开始训练模型"
    5. 查看评估结果和可视化图表
    6. 使用在线预测功能测试新数据
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>🔋 <strong>蓄电池健康状态预测系统</strong> | 基于机器学习 | 支持在线预测</p>
    <p style="color: #666; font-size: 12px;">© 2024 蓄电池SOH预测系统 | 科学预测，智能运维</p>
</div>
""", unsafe_allow_html=True)
