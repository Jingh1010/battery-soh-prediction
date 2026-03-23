# app.py - 极简稳定版蓄电池SOH预测系统

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 最简单的页面配置
st.set_page_config(page_title="蓄电池SOH预测", page_icon="🔋")

# 标题
st.title("🔋 蓄电池健康状态(SOH)预测系统")

# 生成数据函数
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 800
    cycles = np.arange(1, n + 1)
    soh = 100 - 0.028 * cycles + np.random.normal(0, 0.8, n)
    soh = np.clip(soh, 60, 100)
    
    data = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': 12.8 * (soh/100) + np.random.normal(0, 0.08, n),
        'Current': 20 + 6*(1-soh/100) + np.random.normal(0, 0.6, n),
        'Temperature': 25 + 4*(1-soh/100) + np.random.normal(0, 0.4, n),
        'Capacity': 50 * (soh/100) + np.random.normal(0, 0.6, n),
        'SOH': soh
    })
    
    # 裁剪到合理范围
    data['Voltage'] = data['Voltage'].clip(11.2, 12.9)
    data['Current'] = data['Current'].clip(18, 27)
    data['Temperature'] = data['Temperature'].clip(24, 31)
    data['Capacity'] = data['Capacity'].clip(28, 50)
    
    return data

# 加载数据
data = load_data()

# 显示数据
with st.expander("查看数据"):
    st.write(f"数据量: {len(data)} 条")
    st.dataframe(data.head(10))

# 侧边栏
st.sidebar.header("设置")

# 训练按钮
if st.sidebar.button("开始训练模型", use_container_width=True):
    
    with st.spinner("训练中..."):
        # 准备数据
        features = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity']
        X = data[features]
        y = data['SOH']
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # 保存到session
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.features = features
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.r2 = r2
        st.session_state.mae = mae
        st.session_state.mse = mse
        st.session_state.trained = True
        
        st.success("训练完成！")
        
        # 显示指标
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.2f}%")
        col3.metric("MSE", f"{mse:.4f}")

# 显示结果
if st.session_state.get('trained', False):
    
    st.subheader("预测结果对比")
    
    # 简单散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.5)
    ax.plot([60, 100], [60, 100], 'r--', linewidth=2)
    ax.set_xlabel("真实 SOH (%)")
    ax.set_ylabel("预测 SOH (%)")
    ax.set_title("真实值 vs 预测值")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # 误差分布
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    errors = st.session_state.y_test - st.session_state.y_pred
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_xlabel("误差 (%)")
    ax2.set_ylabel("频数")
    ax2.set_title("误差分布")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # 特征重要性
    if hasattr(st.session_state.model, 'feature_importances_'):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        importance = st.session_state.model.feature_importances_
        ax3.barh(st.session_state.features, importance)
        ax3.set_xlabel("重要性")
        ax3.set_title("特征重要性")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

# 预测功能
st.subheader("在线预测")

if st.session_state.get('trained', False):
    
    col1, col2 = st.columns(2)
    
    with col1:
        cycle = st.number_input("循环次数", min_value=1, max_value=3000, value=500)
        voltage = st.number_input("电压 (V)", min_value=10.0, max_value=14.0, value=12.5, step=0.1)
        current = st.number_input("电流 (A)", min_value=0.0, max_value=50.0, value=22.0, step=0.5)
    
    with col2:
        temp = st.number_input("温度 (℃)", min_value=-10.0, max_value=60.0, value=26.0, step=0.5)
        capacity = st.number_input("容量 (Ah)", min_value=0.0, max_value=60.0, value=45.0, step=0.5)
    
    if st.button("预测 SOH"):
        # 准备输入
        input_data = np.array([[cycle, voltage, current, temp, capacity]])
        input_scaled = st.session_state.scaler.transform(input_data)
        
        # 预测
        result = st.session_state.model.predict(input_scaled)[0]
        result = np.clip(result, 0, 100)
        
        # 显示结果
        if result >= 90:
            status = "优秀"
            color = "green"
        elif result >= 80:
            status = "良好"
            color = "blue"
        elif result >= 70:
            status = "注意"
            color = "orange"
        else:
            status = "需更换"
            color = "red"
        
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <h2>预测结果</h2>
            <h1 style="color: {color}; font-size: 48px;">{result:.1f}%</h1>
            <h3>健康状态: {status}</h3>
            <p>模型准确率: R² = {st.session_state.r2:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("请先点击左侧的「开始训练模型」按钮")

# 页脚
st.markdown("---")
st.caption("蓄电池健康状态预测系统 | 基于随机森林算法")
