# app.py - 简化版蓄电池SOH预测系统

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(page_title="蓄电池SOH预测", page_icon="🔋", layout="wide")

# 标题
st.title("🔋 蓄电池健康状态(SOH)预测系统")
st.markdown("---")

# 生成模拟数据
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 1000
    cycles = np.arange(1, n_samples + 1)
    
    # 生成SOH数据
    soh = 100 - 0.028 * cycles + np.random.normal(0, 0.8, n_samples)
    soh = np.clip(soh, 60, 100)
    
    # 生成特征
    capacity = 50 * (soh / 100) + np.random.normal(0, 0.6, n_samples)
    voltage = 12.8 * (soh / 100) + np.random.normal(0, 0.08, n_samples)
    current = 20 + 6 * (1 - soh/100) + np.random.normal(0, 0.6, n_samples)
    temperature = 25 + 4 * (1 - soh/100) + np.random.normal(0, 0.4, n_samples)
    
    df = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': np.clip(voltage, 11.2, 12.9),
        'Current': np.clip(current, 18, 27),
        'Temperature': np.clip(temperature, 24, 31),
        'Capacity': np.clip(capacity, 28, 50),
        'SOH': soh
    })
    return df

# 侧边栏
st.sidebar.header("⚙️ 配置")

# 数据选择
data = generate_data()
st.sidebar.success(f"✅ 已生成 {len(data)} 条数据")

# 显示数据预览
with st.expander("📊 查看数据预览"):
    st.dataframe(data.head(10))
    st.write("数据统计：")
    st.dataframe(data.describe())

# 训练按钮
if st.sidebar.button("🚀 开始训练模型", type="primary"):
    with st.spinner("正在训练模型..."):
        # 准备数据
        features = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity']
        X = data[features]
        y = data['SOH']
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 显示结果
        st.subheader("📈 模型评估结果")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² 分数", f"{r2:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.2f}%")
        with col3:
            st.metric("MSE", f"{mse:.4f}")
        
        # 可视化
        st.subheader("📊 预测结果可视化")
        
        # 创建图表
        fig = go.Figure()
        
        # 真实值 vs 预测值散点图
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            marker=dict(color='steelblue', size=8, opacity=0.6),
            name='预测点'
        ))
        
        # 理想线
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='理想线'
        ))
        
        fig.update_layout(
            title="真实值 vs 预测值",
            xaxis_title="真实 SOH (%)",
            yaxis_title="预测 SOH (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 误差分布
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        errors = y_test - y_pred
        ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('预测误差 (%)')
        ax2.set_ylabel('频数')
        ax2.set_title('预测误差分布')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # 保存模型到session
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['features'] = features
        
        st.success("✅ 模型训练完成！")
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 特征重要性分析")
            importance_df = pd.DataFrame({
                '特征': features,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=True)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.barh(importance_df['特征'], importance_df['重要性'], color='teal')
            ax3.set_xlabel('重要性')
            ax3.set_title('特征重要性')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)

# 在线预测模块
st.subheader("🎯 在线预测")
st.markdown("输入电池参数，预测当前健康状态")

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input("循环次数", min_value=1, max_value=5000, value=500)
    voltage = st.number_input("电压 (V)", min_value=10.0, max_value=15.0, value=12.5, step=0.1)
    current = st.number_input("电流 (A)", min_value=0.0, max_value=100.0, value=22.0, step=0.5)

with col2:
    temperature = st.number_input("温度 (℃)", min_value=-10.0, max_value=60.0, value=26.0, step=0.5)
    capacity = st.number_input("容量 (Ah)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)

if st.button("🔮 预测SOH", type="primary"):
    if 'model' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        features = st.session_state['features']
        
        # 构建输入
        input_data = pd.DataFrame([[cycle, voltage, current, temperature, capacity]], 
                                  columns=features)
        input_scaled = scaler.transform(input_data)
        
        # 预测
        prediction = model.predict(input_scaled)[0]
        prediction = np.clip(prediction, 0, 100)
        
        # 显示结果
        if prediction >= 90:
            status = "🌟 优秀"
            color = "green"
        elif prediction >= 80:
            status = "✅ 良好"
            color = "orange"
        elif prediction >= 70:
            status = "⚠️ 注意"
            color = "orange"
        else:
            status = "🔴 需更换"
            color = "red"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
            <h2 style="color: white;">预测结果</h2>
            <h1 style="color: white; font-size: 64px;">{prediction:.1f}%</h1>
            <h3 style="color: white;">健康状态评级: {status}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 请先点击左侧的'开始训练模型'按钮训练模型！")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 使用步骤
    1. 点击左侧边栏的 **"开始训练模型"** 按钮
    2. 等待模型训练完成（约10-30秒）
    3. 查看模型评估结果和可视化图表
    4. 在"在线预测"部分输入电池参数
    5. 点击 **"预测SOH"** 查看结果
    
    ### 健康状态说明
    - **90-100%**: 🌟 优秀 - 电池性能良好
    - **80-90%**: ✅ 良好 - 电池性能正常
    - **70-80%**: ⚠️ 注意 - 性能下降，建议关注
    - **<70%**: 🔴 需更换 - 性能严重下降
    
    ### 特征说明
    - **循环次数**: 充放电次数越多，电池老化越严重
    - **容量**: 电池当前容量，与SOH正相关
    - **电压**: 电池电压，SOH越低电压越低
    - **电流**: 充放电电流，老化会导致内阻增加
    - **温度**: 电池温度，老化会导致发热增加
    """)

st.markdown("---")
st.markdown("🔋 **蓄电池健康状态预测系统** | 基于随机森林算法")
