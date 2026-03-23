# app.py - 稳定版蓄电池SOH预测系统

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
import io

warnings.filterwarnings('ignore')

# 设置页面配置 - 使用更稳定的配置
st.set_page_config(
    page_title="蓄电池SOH预测系统",
    page_icon="🔋",
    layout="centered",  # 改为centered更稳定
    initial_sidebar_state="auto"
)

# 初始化session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'test_r2' not in st.session_state:
    st.session_state.test_r2 = None

# 标题
st.title("🔋 蓄电池健康状态(SOH)预测系统")
st.markdown("---")

# 生成模拟数据
@st.cache_data
def generate_data():
    """生成模拟的电池循环测试数据"""
    np.random.seed(42)
    n_samples = 800
    cycles = np.arange(1, n_samples + 1)
    
    # 生成SOH数据（非线性退化）
    soh = 100 - 0.025 * cycles - 0.000015 * cycles**2 + np.random.normal(0, 0.6, n_samples)
    soh = np.clip(soh, 55, 100)
    
    # 容量与SOH正相关
    capacity = 50 * (soh / 100) + np.random.normal(0, 0.5, n_samples)
    capacity = np.clip(capacity, 25, 50)
    
    # 电压随SOH下降略有下降
    voltage = 12.8 * (soh / 100) + np.random.normal(0, 0.06, n_samples)
    voltage = np.clip(voltage, 11.0, 13.0)
    
    # 电流与SOH负相关
    current = 19 + 7 * (1 - soh/100) + np.random.normal(0, 0.5, n_samples)
    current = np.clip(current, 17, 28)
    
    # 温度随SOH下降升高
    temperature = 24 + 5 * (1 - soh/100) + np.random.normal(0, 0.4, n_samples)
    temperature = np.clip(temperature, 23, 32)
    
    df = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temperature,
        'Capacity': capacity,
        'SOH': soh
    })
    return df

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.markdown("---")
    
    # 数据生成
    if st.button("🔄 生成新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # 训练按钮
    train_button = st.button("🚀 开始训练模型", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("💡 提示：点击训练按钮后，模型会自动训练并显示结果")

# 生成数据
data = generate_data()

# 显示数据概览
with st.expander("📊 数据概览", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("数据总量", f"{len(data)} 条")
        st.metric("循环次数范围", f"{data['Cycle'].min()} - {data['Cycle'].max()}")
    with col2:
        st.metric("SOH范围", f"{data['SOH'].min():.1f}% - {data['SOH'].max():.1f}%")
        st.metric("容量范围", f"{data['Capacity'].min():.1f} - {data['Capacity'].max():.1f} Ah")
    
    st.dataframe(data.head(10), use_container_width=True)

# 训练模型
if train_button:
    with st.spinner("🔄 正在训练模型，请稍候..."):
        try:
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
            
            # 训练随机森林模型
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 保存到session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = features
            st.session_state.test_r2 = r2
            st.session_state.test_mae = mae
            st.session_state.model_trained = True
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.success("✅ 模型训练完成！")
            
        except Exception as e:
            st.error(f"训练出错: {str(e)}")
            st.stop()

# 显示模型结果
if st.session_state.model_trained:
    st.markdown("---")
    st.header("📈 模型评估结果")
    
    # 指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² 分数", f"{st.session_state.test_r2:.4f}", 
                  delta="接近1表示效果好" if st.session_state.test_r2 > 0.9 else None)
    with col2:
        st.metric("📉 MAE", f"{st.session_state.test_mae:.2f}%", 
                  delta="误差越小越好")
    with col3:
        st.metric("📐 MSE", f"{mean_squared_error(st.session_state.y_test, st.session_state.y_pred):.4f}", 
                  delta="均方误差")
    
    st.markdown("---")
    st.header("📊 可视化分析")
    
    # 创建两个标签页
    tab1, tab2, tab3 = st.tabs(["预测对比", "误差分析", "特征重要性"])
    
    with tab1:
        # 预测值 vs 真实值
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=st.session_state.y_test,
            y=st.session_state.y_pred,
            mode='markers',
            marker=dict(
                color='steelblue',
                size=8,
                opacity=0.6,
                line=dict(color='white', width=0.5)
            ),
            name='预测点'
        ))
        
        # 理想线
        min_val = min(st.session_state.y_test.min(), st.session_state.y_pred.min())
        max_val = max(st.session_state.y_test.max(), st.session_state.y_pred.max())
        fig1.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='理想线 (y=x)'
        ))
        
        fig1.update_layout(
            title="真实值 vs 预测值",
            xaxis_title="真实 SOH (%)",
            yaxis_title="预测 SOH (%)",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # SOH退化曲线
        results_df = pd.DataFrame({
            'Cycle': data.loc[st.session_state.y_test.index, 'Cycle'].values,
            '真实SOH': st.session_state.y_test.values,
            '预测SOH': st.session_state.y_pred
        }).sort_values('Cycle')
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=results_df['Cycle'],
            y=results_df['真实SOH'],
            mode='lines',
            name='真实SOH',
            line=dict(color='blue', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=results_df['Cycle'],
            y=results_df['预测SOH'],
            mode='lines',
            name='预测SOH',
            line=dict(color='orange', width=2, dash='dot')
        ))
        
        fig2.update_layout(
            title="SOH退化曲线对比",
            xaxis_title="循环次数",
            yaxis_title="SOH (%)",
            height=450
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # 误差分布
        errors = st.session_state.y_test - st.session_state.y_pred
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
        ax3.axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=1, 
                   label=f'均值: {np.mean(errors):.2f}%')
        ax3.set_xlabel('预测误差 (%)')
        ax3.set_ylabel('频数')
        ax3.set_title('预测误差分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        
        # 误差统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均误差", f"{np.mean(errors):.2f}%")
        with col2:
            st.metric("误差标准差", f"{np.std(errors):.2f}%")
        with col3:
            st.metric("误差范围", f"{errors.min():.2f}% ~ {errors.max():.2f}%")
    
    with tab3:
        # 特征重要性
        if hasattr(st.session_state.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                '特征': st.session_state.features,
                '重要性': st.session_state.model.feature_importances_
            }).sort_values('重要性', ascending=True)
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            ax4.barh(importance_df['特征'], importance_df['重要性'], color=colors)
            ax4.set_xlabel('重要性得分')
            ax4.set_title('特征重要性分析')
            ax4.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig4)
            
            # 显示详细数值
            st.write("特征重要性详细数值：")
            st.dataframe(importance_df, use_container_width=True)

# 在线预测模块
st.markdown("---")
st.header("🎯 在线预测")

if not st.session_state.model_trained:
    st.info("💡 请先点击左侧边栏的「开始训练模型」按钮训练模型，然后就可以使用预测功能了。")
else:
    st.write("请输入电池参数进行SOH预测：")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cycle = st.number_input("🔄 循环次数", min_value=1, max_value=3000, value=500, step=10)
        voltage = st.number_input("⚡ 电压 (V)", min_value=10.0, max_value=14.0, value=12.5, step=0.05, format="%.2f")
        current = st.number_input("🔌 电流 (A)", min_value=0.0, max_value=50.0, value=22.0, step=0.5)
    
    with col2:
        temperature = st.number_input("🌡️ 温度 (℃)", min_value=-10.0, max_value=60.0, value=26.0, step=0.5)
        capacity = st.number_input("🔋 容量 (Ah)", min_value=0.0, max_value=60.0, value=45.0, step=0.5)
    
    if st.button("🔮 预测SOH", type="primary", use_container_width=True):
        try:
            # 构建输入
            input_data = pd.DataFrame([[cycle, voltage, current, temperature, capacity]], 
                                     columns=st.session_state.features)
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # 预测
            prediction = st.session_state.model.predict(input_scaled)[0]
            prediction = np.clip(prediction, 0, 100)
            
            # 确定评级
            if prediction >= 90:
                status = "🌟 优秀"
                color = "green"
                advice = "电池性能良好，继续正常使用"
            elif prediction >= 80:
                status = "✅ 良好"
                color = "#4caf50"
                advice = "电池性能正常，建议定期检查"
            elif prediction >= 70:
                status = "⚠️ 注意"
                color = "#ff9800"
                advice = "电池性能下降，建议关注充放电情况"
            else:
                status = "🔴 需更换"
                color = "#f44336"
                advice = "电池性能严重下降，建议尽快更换"
            
            # 显示结果
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center; 
                        margin-top: 1rem;">
                <h2 style="color: white; margin-bottom: 0.5rem;">📊 预测结果</h2>
                <h1 style="color: white; font-size: 72px; margin: 0.5rem 0;">{prediction:.1f}%</h1>
                <h3 style="color: white; margin: 0.5rem 0;">健康状态评级: {status}</h3>
                <p style="color: white; margin-top: 1rem;">{advice}</p>
                <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.3);">
                <p style="color: white; font-size: 0.9rem;">模型测试集表现: R² = {st.session_state.test_r2:.3f} | MAE = {st.session_state.test_mae:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"预测出错: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p>🔋 <strong>蓄电池健康状态预测系统</strong> | 基于随机森林算法</p>
    <p style="color: #666; font-size: 0.8rem;">科学预测，智能运维 | 模型持续优化中</p>
</div>
""", unsafe_allow_html=True)
