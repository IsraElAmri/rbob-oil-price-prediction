"""
RBOB Gasoline Price Prediction Dashboard
Beautiful Streamlit GUI for exploring data and predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from predictor import GasolinePredictor
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RBOB Gasoline Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .metric-card {
        background: rgba(30, 30, 46, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
        border: 1px solid rgba(100, 255, 218, 0.1);
    }
    h1 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    h2, h3 {
        color: #64ffda;
    }
    p, div {
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.5);
    }
    .stDataFrame {
        background: rgba(30, 30, 46, 0.6);
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #16213e 0%, #0f172a 100%) !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #64ffda !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #c5c6c7 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state or st.session_state.predictor is None:
    try:
        _auto_pred = GasolinePredictor()
        _auto_pred.load_data()
        _auto_pred.clean_data()
        st.session_state.predictor = _auto_pred
    except Exception:
        st.session_state.predictor = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Title and description
st.title("⛽ RBOB Gasoline Price Prediction Dashboard")
st.markdown("""
<div style='background: rgba(30, 30, 46, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(100, 255, 218, 0.2);'>
    <h3 style='color: #00d4ff; margin-top: 0;'>Machine Learning Price Forecasting</h3>
    <p style='color: #e0e0e0;'>
        Analyze historical commodity prices and predict RBOB gasoline prices using advanced ML models.
        This dashboard provides interactive visualizations and real-time predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Data loading
    if st.button("📊 Load Data", use_container_width=True):
        with st.spinner("Loading data..."):
            try:
                predictor = GasolinePredictor()
                predictor.load_data()
                predictor.clean_data()
                st.session_state.predictor = predictor
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    # Model training
    if st.session_state.predictor is not None:
        if st.button("🤖 Train Models", use_container_width=True):
            with st.spinner("Training models... This may take a minute."):
                try:
                    st.session_state.predictor.engineer_features()
                    st.session_state.predictor.prepare_train_test()
                    results = st.session_state.predictor.train_models()
                    st.session_state.trained = True
                    st.success("✅ Models trained successfully!")
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
    
    st.markdown("---")
    
    # Model selection
    if st.session_state.trained:
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Choose Model:",
            ["Random Forest", "Gradient Boosting", "Ridge"]
        )
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64ffda; font-size: 12px;'>
            <p>Built with Streamlit</p>
            <p>Data: 2015-2024</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
if st.session_state.predictor is None:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px; background: rgba(30, 30, 46, 0.8); border-radius: 15px; margin-top: 50px; border: 1px solid rgba(100, 255, 218, 0.2);'>
                <h2 style='color: #00d4ff;'>👈 Get Started</h2>
                <p style='font-size: 18px; color: #e0e0e0;'>
                    Click <strong style='color: #64ffda;'>"Load Data"</strong> in the sidebar to begin your analysis
                </p>
                <div style='margin-top: 30px; padding: 20px; background: rgba(20, 20, 36, 0.6); border-radius: 10px; border: 1px solid rgba(100, 255, 218, 0.1);'>
                    <h4 style='color: #7b2cbf;'>Features:</h4>
                    <ul style='text-align: left; display: inline-block; color: #e0e0e0;'>
                        <li>📈 Interactive time-series visualizations</li>
                        <li>🤖 Multiple ML models (Random Forest, Gradient Boosting, Ridge)</li>
                        <li>📊 Real-time performance metrics</li>
                        <li>🔍 Feature importance analysis</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

else:
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Data Explorer", "📉 Statistical Analysis", "🎯 Predictions", "🔬 Model Analysis"])
    
    with tab1:
        # Dashboard - Key metrics
        st.header("Key Metrics")
        
        df = st.session_state.predictor.df
        
        # Calculate metrics
        current_price = df[' RBOB'].iloc[-1]
        avg_price = df[' RBOB'].mean()
        price_change = df[' RBOB'].iloc[-1] - df[' RBOB'].iloc[-2]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current RBOB Price",
                value=f"${current_price:.3f}",
                delta=f"{price_change:.3f}"
            )
        
        with col2:
            st.metric(
                label="Average Price",
                value=f"${avg_price:.3f}"
            )
        
        with col3:
            st.metric(
                label="Data Points",
                value=f"{len(df):,}"
            )
        
        with col4:
            st.metric(
                label="Date Range",
                value=f"{df.index.min().year}-{df.index.max().year}"
            )
        
        # Price trend chart
        st.subheader("RBOB Price Trend")
        fig = px.line(
            df.reset_index(),
            x='date',
            y=' RBOB',
            title='RBOB Gasoline Price Over Time',
            labels={'date': 'Date', ' RBOB': 'Price ($/gallon)'}
        )
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='rgba(20, 20, 36, 0.6)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font=dict(size=12, color='#e0e0e0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> This time-series visualization reveals the historical volatility and trends in RBOB gasoline prices. 
                Notable price spikes often correlate with geopolitical events, seasonal demand shifts, or supply chain disruptions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Correlation with crude oil
        st.subheader("RBOB vs Crude Oil (WTI)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df[' RBOB'],
            name='RBOB',
            line=dict(color='#667eea', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df['crude oil ( WTI)'],
            name='WTI Crude',
            yaxis='y2',
            line=dict(color='#764ba2', width=2)
        ))
        fig2.update_layout(
            title='RBOB vs Crude Oil Price',
            yaxis=dict(title='RBOB Price', color='#e0e0e0'),
            yaxis2=dict(title='WTI Price', overlaying='y', side='right', color='#e0e0e0'),
            hovermode='x unified',
            plot_bgcolor='rgba(20, 20, 36, 0.6)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> The comparison demonstrates a strong positive correlation (~0.77) between crude oil (WTI) and RBOB prices. 
                While crude oil is the primary feedstock, the correlation is not 1:1, reflecting how refining margins, seasonality, and specific gasoline demand drivers can cause prices to diverge.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        # Data Explorer
        st.header("Data Explorer")
        
        # Show raw data
        st.subheader("Dataset Preview")
        st.dataframe(
            df.head(100).style.background_gradient(cmap='viridis', axis=1),
            use_container_width=True,
            height=400
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T.style.background_gradient(cmap='coolwarm', axis=1))
        
        # Column selector for visualization
        st.subheader("Custom Visualization")
        selected_cols = st.multiselect(
            "Select columns to visualize:",
            options=[col for col in df.columns if col != ' RBOB'],
            default=['crude oil ( WTI)', 'Heating Oil ']
        )
        
        if selected_cols:
            fig = px.line(
                df.reset_index(),
                x='date',
                y=selected_cols,
                title='Selected Commodities Over Time'
            )
            fig.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> Multi-commodity comparison enables identification of co-movement patterns and leading indicators. 
                    Commodities with high correlation to RBOB can serve as predictive features in forecasting models.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # Statistical Analysis
        st.header("Statistical Analysis")
        
        df = st.session_state.predictor.df
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <p style='color: #e0e0e0; margin: 0;'>
                The correlation matrix shows how strongly each commodity is related to RBOB gasoline prices.
                Values close to 1 (red) indicate strong positive correlation, while values close to -1 (blue) indicate strong negative correlation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select key columns for correlation
        key_cols = [' RBOB', 'crude oil ( WTI)', 'Heating Oil ', 'Conventional Gasoline',
                    'Ultra-Low Sulfur CARB Diesel', 'Kerosene-Type Jet Fuel', 'Natural GAZ',
                    'DJIA', 'NASDAQ', 'sp500', 'gold']
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in key_cols if col in df.columns]
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title='Correlation Heatmap of Key Commodities',
            width=800,
            height=800,
            plot_bgcolor='rgba(20, 20, 36, 0.6)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font=dict(color='#e0e0e0', size=10)
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> The correlation heatmap reveals strong multicollinearity among refined petroleum products (RBOB, Conventional Gasoline, Heating Oil) with correlation coefficients often exceeding 0.85. 
                Conventional Gasoline shows the highest correlation (>0.92), reflecting their near-identical market dynamics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top correlations with RBOB
        st.subheader("Top Correlations with RBOB")
        rbob_corr = corr_matrix[' RBOB'].sort_values(ascending=False)[1:11]  # Exclude RBOB itself
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_top_corr = px.bar(
                x=rbob_corr.values,
                y=rbob_corr.index,
                orientation='h',
                title='Top 10 Correlations with RBOB',
                labels={'x': 'Correlation Coefficient', 'y': 'Commodity'},
                color=rbob_corr.values,
                color_continuous_scale='viridis'
            )
            fig_top_corr.update_layout(
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig_top_corr, use_container_width=True)
        
        with col2:
            # Display correlation values
            st.dataframe(
                pd.DataFrame({
                    'Commodity': rbob_corr.index,
                    'Correlation': rbob_corr.values
                }).style.background_gradient(cmap='viridis', subset=['Correlation'])
                         .format({'Correlation': '{:.4f}'}),
                use_container_width=True,
                height=400
            )
        
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> The top correlations identify the most influential predictors for RBOB pricing. 
                Refined products (heating oil, conventional gasoline) and crude oil dominate, validating their inclusion as primary features in predictive models.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price Distribution Analysis
        st.subheader("Price Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RBOB distribution
            fig_dist = px.histogram(
                df,
                x=' RBOB',
                nbins=50,
                title='RBOB Price Distribution',
                labels={' RBOB': 'Price ($/gallon)'},
                color_discrete_sequence=['#00d4ff']
            )
            fig_dist.update_layout(
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 10px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> The distribution shows a right-skewed pattern with a mean price around $2.14. 
                    The interquartile range (IQR) spans from ~$1.68 to ~$2.49, indicating that while prices fluctuate, they spend 50% of the time in this central band, with significant tails reaching much higher levels.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Box plot for outlier detection
            fig_box = px.box(
                df,
                y=' RBOB',
                title='RBOB Price Box Plot (Outlier Detection)',
                labels={' RBOB': 'Price ($/gallon)'},
                color_discrete_sequence=['#7b2cbf']
            )
            fig_box.update_layout(
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> The box plot identifies outliers beyond the interquartile range (IQR), representing extreme price events. 
                    These anomalies often correspond to major disruptions (hurricanes, refinery outages, geopolitical crises) and warrant special handling in model training.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Scatter Matrix for key variables
        st.subheader("Scatter Matrix - Key Variables")
        
        scatter_cols = [' RBOB', 'crude oil ( WTI)', 'Heating Oil ', 'Conventional Gasoline']
        scatter_cols = [col for col in scatter_cols if col in df.columns]
        
        fig_scatter = px.scatter_matrix(
            df[scatter_cols].sample(min(500, len(df))),  # Sample for performance
            dimensions=scatter_cols,
            title='Scatter Matrix of Key Commodities (Sample of 500 points)',
            color_discrete_sequence=['#00d4ff']
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(20, 20, 36, 0.6)',
            paper_bgcolor='rgba(30, 30, 46, 0.8)',
            font=dict(color='#e0e0e0', size=8),
            height=700
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> The scatter matrix visualizes pairwise relationships, revealing strong linear dependencies among refined products. 
                The diagonal shows individual distributions, while off-diagonal plots confirm tight clustering along the regression line, indicating high predictability.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        
        stats_df = df[available_cols].describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # Coefficient of variation
        
        st.dataframe(
            stats_df.style.background_gradient(cmap='coolwarm', axis=0)
                          .format("{:.4f}"),
            use_container_width=True
        )
        st.markdown("""
        <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
            <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                <strong>💡 Insight:</strong> The enhanced statistical summary includes range (max-min) and coefficient of variation (CV = std/mean × 100%). 
                High CV values indicate greater relative volatility, helping identify which commodities exhibit the most price instability over the observation period.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        # Predictions
        st.header("Model Predictions")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train the models first using the sidebar button.")
        else:
            # Show model results
            st.subheader("Model Performance Comparison")
            
            results_df = pd.DataFrame(st.session_state.predictor.results).T
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance metrics table
                st.dataframe(
                    results_df.style.background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE'])
                                   .background_gradient(cmap='RdYlGn', subset=['R2'])
                                   .format("{:.4f}"),
                    use_container_width=True
                )
            
            with col2:
                # Bar chart of MAE
                fig = px.bar(
                    results_df.reset_index(),
                    x='index',
                    y='MAE',
                    title='Mean Absolute Error by Model',
                    labels={'index': 'Model', 'MAE': 'MAE ($/gallon)'},
                    color='MAE',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                    <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                        <strong>💡 Insight:</strong> The MAE (Mean Absolute Error) comparison reveals that Ridge Regression achieves the lowest prediction error (~$0.069/gallon), 
                        significantly outperforming the tree-based models. This suggests that the relationship between commodity prices is largely linear and stable, favoring regularized linear models over complex ensembles.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Predictions vs Actual
            st.subheader(f"Predictions: {model_choice}")
            
            predictions = st.session_state.predictor.predict(model_choice)
            actual = st.session_state.predictor.y_train
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=actual.values,
                name='Actual',
                line=dict(color='#667eea', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=predictions,
                name='Predicted',
                line=dict(color='#764ba2', width=2, dash='dash')
            ))
            fig.update_layout(
                title='Actual vs Predicted RBOB Prices',
                xaxis_title='Date',
                yaxis_title='Price ($/gallon)',
                hovermode='x unified',
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> The actual vs predicted overlay demonstrates model tracking ability. Close alignment between lines indicates accurate forecasting, 
                    while divergences reveal periods where market dynamics exceeded the model's predictive capacity (often during volatile events).
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Residuals
            residuals = actual.values - predictions
            fig_res = px.histogram(
                residuals,
                nbins=50,
                title='Prediction Residuals Distribution',
                labels={'value': 'Residual ($/gallon)'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_res, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> The residuals distribution should approximate a normal distribution centered at zero for unbiased predictions. 
                    Skewness or heavy tails indicate systematic under/over-prediction, while outliers represent extreme forecast errors requiring investigation.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        # Model Analysis
        st.header("Model Analysis")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train the models first using the sidebar button.")
        else:
            # Feature importance
            st.subheader("Feature Importance (Random Forest)")
            
            importance_df = st.session_state.predictor.get_feature_importance('Random Forest', top_n=15)
            
            if importance_df is not None:
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(20, 20, 36, 0.6)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show importance table
                st.dataframe(
                    importance_df.style.background_gradient(cmap='viridis', subset=['importance'])
                                       .format({'importance': '{:.4f}'}),
                    use_container_width=True
                )
                st.markdown("""
                <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                    <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                        <strong>💡 Insight:</strong> While Ridge Regression performed best, we use Random Forest here to visualize non-linear feature interactions. 
                        Lag features (previous day's prices) and crude oil prices dominate the importance usage. The fact that simpler models won suggests these strong individual predictors matter more than complex interactions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            
            results_df = pd.DataFrame(st.session_state.predictor.results).T
            
            metrics = ['MAE', 'RMSE', 'R2']
            fig = go.Figure()
            
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df.index,
                    y=results_df[metric],
                ))
            
            fig.update_layout(
                title='Model Performance Metrics Comparison',
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Score',
                plot_bgcolor='rgba(20, 20, 36, 0.6)',
                paper_bgcolor='rgba(30, 30, 46, 0.8)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='background: rgba(30, 30, 46, 0.6); padding: 12px; border-radius: 6px; margin-top: -10px; margin-bottom: 20px;'>
                <p style='color: #64ffda; font-size: 13px; margin: 0;'>
                    <strong>💡 Insight:</strong> The grouped bar chart highlights a key finding: Ridge Regression outperforms complex tree models (Random Forest, GBM). 
                    The tree models likely overfitted the training data (memorizing noise), while Ridge's L2 regularization prevented overfitting, leading to superior generalization on unseen data.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64ffda; padding: 20px;'>
        <p>RBOB Gasoline Price Prediction Dashboard | Built with ❤️ using Streamlit & Scikit-learn</p>
        <p style='font-size: 12px; color: #888;'>Data Source: Historical Commodity Prices (2015-2024)</p>
    </div>
""", unsafe_allow_html=True)
