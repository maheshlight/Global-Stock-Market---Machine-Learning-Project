import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Stock Market ML Project",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 10px 0 4px 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #333;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 10px;
    }
    .stSelectbox label, .stMultiselect label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA (CACHED) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_excel("Global_Stock_Market_Master_Dataset.xlsx", header=2)
    df.columns = ['Date', 'Country', 'Company', 'Sector', 'Sub_Sector',
                  'Open', 'High', 'Low', 'Close', 'Volume',
                  'BUY', 'SELL', 'Daily_Return', 'War_Period']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Close'] > 0) & (df['Volume'] > 0)].reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Price_Range'] = df['High'] - df['Low']
    df['BuySell_Ratio'] = df['BUY'] / (df['SELL'] + 1)
    df['Period'] = df['War_Period'].apply(
        lambda x: 'Post-War' if 'Post' in str(x) else 'Pre-War')
    return df

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌍 Global Stock Market ML")
st.sidebar.markdown("---")

page = st.sidebar.radio("📂 Navigate to", [
    "🏠 Overview & EDA",
    "📈 PS1 — Stock Price Prediction",
    "🚦 PS2 — Buy / Sell Signal",
    "🗺️ PS3 — Market Performance Analysis",
    "💡 PS4 — Investor Sentiment",
    "💼 PS5 — Portfolio Optimization",
    "🌪️ PS6 — Volatility Forecasting",
    "🔍 PS7 — Anomaly Detection",
    "📊 PS8 — Trend Classification",
    "⚔️ PS9 — War Period Impact",
    "🔄 PS10 — Sector Rotation"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records:** {len(df):,}")
st.sidebar.markdown(f"**Companies:** {df['Company'].nunique()}")
st.sidebar.markdown(f"**Countries:** {df['Country'].nunique()}")
st.sidebar.markdown(f"**Date Range:** {df['Date'].min().date()} → {df['Date'].max().date()}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW & EDA
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & EDA":
    st.markdown('<div class="main-header">🌍 Global Stock Market — ML Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">90,040 Records · 200 Companies · 10 Countries · Jan 2023 – Mar 2026</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records", f"{len(df):,}")
    c2.metric("🏢 Companies", df['Company'].nunique())
    c3.metric("🌐 Countries", df['Country'].nunique())
    c4.metric("📂 Sectors", df['Sector'].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Records per Country")
        cnt = df['Country'].value_counts().reset_index()
        cnt.columns = ['Country', 'Count']
        fig = px.bar(cnt, x='Country', y='Count', color='Count',
                     color_continuous_scale='Blues', template='plotly_dark')
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📂 Records per Sector")
        sec = df['Sector'].value_counts().reset_index()
        sec.columns = ['Sector', 'Count']
        fig = px.pie(sec, names='Sector', values='Count',
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     template='plotly_dark')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📈 Average Daily Return by Country")
    ret_country = df.groupby('Country')['Daily_Return'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(ret_country, x='Country', y='Daily_Return', color='Daily_Return',
                 color_continuous_scale='RdYlGn', template='plotly_dark',
                 labels={'Daily_Return': 'Avg Daily Return (%)'})
    fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
    fig.update_layout(height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Raw Data Preview")
    st.dataframe(df.head(200), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STOCK PRICE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 PS1 — Stock Price Prediction":
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    st.markdown('<div class="section-title">📈 PS1 — Stock Price Prediction</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Predict next-day closing price using lag features and moving averages.")

    company = st.selectbox("Select Company", sorted(df['Company'].unique()))

    @st.cache_data(show_spinner="Training models...")
    def run_ps1(company):
        d = df[df['Company'] == company].sort_values('Date').copy()
        d['Lag1_Close'] = d['Close'].shift(1)
        d['Lag2_Close'] = d['Close'].shift(2)
        d['Lag1_Return'] = d['Daily_Return'].shift(1)
        d['MA5']  = d['Close'].rolling(5).mean()
        d['MA10'] = d['Close'].rolling(10).mean()
        d['Next_Close'] = d['Close'].shift(-1)
        d = d.dropna()
        features = ['Open','High','Low','Close','Volume',
                    'Lag1_Close','Lag2_Close','Lag1_Return','MA5','MA10','Price_Range']
        X = d[features]; y = d['Next_Close']
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xts_s = sc.transform(Xts)
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        results = {}
        for name, m in models.items():
            m.fit(Xtr_s, ytr)
            preds = m.predict(Xts_s)
            results[name] = {
                'MAE': round(mean_absolute_error(yts, preds), 2),
                'RMSE': round(np.sqrt(mean_squared_error(yts, preds)), 2),
                'R2': round(r2_score(yts, preds), 4),
                'preds': preds
            }
        rf_imp = pd.Series(models['Random Forest'].feature_importances_, index=features).sort_values(ascending=False)
        return results, yts.values, rf_imp

    results, y_true, rf_imp = run_ps1(company)

    col1, col2, col3 = st.columns(3)
    for col, (name, res) in zip([col1, col2, col3], results.items()):
        col.metric(name, f"R² = {res['R2']}", f"MAE = {res['MAE']}")

    st.markdown("#### Actual vs Predicted (Random Forest — first 200 samples)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:200], name='Actual', line=dict(color='steelblue', width=2)))
    fig.add_trace(go.Scatter(y=results['Random Forest']['preds'][:200], name='Predicted',
                             line=dict(color='tomato', width=2, dash='dash')))
    fig.update_layout(template='plotly_dark', height=380,
                      xaxis_title='Sample', yaxis_title='Close Price')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### R² Score Comparison")
        fig = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[r['R2'] for r in results.values()],
            marker_color=['#2196F3','#4CAF50','#FF9800'],
            text=[r['R2'] for r in results.values()],
            textposition='outside'
        ))
        fig.update_layout(template='plotly_dark', height=350,
                          yaxis_title='R² Score', yaxis_range=[-0.1, 1.1])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Feature Importance (Random Forest)")
        fig = go.Figure(go.Bar(
            x=rf_imp.index, y=rf_imp.values, marker_color='steelblue',
            text=rf_imp.round(3).values, textposition='outside'
        ))
        fig.update_layout(template='plotly_dark', height=350,
                          xaxis_tickangle=-30, yaxis_title='Importance')
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUY / SELL SIGNAL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🚦 PS2 — Buy / Sell Signal":
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

    st.markdown('<div class="section-title">🚦 PS2 — Buy / Sell Signal Classification</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Classify each trading day as BUY (1) or SELL (0).")

    @st.cache_data(show_spinner="Training classifiers...")
    def run_ps2():
        d = df.copy().sort_values(['Company','Date']).reset_index(drop=True)
        d['Next_Close'] = d.groupby('Company')['Close'].shift(-1)
        d['Signal'] = (d['Next_Close'] > d['Close']).astype(int)
        d['Lag1_Close']  = d.groupby('Company')['Close'].shift(1)
        d['Lag1_Return'] = d.groupby('Company')['Daily_Return'].shift(1)
        d['MA5']  = d.groupby('Company')['Close'].transform(lambda x: x.rolling(5).mean())
        d['MA10'] = d.groupby('Company')['Close'].transform(lambda x: x.rolling(10).mean())
        d['Volatility5'] = d.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(5).std())
        d = d.dropna()
        feats = ['Open','High','Low','Close','Volume','Lag1_Close',
                 'Lag1_Return','MA5','MA10','Price_Range','Volatility5','BuySell_Ratio']
        X = d[feats]; y = d['Signal']
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xts_s = sc.transform(Xts)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        }
        res = {}
        for name, m in models.items():
            m.fit(Xtr_s, ytr)
            preds = m.predict(Xts_s)
            proba = m.predict_proba(Xts_s)[:,1]
            rep = classification_report(yts, preds, output_dict=True)
            fpr, tpr, _ = roc_curve(yts, proba)
            res[name] = {'acc': rep['accuracy'], 'cm': confusion_matrix(yts, preds),
                         'auc': roc_auc_score(yts, proba), 'fpr': fpr, 'tpr': tpr}
        return res
    res = run_ps2()

    c1, c2, c3 = st.columns(3)
    for col, (name, r) in zip([c1,c2,c3], res.items()):
        col.metric(name, f"Acc = {r['acc']:.3f}", f"AUC = {r['auc']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC Curves")
        fig = go.Figure()
        colors = ['#2196F3','#4CAF50','#FF9800']
        for (name, r), color in zip(res.items(), colors):
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], name=f"{name} (AUC={r['auc']:.3f})",
                                     line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'),
                                 showlegend=False))
        fig.update_layout(template='plotly_dark', height=380,
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Confusion Matrix — Random Forest")
        cm = res['Random Forest']['cm']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x='Predicted', y='Actual'),
                        x=['SELL','BUY'], y=['SELL','BUY'])
        fig.update_layout(template='plotly_dark', height=380)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARKET PERFORMANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ PS3 — Market Performance Analysis":
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    st.markdown('<div class="section-title">🗺️ PS3 — Cross-Sector & Cross-Country Market Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Understand sector performance across countries and cluster similar markets.")

    cs = df.groupby(['Country','Sector']).agg(
        Avg_Return=('Daily_Return','mean'),
        Volatility=('Daily_Return','std'),
        Avg_Volume=('Volume','mean'),
        Avg_Close=('Close','mean')
    ).reset_index().dropna()

    st.markdown("#### Heatmap — Avg Daily Return by Country × Sector")
    pivot = cs.pivot_table(values='Avg_Return', index='Country', columns='Sector', fill_value=0)
    fig = px.imshow(pivot, color_continuous_scale='RdYlGn', text_auto='.2f',
                    aspect='auto', template='plotly_dark')
    fig.update_layout(height=420, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Daily Return Distribution by Country")
        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, country in enumerate(df['Country'].unique()):
            fig.add_trace(go.Box(y=df[df['Country']==country]['Daily_Return'],
                                 name=country, marker_color=palette[i%len(palette)], boxpoints=False))
        fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=1)
        fig.update_layout(template='plotly_dark', height=380, showlegend=False,
                          yaxis_title='Daily Return (%)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### K-Means Clustering — Country-Sector Groups")
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        X_cl = StandardScaler().fit_transform(cs[['Avg_Return','Volatility','Avg_Volume','Avg_Close']])
        cs['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_cl)
        fig = px.scatter(cs, x='Avg_Return', y='Volatility', color=cs['Cluster'].astype(str),
                         hover_data=['Country','Sector'], template='plotly_dark',
                         labels={'color':'Cluster'}, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=380, xaxis_title='Avg Return (%)', yaxis_title='Volatility')
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INVESTOR SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡 PS4 — Investor Sentiment":
    st.markdown('<div class="section-title">💡 PS4 — Investor Behavior & Market Sentiment</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Analyze BUY/SELL ratio as a sentiment indicator and its correlation with returns.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### BuySell Ratio vs Daily Return (Scatter)")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(sample, x='BuySell_Ratio', y='Daily_Return', color='Sector',
                         opacity=0.4, template='plotly_dark',
                         labels={'BuySell_Ratio':'BUY/SELL Ratio','Daily_Return':'Daily Return (%)'})
        fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Average BuySell Ratio by Sector")
        sent = df.groupby('Sector')['BuySell_Ratio'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(sent, x='Sector', y='BuySell_Ratio', color='BuySell_Ratio',
                     color_continuous_scale='RdYlGn', template='plotly_dark')
        fig.update_layout(height=380, xaxis_tickangle=-35, yaxis_title='Avg BUY/SELL Ratio')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rolling 30-Day Sentiment Trend by Country")
    country_sel = st.selectbox("Select Country", sorted(df['Country'].unique()))
    roll = df[df['Country']==country_sel].set_index('Date').sort_index()
    roll['Roll_Sentiment'] = roll['BuySell_Ratio'].rolling(30).mean()
    roll['Roll_Return'] = roll['Daily_Return'].rolling(30).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('30-Day Rolling BUY/SELL Ratio','30-Day Rolling Avg Return'))
    fig.add_trace(go.Scatter(x=roll.index, y=roll['Roll_Sentiment'],
                             line=dict(color='#4CAF50', width=2), name='Sentiment'), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll.index, y=roll['Roll_Return'],
                             line=dict(color='steelblue', width=2), name='Return'), row=2, col=1)
    fig.update_layout(template='plotly_dark', height=460, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PORTFOLIO OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 PS5 — Portfolio Optimization":
    from scipy.optimize import minimize

    st.markdown('<div class="section-title">💼 PS5 — Portfolio Optimization (Modern Portfolio Theory)</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Find optimal stock weights using Monte Carlo simulation on Efficient Frontier.")

    top_companies = df.groupby('Company')['Daily_Return'].count().nlargest(15).index.tolist()
    selected = st.multiselect("Select Stocks (pick 4–8)", top_companies, default=top_companies[:6])
    n_sim = st.slider("Monte Carlo Simulations", 500, 3000, 1000, step=500)

    if len(selected) < 2:
        st.warning("Please select at least 2 stocks.")
    else:
        @st.cache_data(show_spinner="Running Monte Carlo...")
        def run_portfolio(selected, n_sim):
            pivot = df[df['Company'].isin(selected)].pivot_table(
                index='Date', columns='Company', values='Daily_Return')
            pivot = pivot.dropna()
            mu = pivot.mean().values * 252
            cov = pivot.cov().values * 252
            n = len(selected)
            port_ret, port_vol, port_sr, port_w = [], [], [], []
            np.random.seed(42)
            for _ in range(n_sim):
                w = np.random.dirichlet(np.ones(n))
                r = w @ mu
                v = np.sqrt(w @ cov @ w)
                port_ret.append(r); port_vol.append(v)
                port_sr.append(r / v); port_w.append(w)
            idx_max = np.argmax(port_sr)
            best_w = port_w[idx_max]
            return port_ret, port_vol, port_sr, best_w

        port_ret, port_vol, port_sr, best_w = run_portfolio(tuple(selected), n_sim)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Efficient Frontier (Monte Carlo)")
            fig = go.Figure(go.Scatter(
                x=port_vol, y=port_ret, mode='markers',
                marker=dict(color=port_sr, colorscale='Viridis', size=3,
                            colorbar=dict(title='Sharpe')),
                text=[f"SR={s:.2f}" for s in port_sr]
            ))
            best_idx = np.argmax(port_sr)
            fig.add_trace(go.Scatter(
                x=[port_vol[best_idx]], y=[port_ret[best_idx]],
                mode='markers', marker=dict(color='red', size=14, symbol='star'),
                name='Max Sharpe'
            ))
            fig.update_layout(template='plotly_dark', height=400,
                              xaxis_title='Annual Volatility', yaxis_title='Annual Return')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Optimal Portfolio Weights (Max Sharpe)")
            w_df = pd.DataFrame({'Stock': selected, 'Weight': best_w}).sort_values('Weight', ascending=False)
            fig = px.pie(w_df, names='Stock', values='Weight',
                         color_discrete_sequence=px.colors.qualitative.Set2, template='plotly_dark')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Annual Return", f"{port_ret[best_idx]*100:.2f}%")
        c2.metric("Best Annual Volatility", f"{port_vol[best_idx]*100:.2f}%")
        c3.metric("Max Sharpe Ratio", f"{port_sr[best_idx]:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — VOLATILITY FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌪️ PS6 — Volatility Forecasting":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    st.markdown('<div class="section-title">🌪️ PS6 — Stock Volatility Forecasting</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Predict future 7-day volatility (risk) using rolling statistics.")

    company = st.selectbox("Select Company", sorted(df['Company'].unique()), key='ps6')

    @st.cache_data(show_spinner="Training volatility models...")
    def run_ps6(company):
        d = df[df['Company']==company].sort_values('Date').copy()
        d['Vol7']  = d['Daily_Return'].rolling(7).std()
        d['Vol14'] = d['Daily_Return'].rolling(14).std()
        d['Vol30'] = d['Daily_Return'].rolling(30).std()
        d['MA5']   = d['Close'].rolling(5).mean()
        d['MA10']  = d['Close'].rolling(10).mean()
        d['Lag1_Vol'] = d['Vol7'].shift(1)
        d['Lag2_Vol'] = d['Vol7'].shift(2)
        d['Target_Vol'] = d['Vol7'].shift(-7)
        d = d.dropna()
        feats = ['Vol7','Vol14','Vol30','MA5','MA10','Lag1_Vol','Lag2_Vol','Price_Range','BuySell_Ratio']
        X = d[feats]; y = d['Target_Vol']
        Xtr,Xts,ytr,yts = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xts_s = sc.transform(Xts)
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(Xtr_s, ytr); rf_preds = rf.predict(Xts_s)
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr_s, ytr); ridge_preds = ridge.predict(Xts_s)
        return (yts.values, rf_preds, ridge_preds,
                r2_score(yts, rf_preds), r2_score(yts, ridge_preds),
                mean_absolute_error(yts, rf_preds), d['Date'].values, d['Vol7'].values)

    y_true, rf_p, ridge_p, rf_r2, ridge_r2, rf_mae, dates, vol7 = run_ps6(company)

    c1, c2 = st.columns(2)
    c1.metric("Random Forest R²", f"{rf_r2:.4f}", f"MAE={rf_mae:.4f}")
    c2.metric("Ridge Regression R²", f"{ridge_r2:.4f}")

    st.markdown("#### Historical 7-Day Rolling Volatility")
    fig = go.Figure(go.Scatter(x=dates, y=vol7, line=dict(color='#FF9800', width=1.5)))
    fig.update_layout(template='plotly_dark', height=320,
                      xaxis_title='Date', yaxis_title='7-Day Volatility (%)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Actual vs Predicted Volatility (Random Forest)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:150], name='Actual', line=dict(color='steelblue', width=2)))
    fig.add_trace(go.Scatter(y=rf_p[:150], name='RF Predicted', line=dict(color='tomato', width=2, dash='dash')))
    fig.add_trace(go.Scatter(y=ridge_p[:150], name='Ridge Predicted', line=dict(color='#4CAF50', width=1.5, dash='dot')))
    fig.update_layout(template='plotly_dark', height=350,
                      xaxis_title='Sample', yaxis_title='Volatility')
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 PS7 — Anomaly Detection":
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    st.markdown('<div class="section-title">🔍 PS7 — Anomaly Detection in Stock Market Data</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Detect unusual trading days using Isolation Forest and Local Outlier Factor.")

    contamination = st.slider("Expected Anomaly Fraction", 0.01, 0.10, 0.03, step=0.01)

    @st.cache_data(show_spinner="Running anomaly detection...")
    def run_ps7(contamination):
        d = df.copy()
        feats = ['Daily_Return','Volume','Price_Range','BuySell_Ratio']
        X = d[feats].dropna()
        idx = X.index
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        iso = IsolationForest(contamination=contamination, random_state=42)
        d.loc[idx, 'ISO_Anomaly'] = iso.fit_predict(Xs)
        lof = LocalOutlierFactor(contamination=contamination)
        d.loc[idx, 'LOF_Anomaly'] = lof.fit_predict(Xs)
        return d.dropna(subset=['ISO_Anomaly','LOF_Anomaly'])

    d7 = run_ps7(contamination)
    iso_n = (d7['ISO_Anomaly'] == -1).sum()
    lof_n = (d7['LOF_Anomaly'] == -1).sum()

    c1, c2 = st.columns(2)
    c1.metric("Isolation Forest Anomalies", iso_n)
    c2.metric("Local Outlier Factor Anomalies", lof_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Isolation Forest — Anomalies on Return vs Volume")
        sample = d7.sample(min(4000, len(d7)), random_state=42)
        fig = px.scatter(sample, x='Daily_Return', y='Volume',
                         color=sample['ISO_Anomaly'].map({1:'Normal',-1:'Anomaly'}),
                         color_discrete_map={'Normal':'steelblue','Anomaly':'red'},
                         opacity=0.5, template='plotly_dark')
        fig.update_layout(height=380, xaxis_title='Daily Return (%)', yaxis_title='Volume')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### LOF — Anomalies on Price Range vs Return")
        fig = px.scatter(sample, x='Price_Range', y='Daily_Return',
                         color=sample['LOF_Anomaly'].map({1:'Normal',-1:'Anomaly'}),
                         color_discrete_map={'Normal':'steelblue','Anomaly':'orange'},
                         opacity=0.5, template='plotly_dark')
        fig.update_layout(height=380, xaxis_title='Price Range', yaxis_title='Daily Return (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Anomalies Over Time")
    anom_time = d7[d7['ISO_Anomaly']==-1].groupby('Date').size().reset_index(name='Anomaly_Count')
    fig = go.Figure(go.Bar(x=anom_time['Date'], y=anom_time['Anomaly_Count'], marker_color='red'))
    fig.update_layout(template='plotly_dark', height=320,
                      xaxis_title='Date', yaxis_title='Anomaly Count')
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8 — TREND CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 PS8 — Trend Classification":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    st.markdown('<div class="section-title">📊 PS8 — Stock Market Trend Classification (Multi-Class)</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Classify each day as BULLISH / SIDEWAYS / BEARISH.")

    @st.cache_data(show_spinner="Training trend classifiers...")
    def run_ps8():
        d = df.copy().sort_values(['Company','Date']).reset_index(drop=True)
        d['Lag1_Return'] = d.groupby('Company')['Daily_Return'].shift(1)
        d['MA5']  = d.groupby('Company')['Close'].transform(lambda x: x.rolling(5).mean())
        d['MA10'] = d.groupby('Company')['Close'].transform(lambda x: x.rolling(10).mean())
        d['Vol5'] = d.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(5).std())
        def label(r):
            if r > 1: return 'BULLISH'
            elif r < -1: return 'BEARISH'
            else: return 'SIDEWAYS'
        d['Trend'] = d['Daily_Return'].apply(label)
        d = d.dropna()
        le = LabelEncoder(); d['Trend_Enc'] = le.fit_transform(d['Trend'])
        feats = ['Open','High','Low','Close','Volume','Lag1_Return',
                 'MA5','MA10','Vol5','Price_Range','BuySell_Ratio']
        X = d[feats]; y = d['Trend_Enc']
        Xtr,Xts,ytr,yts = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xts_s = sc.transform(Xts)
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        dt = DecisionTreeClassifier(max_depth=6, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=7)
        rf.fit(Xtr_s, ytr);  rf_p  = rf.predict(Xts_s)
        dt.fit(Xtr_s, ytr);  dt_p  = dt.predict(Xts_s)
        knn.fit(Xtr_s, ytr); knn_p = knn.predict(Xts_s)
        rf_rep = classification_report(yts, rf_p, target_names=le.classes_, output_dict=True)
        rf_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
        cm_rf = confusion_matrix(yts, rf_p)
        dist = d['Trend'].value_counts().reset_index()
        dist.columns = ['Trend','Count']
        return rf_rep, rf_imp, cm_rf, le.classes_, dist

    rf_rep, rf_imp, cm_rf, classes, dist = run_ps8()

    c1, c2, c3 = st.columns(3)
    c1.metric("BULLISH F1", f"{rf_rep.get('BULLISH',{}).get('f1-score',0):.3f}")
    c2.metric("SIDEWAYS F1", f"{rf_rep.get('SIDEWAYS',{}).get('f1-score',0):.3f}")
    c3.metric("BEARISH F1", f"{rf_rep.get('BEARISH',{}).get('f1-score',0):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Trend Label Distribution")
        fig = px.pie(dist, names='Trend', values='Count', template='plotly_dark',
                     color_discrete_map={'BULLISH':'#4CAF50','SIDEWAYS':'#FF9800','BEARISH':'#F44336'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Confusion Matrix (Random Forest)")
        fig = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues',
                        x=classes, y=classes,
                        labels=dict(x='Predicted', y='Actual'))
        fig.update_layout(template='plotly_dark', height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature Importance")
    fig = px.bar(rf_imp.reset_index(), x='index', y=0, color=0,
                 color_continuous_scale='Blues', template='plotly_dark',
                 labels={'index':'Feature', '0':'Importance'})
    fig.update_layout(height=350, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 9 — WAR PERIOD IMPACT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ PS9 — War Period Impact":
    from scipy import stats

    st.markdown('<div class="section-title">⚔️ PS9 — War Period Impact Analysis (Statistical Testing)</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Statistically prove whether war had a significant impact on stock returns.")

    pre  = df[df['Period']=='Pre-War']['Daily_Return'].dropna()
    post = df[df['Period']=='Post-War']['Daily_Return'].dropna()

    t_stat, p_t = stats.ttest_ind(pre, post, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(pre, post, alternative='two-sided')
    cohens_d = (post.mean() - pre.mean()) / np.sqrt((pre.std()**2 + post.std()**2) / 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pre-War Avg Return", f"{pre.mean():.4f}%")
    c2.metric("Post-War Avg Return", f"{post.mean():.4f}%")
    c3.metric("T-Test p-value", f"{p_t:.5f}", "Significant" if p_t < 0.05 else "Not Significant")
    c4.metric("Cohen's D (Effect)", f"{cohens_d:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Return Distribution: Pre vs Post War")
        fig = go.Figure()
        fig.add_trace(go.Box(y=pre.sample(min(5000,len(pre)), random_state=1),
                             name='Pre-War', marker_color='#2196F3', boxpoints=False))
        fig.add_trace(go.Box(y=post.sample(min(5000,len(post)), random_state=1),
                             name='Post-War', marker_color='#FF5722', boxpoints=False))
        fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
        fig.update_layout(template='plotly_dark', height=380, yaxis_title='Daily Return (%)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Violin Plot — Return Shape")
        fig = go.Figure()
        fig.add_trace(go.Violin(y=pre.sample(min(5000,len(pre)), random_state=1),
                                name='Pre-War', line_color='#2196F3', box_visible=True))
        fig.add_trace(go.Violin(y=post.sample(min(5000,len(post)), random_state=1),
                                name='Post-War', line_color='#FF5722', box_visible=True))
        fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
        fig.update_layout(template='plotly_dark', height=380, yaxis_title='Daily Return (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Return by Country: Pre vs Post War")
    grp = df.groupby(['Country','Period'])['Daily_Return'].mean().reset_index()
    fig = px.bar(grp, x='Country', y='Daily_Return', color='Period', barmode='group',
                 color_discrete_map={'Pre-War':'#2196F3','Post-War':'#FF5722'},
                 template='plotly_dark', labels={'Daily_Return':'Avg Daily Return (%)'})
    fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=0.8)
    fig.update_layout(height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Statistical Test Summary"):
        st.markdown(f"""
| Test | Statistic | P-Value | Result |
|------|-----------|---------|--------|
| T-Test (Welch) | {t_stat:.4f} | {p_t:.6f} | {'✅ Significant' if p_t < 0.05 else '❌ Not Significant'} |
| Mann-Whitney U | {u_stat:.0f} | {p_u:.6f} | {'✅ Significant' if p_u < 0.05 else '❌ Not Significant'} |
| Cohen's D | {cohens_d:.4f} | — | {'Large' if abs(cohens_d)>0.8 else 'Medium' if abs(cohens_d)>0.5 else 'Small'} effect |
        """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 10 — SECTOR ROTATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔄 PS10 — Sector Rotation":
    st.markdown('<div class="section-title">🔄 PS10 — Sector Rotation Strategy using Return Momentum</div>', unsafe_allow_html=True)
    st.markdown("**Goal:** Rank sectors by 3-month momentum and backtest a rotation strategy vs buy-all benchmark.")

    @st.cache_data(show_spinner="Running backtest...")
    def run_ps10():
        d = df.copy()
        d['YearMonth'] = d['Date'].dt.to_period('M')
        ms = d.groupby(['YearMonth','Sector'])['Daily_Return'].mean().reset_index()
        ms.columns = ['YearMonth','Sector','Monthly_Return']
        ms['Date'] = ms['YearMonth'].dt.to_timestamp()
        ms = ms.sort_values(['Sector','Date']).reset_index(drop=True)
        ms['Momentum_3m'] = ms.groupby('Sector')['Monthly_Return'].transform(lambda x: x.rolling(3).mean())
        ms['Momentum_Rank'] = ms.groupby('YearMonth')['Momentum_3m'].rank(ascending=False, method='dense')
        ms = ms.dropna()
        months = sorted(ms['YearMonth'].unique())
        strat_rets, bench_rets, dates_l = [], [], []
        for i in range(3, len(months)-1):
            cur, nxt = months[i], months[i+1]
            cur_d = ms[ms['YearMonth']==cur]
            nxt_d = ms[ms['YearMonth']==nxt]
            top3 = cur_d.nsmallest(3,'Momentum_Rank')['Sector'].tolist()
            top3_nxt = nxt_d[nxt_d['Sector'].isin(top3)]['Monthly_Return']
            if len(top3_nxt) > 0:
                strat_rets.append(top3_nxt.mean())
                bench_rets.append(nxt_d['Monthly_Return'].mean())
                dates_l.append(cur.to_timestamp())
        strat = pd.Series(strat_rets, index=dates_l)
        bench = pd.Series(bench_rets, index=dates_l)
        cum_strat = (1 + strat/100).cumprod()
        cum_bench = (1 + bench/100).cumprod()
        return strat, bench, cum_strat, cum_bench, ms

    strat, bench, cum_strat, cum_bench, ms = run_ps10()

    c1, c2, c3 = st.columns(3)
    c1.metric("Strategy Avg Monthly Return", f"{strat.mean():.4f}%")
    c2.metric("Benchmark Avg Monthly Return", f"{bench.mean():.4f}%")
    c3.metric("Outperformance", f"{(strat.mean()-bench.mean()):.4f}%",
              "↑ Strategy Wins" if strat.mean() > bench.mean() else "↓ Benchmark Wins")

    st.markdown("#### Cumulative Returns: Strategy vs Benchmark")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat.values,
                             name='Sector Rotation Strategy', line=dict(color='#4CAF50', width=2.5)))
    fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values,
                             name='Buy All Sectors (Benchmark)', line=dict(color='steelblue', width=2.5, dash='dash')))
    fig.add_hline(y=1, line_dash='dot', line_color='white', line_width=0.8)
    fig.update_layout(template='plotly_dark', height=380,
                      xaxis_title='Date', yaxis_title='Cumulative Return (1 = start)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sector Momentum Heatmap (Last 18 Months)")
    pivot_hm = ms.pivot_table(index='YearMonth', columns='Sector', values='Momentum_3m', aggfunc='mean')
    pivot_hm = pivot_hm.tail(18)
    top_sec = pivot_hm.std().nlargest(8).index
    pivot_hm = pivot_hm[top_sec]
    pivot_hm.index = [str(x) for x in pivot_hm.index]
    fig = px.imshow(pivot_hm, color_continuous_scale='RdYlGn', aspect='auto',
                    text_auto='.2f', template='plotly_dark',
                    labels=dict(color='Momentum'))
    fig.update_layout(height=480, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly Strategy vs Benchmark Returns")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strat.index, y=strat.values, name='Strategy', marker_color='#4CAF50', opacity=0.8))
    fig.add_trace(go.Bar(x=bench.index, y=bench.values, name='Benchmark', marker_color='steelblue', opacity=0.8))
    fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=0.8)
    fig.update_layout(template='plotly_dark', height=340, barmode='group',
                      xaxis_title='Month', yaxis_title='Monthly Return (%)')
    st.plotly_chart(fig, use_container_width=True)
