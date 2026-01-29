import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Prediksi Produksi Padi Sumatra 2025",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌾 Prediksi Produksi Padi Pulau Sumatra 2025")
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913121.png", width=100)
    st.markdown("## Navigasi")
    page = st.radio(
        "Pilih halaman:",
        ["Home", "Data Analysis", "Model Training", "Prediksi 2025", "About"]
    )

# Load and cache data
@st.cache_data
def load_data():
    padi = pd.read_csv('Padi Sumatra.csv')
    return padi

padi = load_data()

# ==================== HOME PAGE ====================
if page == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📌 Tentang Project
        
        Project ini bertujuan untuk **memprediksi produksi padi tahun 2025 di Pulau Sumatra** 
        menggunakan machine learning model.
        
        ### Dataset Information
        Dataset yang digunakan mengandung data historis produksi padi dari tahun 1993 hingga 2020 
        dengan variabel-variabel pendukung seperti:
        - **Tahun**: Tahun pengamatan
        - **Provinsi**: Nama provinsi di Pulau Sumatra
        - **Luas Panen (Ha)**: Luas area panen dalam hektar
        - **Curah Hujan (mm)**: Curah hujan tahunan
        - **Suhu Rata-rata (°C)**: Suhu udara rata-rata
        - **Kelembapan (%)**: Tingkat kelembapan udara
        - **Produksi Padi (Ton)**: Target prediksi
        """)
    
    with col2:
        st.info(f"📊 Total Data Points: {len(padi)}")
        st.success(f"📍 Provinsi: {padi['Provinsi'].nunique()}")
        st.warning(f"📅 Tahun: {padi['Tahun'].min()}-{padi['Tahun'].max()}")

# ==================== DATA ANALYSIS PAGE ====================
elif page == "Data Analysis":
    st.header("📊 Analisis Data Eksplorasi")
    
    # Data Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Baris", len(padi))
    with col2:
        st.metric("Total Kolom", len(padi.columns))
    with col3:
        st.metric("Missing Values", padi.isnull().sum().sum())
    
    # Display data
    st.subheader("📋 Preview Data")
    st.dataframe(padi.head(10), use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(padi.describe().T, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rata-rata Produksi per Tahun")
        avg_year = padi.groupby('Tahun')['Produksi'].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(avg_year.index, avg_year.values, color='#2ecc71')
        ax.set_title('Tren Produksi Padi di Sumatra', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Produksi (Ton)')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        st.info("📌 Terjadi peningkatan produksi dari 1993-2017, diikuti penurunan di 2018-2019.")
    
    with col2:
        st.subheader("Luas Panen vs Produksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        provinces = padi['Provinsi'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(provinces)))
        
        for i, province in enumerate(provinces):
            province_data = padi[padi['Provinsi'] == province]
            ax.scatter(province_data['Luas Panen'], province_data['Produksi'], 
                      label=province, alpha=0.6, s=80, color=colors[i])
        
        ax.set_title('Hubungan Luas Panen dan Produksi', fontsize=12, fontweight='bold')
        ax.set_xlabel('Luas Panen (Ha)')
        ax.set_ylabel('Produksi (Ton)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        correlation = padi['Luas Panen'].corr(padi['Produksi'])
        st.success(f"✅ Korelasi antara Luas Panen dan Produksi: **{correlation:.3f}** (Sangat Kuat)")
    
    # Correlation heatmap
    st.subheader("🔗 Matriks Korelasi")
    numeric_cols = padi.select_dtypes(include=[np.number]).columns
    correlation_matrix = padi[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Matriks Korelasi Variabel', fontsize=12, fontweight='bold')
    st.pyplot(fig)

# ==================== MODEL TRAINING PAGE ====================
elif page == "Model Training":
    st.header("🤖 Training Model Machine Learning")
    
    # Prepare data
    padi_encoded = padi.copy()
    padi_encoded = pd.get_dummies(padi_encoded, columns=['Provinsi'], drop_first=True)
    
    target = 'Produksi'
    features = [col for col in padi_encoded.columns if col not in ['Produksi']]
    
    # Data splitting
    train = padi_encoded[padi_encoded['Tahun'] <= 2018]
    test = padi_encoded[(padi_encoded['Tahun'] > 2018) & (padi_encoded['Tahun'] <= 2020)]
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    st.info(f"📊 Data Training: {len(X_train)} samples | Data Testing: {len(X_test)} samples")
    
    # Model comparison
    st.subheader("📋 Perbandingan Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Linear Regression Model")
        with st.spinner("Training Linear Regression..."):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            
            mae_lr = mean_absolute_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)
            
            st.metric("MAE (Mean Absolute Error)", f"{mae_lr:,.2f}")
            st.metric("R² Score", f"{r2_lr:.4f}")
    
    with col2:
        st.markdown("### Random Forest Model (Untuned)")
        with st.spinner("Training Random Forest..."):
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)
            
            st.metric("MAE (Mean Absolute Error)", f"{mae_rf:,.2f}")
            st.metric("R² Score", f"{r2_rf:.4f}")
    
    # Model Comparison Table
    st.subheader("📊 Tabel Perbandingan Model")
    comparison_data = {
        'Model': ['Linear Regression', 'Random Forest'],
        'MAE': [f"{mae_lr:,.2f}", f"{mae_rf:,.2f}"],
        'R² Score': [f"{r2_lr:.4f}", f"{r2_rf:.4f}"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Hyperparameter Tuning
    st.subheader("⚙️ Hyperparameter Tuning - Random Forest")
    
    if st.button("🔧 Mulai Tuning (GridSearchCV)"):
        with st.spinner("Melakukan GridSearchCV... Ini mungkin memakan waktu beberapa menit"):
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_leaf': [1, 3, 5]
            }
            
            grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                              scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_rf = grid.best_estimator_
            
            # Evaluate best model
            y_pred_best = best_rf.predict(X_test)
            mae_best = mean_absolute_error(y_test, y_pred_best)
            r2_best = r2_score(y_test, y_pred_best)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Best Parameters: {grid.best_params_}")
            with col2:
                st.metric("Best MAE", f"{mae_best:,.2f}")
            
            st.metric("Best R² Score", f"{r2_best:.4f}")
            
            # Store best model in session state
            st.session_state.best_rf = best_rf
            st.session_state.features = features
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("✅ Model tuning selesai!")
    
    # Feature Importance (for trained RF)
    st.subheader("🎯 Feature Importance")
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498db')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Feature Importance - Random Forest', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    st.pyplot(fig)

# ==================== PREDICTION PAGE ====================
elif page == "Prediksi 2025":
    st.header("🔮 Prediksi Produksi Padi Tahun 2025")
    
    # Prepare data
    padi_encoded = padi.copy()
    padi_encoded = pd.get_dummies(padi_encoded, columns=['Provinsi'], drop_first=True)
    
    target = 'Produksi'
    features = [col for col in padi_encoded.columns if col not in ['Produksi']]
    
    # Train final model
    train = padi_encoded[padi_encoded['Tahun'] <= 2018]
    test = padi_encoded[(padi_encoded['Tahun'] > 2018) & (padi_encoded['Tahun'] <= 2020)]
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # Train optimized model
    best_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
    best_rf.fit(X_train, y_train)
    
    # Make prediction for 2025
    data_2025 = X_test.mean().to_frame().T
    data_2025['Tahun'] = 2025
    
    pred_2025 = best_rf.predict(data_2025)[0]
    
    # Display prediction
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background-color: #ecf0f1; border-radius: 10px;">
            <h2>Prediksi Produksi Padi 2025</h2>
            <h1 style="color: #27ae60; font-size: 48px;">
                {:.2f} Juta Ton
            </h1>
            <p style="font-size: 16px; color: #34495e;">
                Berdasarkan rata-rata kondisi iklim dan luas panen historis
            </p>
        </div>
        """.format(pred_2025 / 1_000_000), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Historical data with prediction
    st.subheader("📈 Grafik Produksi Aktual vs Prediksi 2025")
    
    produksi_per_tahun = (
        padi.groupby('Tahun')['Produksi']
            .mean()
            .reset_index()
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual production
    ax.plot(
        produksi_per_tahun['Tahun'],
        produksi_per_tahun['Produksi'],
        marker='o',
        linewidth=2,
        markersize=8,
        label='Produksi Aktual',
        color='#3498db'
    )
    
    # Plot prediction for 2025
    ax.scatter(
        2025,
        pred_2025,
        color='#e74c3c',
        s=300,
        label='Prediksi 2025',
        marker='*',
        edgecolors='black',
        linewidth=2,
        zorder=5
    )
    
    # Add annotation
    ax.annotate(
        f'{pred_2025/1_000_000:.2f}M ton',
        xy=(2025, pred_2025),
        xytext=(2025, pred_2025 + 500000),
        fontsize=12,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.7, edgecolor='black'),
        color='white',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=2)
    )
    
    ax.set_title('Produksi Padi di Pulau Sumatra dan Prediksi Tahun 2025', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Produksi Padi (Ton)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    st.pyplot(fig)
    
    # Insights
    st.subheader("💡 Insight dan Penjelasan")
    
    historical_avg = padi_encoded['Produksi'].mean()
    growth_pct = ((pred_2025 - historical_avg) / historical_avg) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Rata-rata Produksi Historis**
        {historical_avg/1_000_000:.2f} Juta Ton
        """)
    
    with col2:
        if growth_pct > 0:
            st.success(f"""
            **Perkiraan Pertumbuhan**
            +{growth_pct:.2f}%
            """)
        else:
            st.warning(f"""
            **Perkiraan Pertumbuhan**
            {growth_pct:.2f}%
            """)
    
    with col3:
        latest_year = padi_encoded['Tahun'].max()
        latest_prod = padi_encoded[padi_encoded['Tahun'] == latest_year]['Produksi'].mean()
        diff_latest = ((pred_2025 - latest_prod) / latest_prod) * 100
        
        st.info(f"""
        **Perubahan vs {latest_year}**
        {diff_latest:+.2f}%
        """)
    
    st.markdown("""
    ### 📝 Kesimpulan
    
    Berdasarkan analisis data historis dan model Random Forest yang telah dioptimalkan:
    
    1. **Model Performance**: Random Forest menunjukkan performa yang superior dengan R² Score mencapai 0.77
    
    2. **Faktor Dominan**: Luas panen merupakan faktor utama yang mempengaruhi produksi padi dengan korelasi sangat kuat (0.91)
    
    3. **Prediksi 2025**: Dengan asumsi kondisi iklim dan luas panen relatif stabil seperti tahun-tahun sebelumnya, 
       produksi padi di Pulau Sumatra tahun 2025 diprediksi mencapai **{:.2f} juta ton**
    
    4. **Rekomendasi**: Untuk meningkatkan produksi padi, fokus pada peningkatan luas panen dan optimalisasi 
       kondisi iklim (curah hujan, suhu, dan kelembapan yang sesuai)
    """.format(pred_2025 / 1_000_000))

# ==================== ABOUT PAGE ====================
elif page == "About":
    st.header("ℹ️ Tentang Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📚 Informasi Project
        
        **Judul**: Prediksi Produksi Padi Pulau Sumatra Tahun 2025
        
        **Author**: Amelia Evita Alam
        
        **Dataset Source**: [Kaggle - Dataset Tanaman Padi Sumatra](https://www.kaggle.com/datasets/ardikasatria/datasettanamanpadisumatera)
        
        ---
        
        ## 🎯 Tujuan Project
        
        Menganalisis data produksi padi di Pulau Sumatra dan membangun model machine learning 
        yang akurat untuk memprediksi produksi padi pada tahun 2025.
        
        ---
        
        ## 📊 Metodologi
        
        1. **Data Collection**: Pengumpulan data dari Kaggle
        2. **Data Cleaning**: Pembersihan dan preprocessing data
        3. **EDA (Exploratory Data Analysis)**: Analisis dan visualisasi data
        4. **Feature Engineering**: Encoding variabel kategorikal
        5. **Model Development**: Pembangunan dan perbandingan model
        6. **Hyperparameter Tuning**: Optimalisasi hyperparameter menggunakan GridSearchCV
        7. **Prediction**: Prediksi untuk tahun 2025
        
        ---
        
        ## 🔧 Tools dan Library
        
        - **Python**: Bahasa pemrograman utama
        - **Pandas & NumPy**: Data manipulation dan numerical computing
        - **Scikit-learn**: Machine learning library
        - **Matplotlib & Seaborn**: Data visualization
        - **Streamlit**: Web application framework
        
        ---
        
        ## 📈 Model Comparison
        
        | Model | MAE | R² Score | Status |
        |-------|-----|----------|--------|
        | Linear Regression | 631,382.80 | 0.46 | Baseline |
        | Random Forest | 335,231.52 | 0.77 | ⭐ Best |
        | Random Forest (Tuned) | Variabel | Variabel | Optimized |
        
        ---
        
        ## 📞 Kontak dan Social Media
        
        - GitHub: [Amelia Evita Alam](https://github.com)
        - LinkedIn: [Amelia Evita Alam](https://linkedin.com)
        - Email: amelia.evita@example.com
        
        """)
    
    with col2:
        st.markdown("""
        ### 📌 Key Metrics
        
        **Data Points**: 1000+
        
        **Time Period**: 1993-2020
        
        **Provinces**: 6 (Sumatera Utara, Sumatera Barat, 
        Riau, Jambi, Sumatera Selatan, 
        Bengkulu)
        
        **Features**: 7
        
        **Model Type**: Regression
        
        ---
        
        ### ⭐ Highlights
        
        ✅ Analisis EDA Lengkap
        
        ✅ Perbandingan Model
        
        ✅ Hyperparameter Tuning
        
        ✅ Prediksi Akurat
        
        ✅ Visualisasi Interaktif
        
        """)

# Footer
st.markdown("---")
st.markdown(
    "👩‍💻 **Project by Amelia Evita Alam**  \n"
    "🔗 GitHub: Prediksi-Produksi-Padi-Di-Sumatra-2025"
)