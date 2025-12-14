import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- SAYFA AYARLARI ---
# layout="centered" yaparak mobilde içeriğin ortalanmasını sağlıyoruz
st.set_page_config(page_title="OMÜ MatrixLab Web", page_icon="🧪", layout="centered")

# --- CSS İLE MOBİL İYİLEŞTİRMELERİ ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            font-size: 1.8rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER (LOGO & BAŞLIK) ---
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("omu_logo.png"):
        logo = Image.open("omu_logo.png")
        st.image(logo, use_container_width=True)
    else:
        st.write("🧪")
with col2:
    st.markdown("### OMÜ Kimya Mühendisliği")
    st.caption("Lineer Cebir Analiz Sistemi")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    n = st.number_input("Matris Boyutu (N)", 2, 10, 3)
    method = st.selectbox("Yöntem", [
        "LU Doolittle", "Cholesky", "Gauss Yok Etme", 
        "Cramer", "Jacobi", "Gauss-Seidel"
    ])
    st.divider()
    tol = st.text_input("Tolerans", "0.0001")
    max_it = st.number_input("Max İter.", 100)

# --- MATEMATİK FONKSİYONLARI ---
def forward_sub(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)): y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def back_sub(U, y):
    x = np.zeros_like(y)
    for i in range(len(y)-1, -1, -1): x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# --- MOBİL UYUMLU GİRİŞ ALANI (TABS) ---
st.write("---")
st.info("Aşağıdaki sekmeleri kullanarak verileri giriniz.")

# MOBİL ÇÖZÜM BURADA: Tabs (Sekmeler) kullanıyoruz
tab1, tab2 = st.tabs(["🟦 Matris A (Katsayılar)", "🟧 Vektör B (Sonuçlar)"])

# Matrislerin boyutlarını N değiştikçe sıfırlıyoruz
if 'n_prev' not in st.session_state or st.session_state.n_prev != n:
    st.session_state.df_a = pd.DataFrame(np.zeros((n, n)))
    st.session_state.df_b = pd.DataFrame(np.zeros((n, 1)), columns=["Değer"])
    st.session_state.n_prev = n

with tab1:
    st.write(f"**{n}x{n} Katsayılar Matrisi**")
    # use_container_width=True telefonda tabloyu ekrana yayar
    matrix_a = st.data_editor(st.session_state.df_a, key="editor_a", use_container_width=True)

with tab2:
    st.write("**Sonuç Vektörü**")
    vector_b = st.data_editor(st.session_state.df_b, key="editor_b", use_container_width=True)

st.write("") # Boşluk
if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True, type="primary"):
    try:
        A = matrix_a.to_numpy()
        B = vector_b.to_numpy().flatten()
        msg = []

        if method == "LU Doolittle":
            L = np.eye(n); U = np.zeros((n, n))
            for i in range(n):
                for k in range(i, n): U[i, k] = A[i, k] - np.dot(L[i, :i], U[:i, k])
                for k in range(i+1, n): L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]
            x = back_sub(U, forward_sub(L, B))
            msg = [f"L Matrisi:\n{L}", f"U Matrisi:\n{U}"]
        
        elif method == "Cholesky":
            L = np.linalg.cholesky(A)
            x = back_sub(L.T, forward_sub(L, B))
            msg = [f"L Matrisi:\n{L}"]
        
        else:
            x = np.linalg.solve(A, B)
            msg = ["Standart çözüm uygulandı."]

        # --- SONUÇ EKRANI ---
        st.divider()
        st.success("✅ Çözüm Tamamlandı")
        
        # Sonuçları da sekmeli gösterelim ki telefonda uzamasın
        res_tab1, res_tab2 = st.tabs(["📊 Sonuç Tablosu", "📑 İşlem Kayıtları"])
        
        with res_tab1:
            df_res = pd.DataFrame({"Bilinmeyen": [f"x{i+1}" for i in range(n)], "Hesaplanan": x})
            st.dataframe(df_res, use_container_width=True)
            
            # Grafik
            fig, ax = plt.subplots(figsize=(4, 3)) # Mobilde küçük grafik
            ax.bar(df_res["Bilinmeyen"], df_res["Hesaplanan"], color="#2980B9")
            ax.set_title("Sonuç Dağılımı")
            st.pyplot(fig, use_container_width=True)

        with res_tab2:
            for m in msg: st.code(m)
            
    except Exception as e:
        st.error(f"Hata: {e}")
