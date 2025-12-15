import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="OMÜ MatrixLab Web", page_icon="🧪", layout="centered")

# --- CSS: LOGO VE TASARIM DÜZELTMELERİ ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* Font Ayarları */
        h1, h2, h3, h4, p, div {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        }
        
        /* Ana Başlık */
        div[data-testid="column"] h3 {
             color: #1B2631; 
             font-weight: 800 !important;
             font-size: 2rem !important;
             margin-bottom: 0px !important;
        }

        /* --- LOGO DÜZELTME (Final) --- */
        /* Logoyu kapsayan kutuya üstten boşluk ver ve resmi sığdır */
        div[data-testid="stImage"] {
            padding-top: 20px !important; /* Yukarıdan aşağı it */
        }
        [data-testid="stImage"] > img {
            object-fit: contain !important; 
            max-height: 150px; 
            width: auto !important; 
            margin: auto; 
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1.2, 4.8])
with col1:
    if os.path.exists("omu_logo.png"):
        logo = Image.open("omu_logo.png")
        st.image(logo)
    else:
        st.write("🧪")
with col2:
    st.markdown("### OMÜ Kimya Mühendisliği")
    st.caption("Lineer Cebir Analiz ve Çözüm Sistemi")

# --- TANITIM METNİ (DÜZELTİLDİ: Ax=B) ---
st.markdown("""
    <div style="background-color:#F8F9F9; padding:15px; border-radius:10px; border-left: 5px solid #2E86C1; margin-top:10px; font-size:15px; color:#424949; line-height: 1.5;">
        <strong>Hakkında:</strong> Bu yazılım, <strong>Ondokuz Mayıs Üniversitesi Kimya Mühendisliği</strong> laboratuvar ve proje çalışmalarında 
        karşılaşılan lineer denklem sistemlerinin (Ax = B) çözümü için geliştirilmiştir. 
        <br><br>
        Mühendislik problemlerinin çözümünde kullanılan nümerik yöntemleri uygular. 
        Hesaplama süreçlerini şeffaf bir şekilde gösterir ve sonuçları grafik destekli <strong>Excel raporlarına</strong> dönüştürür.
    </div>
""", unsafe_allow_html=True)

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

# --- MATEMATİK MOTORU ---
def forward_sub(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)): y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def back_sub(U, y):
    x = np.zeros_like(y)
    for i in range(len(y)-1, -1, -1): x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# --- GİRİŞ ALANI ---
st.write("") 
tab1, tab2 = st.tabs(["🟦 Matris A (Katsayılar)", "🟧 Vektör B (Sonuçlar)"])

if 'n_prev' not in st.session_state or st.session_state.n_prev != n:
    index_labels = list(range(1, n + 1))
    st.session_state.df_a = pd.DataFrame(np.zeros((n, n)), index=index_labels, columns=index_labels)
    st.session_state.df_b = pd.DataFrame(np.zeros((n, 1)), index=index_labels, columns=["Değer"])
    st.session_state.n_prev = n

with tab1:
    matrix_a = st.data_editor(st.session_state.df_a, key="editor_a", use_container_width=True)

with tab2:
    vector_b = st.data_editor(st.session_state.df_b, key="editor_b", use_container_width=True)

st.write("")
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

        # --- SONUÇLAR ---
        st.divider()
        st.success("✅ Çözüm Tamamlandı")
        
        res_tab1, res_tab2 = st.tabs(["📊 Tablo & Excel", "📑 İşlem Kayıtları"])
        
        with res_tab1:
            df_res = pd.DataFrame({"Bilinmeyen": [f"x{i+1}" for i in range(n)], "Hesaplanan": x})
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Sonuclar')
                
            st.download_button(
                label="📥 Sonuçları Excel Olarak İndir",
                data=buffer.getvalue(),
                file_name="OMU_Cozum_Raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.write("**Değer Dağılımı:**")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.bar(df_res["Bilinmeyen"], df_res["Hesaplanan"], color="#2980B9")
            st.pyplot(fig, use_container_width=True)

        with res_tab2:
            for m in msg: st.code(m)
            
    except Exception as e:
        st.error(f"Hata: {e}")
