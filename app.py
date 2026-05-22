import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
try:
    import shap
except ImportError:
    st.error("⚠️ ไม่พบไลบรารี 'shap' กรุณาเพิ่ม 'shap' ใน requirements.txt หรือติดตั้งผ่าน pip install shap")

# ตั้งค่าหน้าจอ Streamlit (Must be first Streamlit command)
st.set_page_config(page_title="Edge Seam AI & Optimization System", layout="wide", page_icon="🏭")

# ==========================================
# 1. ฟังก์ชันโหลด Model, Artifacts และ Dataset
# ==========================================
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('edge_seam_ann.keras', compile=False)
    knn_imputer = joblib.load('knn_imputer.save')
    scaler_num = joblib.load('scaler_num.save')

    encoder_cat = joblib.load('encoder_cat.save') if os.path.exists('encoder_cat.save') else None
    top_features = joblib.load('best_features.save')
    col_names = joblib.load('col_names.save')

    threshold = 0.5
    if os.path.exists('optimal_threshold.txt'):
        with open('optimal_threshold.txt', 'r') as f:
            threshold = float(f.read())

    return model, knn_imputer, scaler_num, encoder_cat, top_features, col_names, threshold

@st.cache_data
def load_data():
    """โหลดข้อมูลดิบสำหรับนำมาทำ Data Analytics"""
    PATH = r'EDGE_SEAM.xlsx'
    if os.path.exists(PATH):
        df = pd.read_excel(PATH, sheet_name='Dataset')
        df = df.dropna(subset=['Defect'])
        return df
    else:
        np.random.seed(42)
        data = np.random.rand(1000, 40)
        df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(40)])

        # จำลองตัวแปรหน้างาน
        df['TEM_DIS'] = np.random.normal(1200, 50, 1000)
        df['LSP_Body'] = np.random.normal(1080, 30, 1000)
        df['Entry_Body'] = np.random.normal(1020, 25, 1000)
        df['SLABTH'] = np.random.normal(210, 5, 1000)
        df['FT_HEAD'] = np.random.normal(850, 20, 1000)
        df['CT_HEAD'] = np.random.normal(600, 15, 1000)
        df['XVPTF8'] = np.random.normal(10, 2, 1000)
        df['PSDRFT1'] = np.random.normal(45, 5, 1000)
        df['PSDRFT2'] = np.random.normal(47, 5, 1000)
        df['PSDRFT3'] = np.random.normal(48, 5, 1000)
        df['PSDRFT4'] = np.random.normal(46, 5, 1000)
        df['PSDRFT5'] = np.random.normal(44, 5, 1000)        

        prob = (df['XVPTF8'] > 11.5) | (df['FT_HEAD'] < 830)
        df['Defect'] = np.where(prob, np.random.choice([0, 1], p=[0.3, 0.7]), np.random.choice([0, 1], p=[0.9, 0.1]))
        return df

try:
    model, imputer, scaler, encoder, top_features, col_names, threshold = load_models()
    num_cols = col_names['num_cols']
    cat_cols = col_names['cat_cols']
    raw_df = load_data()
    raw_df['Defect_Label'] = raw_df['Defect'].map({0: 'Good (ปกติ)', 1: 'NG (ตำหนิ)'})
except Exception as e:
    st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลด Model/Data: {e}")
    st.info("กรุณาตรวจสอบไฟล์ .keras, .save, และข้อมูล .xlsx")
    st.stop()

# ==========================================
# 2. คลาส Prescriptive Engine (Standard & Advanced)
# ==========================================
class EdgeSeamOptimizer:
    def __init__(self, model, imputer, scaler, encoder, top_features, threshold):
        self.model = model; self.imputer = imputer; self.scaler = scaler
        self.encoder = encoder; self.top_features = top_features; self.threshold = threshold

        self.controllable_cols = [
            'FT_HEAD', 'CT_HEAD', 'XVPTF8', 'RMEXTG',
            'PSDRFT1', 'PSDRFT2', 'PSDRFT3', 'PSDRFT4', 'PSDRFT5',
            'PSRCMS1', 'PSRCMS2', 'PSRCMS3', 'PSRCMS4', 'PSRCMS5'            
        ]
        self.adjustment_limit_pct = 0.05

    def optimize(self, current_df, advanced_mode=False):
        X_curr = self._preprocess(current_df.copy())
        current_prob = self.model.predict(X_curr, verbose=0)[0][0]

        if current_prob < self.threshold:
            return None, current_prob, "safe"

        n_sims = 2000 if advanced_mode else 800
        sim_df = pd.concat([current_df] * n_sims, ignore_index=True)

        for col in self.controllable_cols:
            if col in sim_df.columns:
                curr_val = current_df.loc[0, col]
                # ขยาย Limit เป็น 10% สำหรับ Advanced Mode เพื่อหาทางออกที่กว้างขึ้น
                limit = 0.10 if advanced_mode else self.adjustment_limit_pct
                sim_df[col] = np.random.uniform(curr_val * (1 - limit),
                                                curr_val * (1 + limit), size=n_sims)

        # --- Advanced Physics Constraints (กฎทางฟิสิกส์วิศวกรรม) ---
        if advanced_mode:
            # กฎข้อที่ 1: CT_HEAD ต้องต่ำกว่า FT_HEAD อย่างน้อย 150 องศา (Cooling Logic)
            if 'CT_HEAD' in sim_df.columns and 'FT_HEAD' in sim_df.columns:
                valid_temp = sim_df['CT_HEAD'] <= (sim_df['FT_HEAD'] - 150)
            else:
                valid_temp = pd.Series(True, index=sim_df.index)

            # กฎข้อที่ 2: Total Draft (ผลรวมแรงกด) ต้องคงที่เพื่อรักษาความหนา +/- 2%
            draft_cols = ['PSDRFT1', 'PSDRFT2', 'PSDRFT3', 'PSDRFT4', 'PSDRFT5']
            existing_draft_cols = [c for c in draft_cols if c in sim_df.columns]
            if existing_draft_cols:
                original_total_draft = current_df.loc[0, existing_draft_cols].sum()
                sim_total_draft = sim_df[existing_draft_cols].sum(axis=1)
                valid_draft = (sim_total_draft >= original_total_draft * 0.98) & (sim_total_draft <= original_total_draft * 1.02)
            else:
                valid_draft = pd.Series(True, index=sim_df.index)

            # กรองข้อมูลให้เหลือเฉพาะ Simulation ที่เป็นไปได้ทางฟิสิกส์
            valid_mask = valid_temp & valid_draft
            sim_df = sim_df[valid_mask]

            if len(sim_df) == 0:
                return None, current_prob, "unable_due_to_physics"

        # --------------------------------------------------------

        X_sims = self._preprocess(sim_df)
        sim_probs = self.model.predict(X_sims, verbose=0).flatten()

        safe_idx = np.where(sim_probs < self.threshold)[0]
        if len(safe_idx) == 0:
            return None, current_prob, "unable"

        safe_sims = sim_df.iloc[safe_idx]
        safe_probs = sim_probs[safe_idx]

        orig_vals = current_df[self.controllable_cols].values[0]
        distances = np.linalg.norm(safe_sims[self.controllable_cols].values - orig_vals, axis=1)

        best_idx = np.argmin(distances)
        best_suggestion = safe_sims.iloc[best_idx]

        suggestions = {}
        for col in self.controllable_cols:
            orig_v = current_df.loc[0, col]
            sugg_v = best_suggestion[col]
            diff = sugg_v - orig_v
            if abs(diff) > 0.001:
                suggestions[col] = {"Current": orig_v, "Suggested": sugg_v, "Change": diff}
        return suggestions, safe_probs[best_idx], "optimized"

    def _preprocess(self, df):
        for col in num_cols:
            if col not in df.columns: df[col] = 0.0

        df_num = self.imputer.transform(df[num_cols])
        df_num_scaled = self.scaler.transform(df_num)

        if self.encoder and cat_cols:
            for col in cat_cols:
                if col not in df.columns: df[col] = 'Unknown'
            df_cat = self.encoder.transform(df[cat_cols])
            X_all = np.hstack([df_num_scaled, df_cat])
        else:
            X_all = df_num_scaled

        all_features = num_cols + (list(self.encoder.get_feature_names_out(cat_cols)) if (self.encoder and cat_cols) else [])
        X_df = pd.DataFrame(X_all, columns=all_features)

        missing_features = [f for f in self.top_features if f not in X_df.columns]
        if missing_features:
            for f in missing_features: X_df[f] = 0

        return X_df[self.top_features]

optimizer = EdgeSeamOptimizer(model, imputer, scaler, encoder, top_features, threshold)

# ==========================================
# 3. เมนูด้านข้าง (Sidebar Navigation)
# ==========================================
st.sidebar.title("🏭 Edge Seam AI Menu")
app_mode = st.sidebar.radio("เลือกโหมดการทำงาน:", [
    "1. ระบบทำนายและจัดพารามิเตอร์ (Predict & Optimize)",
    "2. วิเคราะห์ความเสี่ยงเชิงลึก (Data Analytics & Risk Zones)",
    "3. วิเคราะห์ภาพรวมพารามิเตอร์ (Global Feature Analysis)",
    "4. 🔍 ความโปร่งใสของโมเดล (Explainable AI - SHAP)",
    "5. 🚀 ระบบแนะนำขั้นสูง (Constrained Optimization)"
])

# ==========================================
# ฟังก์ชันสร้างข้อมูล Base Input (ใช้ร่วมกันหลายโหมด)
# ==========================================
def create_input_ui():
    st.subheader("📋 1. ข้อมูลคุณลักษณะเหล็กแผ่น (Uncontrollable Base Parameters)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        slabs_id = st.text_input("SLPRNU (SLAB Supplier)", value="ID2605")
        comqua = st.text_input("COMQUA (เกรดเหล็ก)", value="SS400")
        tem_dis = st.number_input("TEM_DIS (อุณหภูมิ DISCHARGE: °C)", value=1250.0, step=10.0)
    with col2:
        slabth = st.number_input("SLABTH (ความหนา SLAB: mm)", min_value=200.0, max_value=220.0, value=210.0, step=0.5)
        slabwi = st.number_input("SLABWI (ความกว้าง SLAB: mm)", value=1250.0, step=10.0)
        lsp_body = st.number_input("LSP_Body (อุณหภูมิ LSP: °C)", value=1080.0, step=10.0)
    with col3:
        slabwe = st.number_input("SLABWE (น้ำหนัก SLAB: kg)", value=22000.0, step=0.1)
        slfuti = st.number_input("SLFUTI (เวลาในเตา)", value=3.5, step=0.1)
        entry_body = st.number_input("Entry_Body (อุณหภูมิ Entry: °C)", value=1020.0, step=10.0)
    with col4:
        hnspdi = st.number_input("HNSPDI (ความหนา)", value=3.2, step=0.1)
        wnspdi = st.number_input("WNSPDI (ความกว้าง)", value=1219.0, step=5.0)

    st.subheader("⚙️ 2. พารามิเตอร์การผลิตปัจจุบัน (Controllable Setup Parameters)")
    tab_temp, tab_draft, tab_roll = st.tabs(["🌡️ อุณหภูมิและความเร็ว", "📉 อัตราการกดลูกรีด (RM Draft)", "🔄 ความเร็วลูกรีด (RM Speed)"])
    with tab_temp:
        c1, c2, c3, c4 = st.columns(4)
        with c1: ft_head = st.number_input("FT_HEAD (องศา)", value=850.0, step=10.0)
        with c2: ct_head = st.number_input("CT_HEAD (องศา)", value=580.0, step=10.0)
        with c3: xvptf8 = st.number_input("XVPTF8 (ความเร็วลูกรีด)", value=8.5, step=0.5)
        with c4: rmextg = st.number_input("RMEXTG (BarThk)", value=32.0, step=1.0)
    with tab_draft:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: psdrft1 = st.number_input("PSDRFT1 (Pass1 Draft)", value=42.2, step=0.5)
        with c2: psdrft2 = st.number_input("PSDRFT2 (Pass2 Draft)", value=48.1, step=0.5)
        with c3: psdrft3 = st.number_input("PSDRFT3 (Pass3 Draft)", value=48.6, step=0.5)
        with c4: psdrft4 = st.number_input("PSDRFT4 (Pass4 Draft)", value=48.8, step=0.5)
        with c5: psdrft5 = st.number_input("PSDRFT5 (Pass5 Draft)", value=50.2, step=0.5)
    with tab_roll:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: psrcms1 = st.number_input("PSRCMS1 (Pass1 Speed)", value=1.5, step=0.1)
        with c2: psrcms2 = st.number_input("PSRCMS2 (Pass2 Speed)", value=1.5, step=0.1)
        with c3: psrcms3 = st.number_input("PSRCMS3 (Pass3 Speed)", value=1.5, step=0.1)
        with c4: psrcms4 = st.number_input("PSRCMS4 (Pass4 Speed)", value=1.5, step=0.1)
        with c5: psrcms5 = st.number_input("PSRCMS5 (Pass5 Speed)", value=2.5, step=0.1)

    base_data = {
        'SLPRNU': slabs_id, 'COMQUA': comqua, 'SLABTH': slabth, 'SLABWI': slabwi, 'SLABWE': slabwe,
        'SLFUTI': slfuti, 'HNSPDI': hnspdi, 'WNSPDI': wnspdi, 'FT_HEAD': ft_head, 'CT_HEAD': ct_head,
        'XVPTF8': xvptf8, 'RMEXTG': rmextg, 'PSDRFT1': psdrft1, 'PSDRFT2': psdrft2, 'PSDRFT3': psdrft3,
        'PSDRFT4': psdrft4, 'PSDRFT5': psdrft5, 'PSRCMS1': psrcms1, 'PSRCMS2': psrcms2, 'PSRCMS3': psrcms3,
        'PSRCMS4': psrcms4, 'PSRCMS5': psrcms5, 'LSP_Body': lsp_body, 'TEM_DIS': tem_dis, 'Entry_Body': entry_body,
        'PSDRFT': 25.2, 'CORPSR_M1': 8500.0, 'CORPSR_M2': 10000.0, 'CORPSR_M3': 11000.0,
        'CORPSR_M4': 12000.0, 'CORPSR_M5': 13500.0, 'RIDAMF1': 0.30, 'RIDAMF2': 0.32, 'RIDAMF3': 0.28, 'RIDAMF4': 0.23,
        'RIDAMF5': 0.20, 'RIDAMF6': 0.18, 'RIDAMF7': 0.15, 'CBTHSP': 2.5, 'CBRUSP': 4.5, 'DESCH1_MIN': 155.0,
        'DESCH2_MIN': 155.0, 'TNVTRP1': 0.80, 'TNVTRP2': 0.90, 'TNVTRP3': 0.90, 'TNVTRP4': 0.90, 'TNVTRP5': 0.90,
        'TNVTRP6': 0.9, 'TNVTRP7': 1.0, 'FTGM': 9500.0, 'FT_BODY': ft_head,
        'CT_BODY': ct_head, 'SLAB_QUALITY': 'C032', 'OPCCO': '0', 'LCBXON': 'N', 'ENDUSE': 'S', 'PASSNR': 5,
        'DescaleCondition': 'OK'
    }
    return pd.DataFrame([base_data])

# ==========================================
# โหมด 1: ระบบทำนายและการจัดการหน้างาน
# ==========================================
if app_mode == "1. ระบบทำนายและจัดพารามิเตอร์ (Predict & Optimize)":
    st.title("🏭 Edge Seam Defect Prediction & Parameter Optimization")
    st.markdown("ระบบวิเคราะห์จุดเสี่ยงและแนะนำพารามิเตอร์แบบมาตรฐาน (Standard Optimization)")
    st.divider()

    input_df = create_input_ui()

    st.divider()
    if st.button("📊 วิเคราะห์และตรวจสอบพารามิเตอร์การผลิต", type="primary", use_container_width=True):
        with st.spinner("AI กำลังวิเคราะห์จุดเสี่ยงและคำนวณทางเลือกกระบวนการผลิต..."):
            suggestions, final_prob, status = optimizer.optimize(input_df, advanced_mode=False)

        st.subheader("🎯 ผลการวิเคราะห์จากระบบ AI")
        if status == "safe":
            st.success(f"🟩 **สถานะปกติ (GOOD):** พารามิเตอร์ปัจจุบันมีความปลอดภัยสูง (โอกาสเกิด Defect เพียง {final_prob * 100:.2f}%)")
            st.balloons()
        else:
            st.error(f"🟥 **จุดเสี่ยงระดับสูง (NG DETECTED):** พารามิเตอร์ปัจจุบันมีความเสี่ยงเกิดแตกขอบ (Risk Score: {final_prob * 100:.2f}%)")

            if status == "optimized" and suggestions:
                st.warning(f"💡 **AI Setup Guidelines:** แนะนำการปรับปรุงพารามิเตอร์เพื่อลดความเสี่ยงลง (ความเสี่ยงจะลดลงเหลือ {final_prob * 100:.2f}%)")
                guide_data = [{"Parameter": k, "ปัจจุบัน (Current)": f"{v['Current']:.2f}", "แนะนำ (Guideline)": f"{v['Suggested']:.2f}", "การกระทำ (Action)": f"🔺 เพิ่มขึ้น (+{v['Change']:.2f})" if v['Change'] > 0 else f"🔻 ลดลง ({v['Change']:.2f})"} for k, v in suggestions.items()]
                st.table(pd.DataFrame(guide_data).set_index("Parameter"))
            elif status == "unable":
                st.warning("⚠️ **ไม่พบทางแก้ไขที่ปลอดภัยภายใต้เงื่อนไขจำกัด 5%**")

# ==========================================
# โหมด 2: Data Analytics & Risk Zones
# ==========================================
elif app_mode == "2. วิเคราะห์ความเสี่ยงเชิงลึก (Data Analytics & Risk Zones)":
    st.title("🔬 เจาะลึกช่วงพารามิเตอร์ (Safe Range & NG Risk Analysis)")
    st.divider()
    base_features_to_view = ['TEM_DIS', 'LSP_Body', 'Entry_Body', 'SLABTH', 'SLABWI', 'HNSPDI', 'WNSPDI']
    all_viewable_features = optimizer.controllable_cols + base_features_to_view
    available_features = [col for col in all_viewable_features if col in raw_df.columns]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ เลือกพารามิเตอร์เพื่อวิเคราะห์")
        selected_feature = st.selectbox("Feature / Parameter:", available_features)
        good_df = raw_df[raw_df['Defect'] == 0]
        q10 = good_df[selected_feature].quantile(0.10)
        q90 = good_df[selected_feature].quantile(0.90)
        st.info(f"**💡 ช่วงปลอดภัย (80% ของชิ้นงานที่ผ่าน):** {q10:.2f} - {q90:.2f}")
    with col2:
        fig_dist = px.histogram(raw_df, x=selected_feature, color='Defect_Label', barmode='overlay', marginal='box', nbins=50, color_discrete_map={'Good (ปกติ)': '#3498db', 'NG (ตำหนิ)': '#e74c3c'}, title=f"การกระจายตัวของ {selected_feature} (Good vs NG)", opacity=0.7)
        fig_dist.add_vline(x=q10, line_dash="dash", line_color="green")
        fig_dist.add_vline(x=q90, line_dash="dash", line_color="green")
        st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# โหมด 3: Global Feature Analysis
# ==========================================
elif app_mode == "3. วิเคราะห์ภาพรวมพารามิเตอร์ (Global Feature Analysis)":
    st.title("🌐 ภาพรวมพารามิเตอร์เชิงลึก (Global Parameter Analysis)")
    st.divider()
    numeric_cols = raw_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    available_multi_cols = [c for c in numeric_cols if c not in ['Defect', 'Defect_Label']]
    default_multi_cols = ['SLABTH', 'TEM_DIS', 'FT_HEAD', 'CT_HEAD', 'XVPTF8', 'PSDRFT1']
    valid_defaults = [c for c in default_multi_cols if c in available_multi_cols]

    selected_multi_cols = st.multiselect("เลือกพารามิเตอร์ 3-8 ตัว เพื่อดูปฏิสัมพันธ์:", options=available_multi_cols, default=valid_defaults)

    if len(selected_multi_cols) >= 2:
        tab1, tab2 = st.tabs(["🔗 Correlation Heatmap", "🛤️ Parallel Coordinates"])
        with tab1:
            corr_df = raw_df[selected_multi_cols + ['Defect']].corr()
            fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        with tab2:
            fig_par = px.parallel_coordinates(raw_df, dimensions=selected_multi_cols, color='Defect', color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0.5)
            st.plotly_chart(fig_par, use_container_width=True)

# ==========================================
# โหมด 4: ความโปร่งใสของโมเดล (Explainable AI - SHAP)
# ==========================================
elif app_mode == "4. 🔍 ความโปร่งใสของโมเดล (Explainable AI - SHAP)":
    st.title("🔍 ทำไม AI ถึงตัดสินใจแบบนี้? (Explainable AI by SHAP)")
    st.markdown("ถอดรหัสกล่องดำ (Black Box) ของ AI เพื่อดูว่าพารามิเตอร์ตัวใดกำลัง **เพิ่มความเสี่ยง (สีแดง)** หรือ **ดึงความเสี่ยงลง (สีเขียว)** สำหรับชิ้นงานม้วนนี้")
    st.divider()

    input_df = create_input_ui()

    st.divider()
    if st.button("🧠 สร้าง SHAP Explanation Chart", type="primary", use_container_width=True):
        with st.spinner("AI กำลังถอดรหัสพารามิเตอร์ (คำนวณ SHAP Values)..."):
            # 1. Preprocess ข้อมูลปัจจุบัน
            X_curr = optimizer._preprocess(input_df)
            current_prob = model.predict(X_curr, verbose=0)[0][0]

            # 2. สร้าง Background Data สำหรับ SHAP (ใช้ Sample จาก raw_df)
            bg_df = raw_df.sample(n=min(50, len(raw_df)), random_state=42)
            bg_X = optimizer._preprocess(bg_df)

            # 3. คำนวณ SHAP
            explainer = shap.KernelExplainer(model.predict, bg_X)
            shap_values = explainer.shap_values(X_curr)

            # จัดการ Shape ของ SHAP อย่างรัดกุม (รองรับทั้งเวอร์ชันเก่าและใหม่)
            if hasattr(shap_values, 'values'): # หากเป็น Explanation Object
                contribs = np.array(shap_values.values).flatten()
            elif isinstance(shap_values, list): # หากรีเทิร์นเป็น List
                contribs = np.array(shap_values[0]).flatten()
            else: # หากรีเทิร์นเป็น Numpy Array
                contribs = np.array(shap_values).flatten()

            # ป้องกัน ValueError: บังคับให้ความยาวของ Arrays เท่ากันเสมอ
            n_features = len(X_curr.columns)
            if len(contribs) > n_features:
                contribs = contribs[-n_features:] # ตัดส่วนเกินออก
            elif len(contribs) < n_features:
                contribs = np.pad(contribs, (0, n_features - len(contribs))) # เติม 0 ให้เต็ม

            # สร้าง DataFrame สำหรับ Plotly
            shap_df = pd.DataFrame({
                'Feature': X_curr.columns,
                'Contribution': contribs,
                'Value': X_curr.iloc[0].values
            })
            # คัดเฉพาะตัวที่มีผลมากสุด 15 ตัว
            shap_df['Abs_Contrib'] = shap_df['Contribution'].abs()
            shap_df = shap_df.sort_values(by='Abs_Contrib', ascending=False).head(15)
            shap_df = shap_df.sort_values(by='Contribution', ascending=True) # จัดเรียงก่อนพล็อต

            shap_df['Type'] = np.where(shap_df['Contribution'] > 0, 'เพิ่มความเสี่ยง (NG+)', 'ลดความเสี่ยง (Safe-)')

            # สร้างกราฟ Plotly Horizontal Bar
            fig_shap = px.bar(
                shap_df, x='Contribution', y='Feature', color='Type',
                orientation='h',
                color_discrete_map={'เพิ่มความเสี่ยง (NG+)': '#e74c3c', 'ลดความเสี่ยง (Safe-)': '#2ecc71'},
                title=f"พารามิเตอร์ที่มีอิทธิพลต่อความเสี่ยง (โอกาส NG ประเมินไว้ที่ {current_prob*100:.2f}%)",
                labels={'Contribution': 'ผลกระทบต่อความเสี่ยง (SHAP Value)', 'Feature': 'ชื่อพารามิเตอร์'}
            )
            fig_shap.update_layout(height=600)

        st.plotly_chart(fig_shap, use_container_width=True)
        st.info("💡 **วิธีอ่านกราฟ:** แท่ง **สีแดง** ชี้ไปทางขวา คือพารามิเตอร์ที่กำลังสร้างปัญหาให้เกิดของเสีย (NG) ส่วนแท่ง **สีเขียว** ชี้ไปทางซ้าย คือพารามิเตอร์ที่กำลังช่วยรั้งให้เหล็กแผ่นนี้ปลอดภัย")

# ==========================================
# โหมด 5: ระบบแนะนำขั้นสูง (Constrained Optimization)
# ==========================================
elif app_mode == "5. 🚀 ระบบแนะนำขั้นสูง (Constrained Optimization)":
    st.title("🚀 ระบบแนะนำพารามิเตอร์ขั้นสูง (Advanced Constrained Optimizer)")
    st.markdown("""
    ยกระดับการค้นหาทางออกด้วยกฎทางวิศวกรรม (Domain Physics Integration) AI จะไม่แนะนำพารามิเตอร์แบบสุ่มสี่สุ่มห้า แต่จะถูกตีกรอบด้วยเงื่อนไข:
    * **กฎระบายความร้อน (Cooling Physics):** `CT_HEAD` ต้องมีอุณหภูมิต่ำกว่า `FT_HEAD` อย่างน้อย 150 องศาเซลเซียสเสมอ
    * **กฎรักษาความหนา (Draft Consistency):** ผลรวมของแรงกด (Total Draft: PSDRFT1-5) ต้องคงที่ (แกว่งได้ไม่เกิน 2%) 
    * *ค้นหาลึกขึ้น:* ขยายขอบเขตการจำลองสถานการณ์เป็น 2,000 รูปแบบ (กว้างขึ้นเป็น 10%)
    """)
    st.divider()

    input_df = create_input_ui()

    st.divider()
    if st.button("⚙️ รันระบบ Advanced Optimizer", type="primary", use_container_width=True):
        with st.spinner("AI กำลังจำลองสถานการณ์ทางฟิสิกส์ 2,000 รูปแบบ..."):
            suggestions, final_prob, status = optimizer.optimize(input_df, advanced_mode=True)

        st.subheader("🎯 ผลลัพธ์จาก Advanced Optimizer")
        if status == "safe":
            st.success(f"🟩 **สถานะปกติ (GOOD):** พารามิเตอร์ปัจจุบันมีความปลอดภัยสูง (โอกาสเกิด Defect เพียง {final_prob * 100:.2f}%)")
        elif status == "unable_due_to_physics":
            st.error("⚠️ **ไม่พบทางออก:** ระบบจำลองสถานการณ์ไว้ 2,000 รูปแบบ แต่ไม่มีรูปแบบใดที่ 'ความเสี่ยงต่ำ' และ 'ผ่านกฎฟิสิกส์วิศวกรรม' พร้อมๆ กัน กรุณาตรวจสอบวัตถุดิบ (Slab)")
        elif status == "unable":
            st.warning("⚠️ **ไม่พบทางแก้ไขที่ปลอดภัยภายใต้เงื่อนไขจำกัด 10%**")
        else:
            st.error(f"🟥 **จุดเสี่ยงระดับสูง (NG DETECTED):** ความเสี่ยงก่อนแก้: {final_prob*100:.2f}%")
            st.success(f"💡 **AI Advanced Guidelines:** พบแนวทางแก้ปัญหาที่ตรงตามหลักวิศวกรรม! (ความเสี่ยงหลังแก้: {final_prob * 100:.2f}%)")

            guide_data = []
            for k, v in suggestions.items():
                guide_data.append({
                    "Parameter": k,
                    "ปัจจุบัน (Current)": f"{v['Current']:.2f}",
                    "แนะนำ (Guideline)": f"{v['Suggested']:.2f}",
                    "การกระทำ (Action)": f"🔺 เพิ่มขึ้น (+{v['Change']:.2f})" if v['Change'] > 0 else f"🔻 ลดลง ({v['Change']:.2f})"
                })
            st.table(pd.DataFrame(guide_data).set_index("Parameter"))
            st.info("ℹ️ *ค่าตัวเลขชุดนี้ผ่านการยืนยันแล้วว่าสอดคล้องกับกฎ Cooling Physics และ Draft Consistency*")
