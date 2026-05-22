import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

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
        # Mock Data (กรณีไม่พบไฟล์ เพื่อให้ระบบทำงานและแสดงผลกราฟได้)
        np.random.seed(42)
        data = np.random.rand(1000, 40)
        df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(40)])

        # จำลองตัวแปรหน้างานบางตัวให้เห็นภาพ
        df['TEM_DIS'] = np.random.normal(1200, 50, 1000)
        df['LSP_Body'] = np.random.normal(1080, 30, 1000)
        df['Entry_Body'] = np.random.normal(1020, 25, 1000)
        df['FT_HEAD'] = np.random.normal(850, 20, 1000)
        df['CT_HEAD'] = np.random.normal(600, 15, 1000)
        df['XVPTF8'] = np.random.normal(10, 2, 1000)
        df['PSDRFT1'] = np.random.normal(45, 5, 1000)
        df['PSDRFT2'] = np.random.normal(47, 5, 1000)
        df['PSDRFT3'] = np.random.normal(48, 5, 1000)
        df['PSDRFT4'] = np.random.normal(46, 5, 1000)
        df['PSDRFT5'] = np.random.normal(44, 5, 1000)        

        # จำลองความสัมพันธ์: ถ้าอุณหภูมิต่ำไป หรือความเร็วสูงไป จะมีโอกาส NG มากขึ้น
        prob = (df['XVPTF8'] > 11.5) | (df['FT_HEAD'] < 830)
        df['Defect'] = np.where(prob, np.random.choice([0, 1], p=[0.3, 0.7]), np.random.choice([0, 1], p=[0.9, 0.1]))
        return df

try:
    model, imputer, scaler, encoder, top_features, col_names, threshold = load_models()
    num_cols = col_names['num_cols']
    cat_cols = col_names['cat_cols']
    raw_df = load_data()
except Exception as e:
    st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลด Model/Data: {e}")
    st.info("กรุณาตรวจสอบไฟล์ .keras, .save, และข้อมูล .xlsx")
    st.stop()

# ==========================================
# 2. คลาส Prescriptive Engine (AI Suggestion)
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

    def optimize(self, current_df):
        X_curr = self._preprocess(current_df.copy())
        current_prob = self.model.predict(X_curr, verbose=0)[0][0]

        if current_prob < self.threshold:
            return None, current_prob, "safe"

        n_sims = 800
        sim_df = pd.concat([current_df] * n_sims, ignore_index=True)

        for col in self.controllable_cols:
            if col in sim_df.columns:
                curr_val = current_df.loc[0, col]
                sim_df[col] = np.random.uniform(curr_val * (1 - self.adjustment_limit_pct),
                                                curr_val * (1 + self.adjustment_limit_pct), size=n_sims)

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
        df_num = self.imputer.transform(df[num_cols])
        df_num_scaled = self.scaler.transform(df_num)

        if self.encoder and cat_cols:
            df_cat = self.encoder.transform(df[cat_cols])
            X_all = np.hstack([df_num_scaled, df_cat])
        else:
            X_all = df_num_scaled

        all_features = num_cols + (list(self.encoder.get_feature_names_out(cat_cols)) if (self.encoder and cat_cols) else [])
        X_df = pd.DataFrame(X_all, columns=all_features)

        # ตรวจสอบว่า top_features มีอยู่ใน X_df หรือไม่
        missing_features = [f for f in self.top_features if f not in X_df.columns]
        if missing_features:
            st.error(f"Missing features in model input: {missing_features}")
            for f in missing_features:
                X_df[f] = 0 # Fallback

        return X_df[self.top_features]

optimizer = EdgeSeamOptimizer(model, imputer, scaler, encoder, top_features, threshold)

# ==========================================
# 3. เมนูด้านข้าง (Sidebar Navigation)
# ==========================================
st.sidebar.title("🏭 Edge Seam AI Menu")
app_mode = st.sidebar.radio("เลือกโหมดการทำงาน:", [
    "1. ระบบทำนายและจัดพารามิเตอร์ (Predict & Optimize)",
    "2. วิเคราะห์ความเสี่ยงเชิงลึก (Data Analytics & Risk Zones)"
])

# ==========================================
# โหมด 1: ระบบทำนายและการจัดการหน้างาน
# ==========================================
if app_mode == "1. ระบบทำนายและจัดพารามิเตอร์ (Predict & Optimize)":
    st.title("🏭 Edge Seam Defect Prediction & Parameter Optimization")
    st.markdown("ระบบวิเคราะห์จุดเสี่ยงการเกิด Edge Seam Defect และแนะนำการตั้งค่าพารามิเตอร์ลูกรีดอัตโนมัติ")
    st.divider()

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
    st.caption("ระบุค่าที่กำลังตั้งค่าอยู่ในปัจจุบันเพื่อตรวจสอบจุดเสี่ยง")

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

    # Base data construction
    base_data = {
        'SLPRNU': slabs_id, 'COMQUA': comqua, 'SLABTH': slabth, 'SLABWI': slabwi, 'SLABWE': slabwe,
        'SLFUTI': slfuti, 'HNSPDI': hnspdi, 'WNSPDI': wnspdi, 'FT_HEAD': ft_head, 'CT_HEAD': ct_head,
        'XVPTF8': xvptf8, 'RMEXTG': rmextg, 'PSDRFT1': psdrft1, 'PSDRFT2': psdrft2, 'PSDRFT3': psdrft3,
        'PSDRFT4': psdrft4, 'PSDRFT5': psdrft5, 'PSRCMS1': psrcms1, 'PSRCMS2': psrcms2, 'PSRCMS3': psrcms3,
        'PSRCMS4': psrcms4, 'PSRCMS5': psrcms5, 'LSP_Body': lsp_body, 'TEM_DIS': tem_dis, 'Entry_Body': entry_body,
        # Fallbacks
        'PSDRFT': 25.2, 'CORPSR_M1': 8500.0, 'CORPSR_M2': 10000.0, 'CORPSR_M3': 11000.0,
        'CORPSR_M4': 12000.0, 'CORPSR_M5': 13500.0, 'RIDAMF1': 0.30, 'RIDAMF2': 0.32, 'RIDAMF3': 0.28, 'RIDAMF4': 0.23,
        'RIDAMF5': 0.20, 'RIDAMF6': 0.18, 'RIDAMF7': 0.15, 'CBTHSP': 2.5, 'CBRUSP': 4.5, 'DESCH1_MIN': 155.0,
        'DESCH2_MIN': 155.0, 'TNVTRP1': 0.80, 'TNVTRP2': 0.90, 'TNVTRP3': 0.90, 'TNVTRP4': 0.90, 'TNVTRP5': 0.90,
        'TNVTRP6': 0.9, 'TNVTRP7': 1.0, 'FTGM': 9500.0, 'FT_BODY': ft_head, 'CT_BODY': ct_head,
        'SLAB_QUALITY': 'C032', 'OPCCO': '0', 'LCBXON': 'N', 'ENDUSE': 'S', 'PASSNR': 5, 'DescaleCondition': 'OK'
    }
    input_df = pd.DataFrame([base_data])

    st.divider()
    if st.button("📊 วิเคราะห์และตรวจสอบพารามิเตอร์การผลิต", type="primary", use_container_width=True):
        with st.spinner("AI กำลังวิเคราะห์จุดเสี่ยงและคำนวณทางเลือกกระบวนการผลิต..."):
            suggestions, final_prob, status = optimizer.optimize(input_df)

        st.subheader("🎯 ผลการวิเคราะห์จากระบบ AI")
        if status == "safe":
            st.success(f"🟩 **สถานะปกติ (GOOD):** พารามิเตอร์ปัจจุบันมีความปลอดภัยสูง (โอกาสเกิด Defect เพียง {final_prob * 100:.2f}%)")
            st.balloons()
        else:
            st.error(f"🟥 **จุดเสี่ยงระดับสูง (NG DETECTED):** พารามิเตอร์ปัจจุบันมีความเสี่ยงเกิดแตกขอบ (Risk Score: {final_prob * 100:.2f}%)")

            if status == "optimized" and suggestions:
                st.warning(f"💡 **AI Setup Guidelines:** ค้นพบแนวทางการปรับปรุงพารามิเตอร์เพื่อลดความเสี่ยงลง (ความเสี่ยงจะลดลงเหลือ {final_prob * 100:.2f}%)")
                guide_data = []
                for k, v in suggestions.items():
                    guide_data.append({
                        "Parameter": k,
                        "ปัจจุบัน (Current)": f"{v['Current']:.2f}",
                        "แนะนำ (Guideline)": f"{v['Suggested']:.2f}",
                        "การกระทำ (Action)": f"🔺 เพิ่มขึ้น (+{v['Change']:.2f})" if v['Change'] > 0 else f"🔻 ลดลง ({v['Change']:.2f})"
                    })
                st.table(pd.DataFrame(guide_data).set_index("Parameter"))
            elif status == "unable":
                st.warning("⚠️ **ไม่พบทางแก้ไขที่ปลอดภัยภายใต้เงื่อนไขจำกัด 5%** กรุณาตรวจสอบสภาวะเครื่องจักร")

# ==========================================
# โหมด 2: Data Science Analytics & Risk Zones
# ==========================================
elif app_mode == "2. วิเคราะห์ความเสี่ยงเชิงลึก (Data Analytics & Risk Zones)":
    st.title("🔬 เจาะลึกช่วงพารามิเตอร์ (Safe Range & NG Risk Analysis)")
    st.markdown("วิเคราะห์ข้อมูลทางสถิติเพื่อหา **'ช่วงปลอดภัย (Operating Envelope)'** และ **'พื้นที่เสี่ยง (Red Zone)'** จากประวัติการผลิตจริง")
    st.divider()

    # ดึงเฉพาะตัวแปรที่เป็น Numeric และ Control ได้ เพื่อความเข้าใจง่าย
    available_features = [col for col in optimizer.controllable_cols if col in raw_df.columns]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ เลือกพารามิเตอร์เพื่อวิเคราะห์")
        selected_feature = st.selectbox("Controllable Feature:", available_features)

        # คัดกรองข้อมูลเพื่อหาค่าสถิติสำหรับ 'Good' (Defect=0)
        good_df = raw_df[raw_df['Defect'] == 0]
        ng_df = raw_df[raw_df['Defect'] == 1]

        # คำนวณขอบเขตที่ปลอดภัย (อ้างอิงจาก IQR หรือ Percentile ของกลุ่ม Good)
        q10 = good_df[selected_feature].quantile(0.10)
        q90 = good_df[selected_feature].quantile(0.90)
        median_good = good_df[selected_feature].median()

        st.info(f"""
        **💡 ขอบเขตแนะนำทางสถิติ (Statistical Safe Range):**
        - ค่าที่พบบ่อย (Median): **{median_good:.2f}**
        - ช่วงปลอดภัย (80% ของชิ้นงานที่ผ่าน): **{q10:.2f} - {q90:.2f}**
        """)

        st.markdown("""
        **คำอธิบายกราฟ (Interpretation):**
        *กราฟด้านขวาแสดงการกระจายตัว หากพื้นที่สีแดง (NG) กระจุกตัวสูงแปลกแยกจากสีฟ้า (Good) นั่นคือ **Risk Zone** ที่ควรหลีกเลี่ยงเวลาตั้งค่าเครื่องจักร*
        """)

    with col2:
        # กราฟ Histogram / KDE Plot เพื่อดู Distribution 
        fig_dist = px.histogram(
            raw_df, x=selected_feature, color='Defect', 
            barmode='overlay', marginal='box', nbins=50,
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            title=f"การกระจายตัวของ {selected_feature} (Good vs NG)",
            opacity=0.7,
            labels={'Defect': 'สถานะ', selected_feature: f'ค่า {selected_feature}'}
        )

        # เพิ่มเส้น Safe Zone (Vertical lines)
        fig_dist.add_vline(x=q10, line_dash="dash", line_color="green", annotation_text="10th %ile (Safe)")
        fig_dist.add_vline(x=q90, line_dash="dash", line_color="green", annotation_text="90th %ile (Safe)")

        # ปรับชื่อ Legend ให้ดูง่าย
        fig_dist.for_each_trace(lambda t: t.update(name='Good (ปกติ)' if t.name == '0' else 'NG (ตำหนิ)'))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ส่วนที่ 2: Risk Probability Curve (Binned Risk Analysis)
    st.subheader(f"📈 โอกาสเกิด NG ในแต่ละช่วงของ {selected_feature} (Risk Probability Curve)")

    # แบ่งช่วงพารามิเตอร์ออกเป็น 10 ช่วง (Bins) และหา % NG ในช่วงนั้นๆ
    try:
        raw_df['bin'] = pd.qcut(raw_df[selected_feature], q=10, duplicates='drop')
        risk_df = raw_df.groupby('bin')['Defect'].agg(['mean', 'count']).reset_index()
        risk_df['bin_str'] = risk_df['bin'].astype(str)
        risk_df['Risk_Percent'] = risk_df['mean'] * 100

        fig_risk = px.line(
            risk_df, x='bin_str', y='Risk_Percent', markers=True,
            title=f"% ความเสี่ยงการเกิดตำหนิในแต่ละช่วงพารามิเตอร์",
            labels={'bin_str': f'ช่วงของ {selected_feature}', 'Risk_Percent': 'โอกาสเกิด NG (%)'}
        )
        # ตกแต่งกราฟ (เปลี่ยนสี, ขนาดจุด)
        fig_risk.update_traces(line_color='#e67e22', marker=dict(size=10, color='red'))

        # เพิ่มเส้น Threshold เพื่อเตือนภัย
        fig_risk.add_hline(y=threshold*100, line_dash="dot", line_color="red", annotation_text="AI Threshold Limits")

        st.plotly_chart(fig_risk, use_container_width=True)
    except Exception as e:
        st.warning(f"ไม่สามารถสร้าง Risk Curve ได้เนื่องจากการกระจายตัวของข้อมูลน้อยเกินไป: {e}")
