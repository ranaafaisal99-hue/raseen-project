import sys
import os
import streamlit as st

# إضافة المسار الحالي إلى sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from supervisor_agent import run_supervisor


# إعداد الصفحة
st.set_page_config(
    page_title="رَصِين | Turbofan Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ منصة رَصِين | Turbofan Predictive Maintenance")
st.caption(
    "نظام صيانة تنبؤية يعتمد على Machine Learning وLangChain "
    "لتحليل حالة المحرك والتنبؤ بـ RUL"
)

st.sidebar.header("🎛️ بيانات المحرك")

# Engine ID
engine_id = st.sidebar.selectbox(
    "رقم المحرك",
    ["ENGINE-001", "ENGINE-002"]
)

# Cycle
cycle = st.sidebar.number_input(
    "رقم الدورة (Cycle)",
    min_value=1,
    value=1,
    step=1
)

st.sidebar.divider()

st.sidebar.subheader("قراءات الحساسات")

# Sensor 1
sensor_1 = st.sidebar.number_input(
    "T2 — درجة الحرارة عند مدخل المكبس/الضاغط (Compressor Inlet Temperature)",
    value=0.0,
    format="%.4f"
)

# Sensor 2
sensor_2 = st.sidebar.number_input(
    "T24 — درجة الحرارة عند مخرج الضاغط ذو الضغط المنخفض (LPC Outlet Temperature)",
    value=0.0,
    format="%.4f"
)

# Sensor 3
sensor_3 = st.sidebar.number_input(
    "T30 — درجة الحرارة عند مخرج الضاغط ذو الضغط العالي (HPC Outlet Temperature)",
    value=0.0,
    format="%.4f"
)

# Sensor 4
sensor_4 = st.sidebar.number_input(
    "T48 — درجة الحرارة عند مخرج التوربين ذو الضغط العالي (HPT Outlet Temperature)",
    value=0.0,
    format="%.4f"
)

# Sensor 5
sensor_5 = st.sidebar.number_input(
    "T50 — درجة الحرارة عند مخرج التوربين ذو الضغط المنخفض (LPT Outlet Temperature)",
    value=0.0,
    format="%.4f"
)

# Sensor 6
sensor_6 = st.sidebar.number_input(
    "P2 — الضغط عند مدخل الضاغط (Compressor Inlet Pressure)",
    value=0.0,
    format="%.4f"
)

# Sensor 7
sensor_7 = st.sidebar.number_input(
    "P21 — الضغط داخل أنبوب السحب/المروحة (Fan Outlet Pressure)",
    value=0.0,
    format="%.4f"
)

# Sensor 8
sensor_8 = st.sidebar.number_input(
    "P30 — الضغط الكلي عند مخرج الضاغط ذو الضغط العالي (HPC Outlet Total Pressure)",
    value=0.0,
    format="%.4f"
)

# Sensor 9
sensor_9 = st.sidebar.number_input(
    "P48 — الضغط عند مخرج التوربين ذو الضغط العالي (HPT Outlet Pressure)",
    value=0.0,
    format="%.4f"
)

# Sensor 10
sensor_10 = st.sidebar.number_input(
    "Nf — سرعة دوران المروحة / التوربين المنخفض (Fan Speed / Physical Low-Pressure Spool Speed)",
    value=0.0,
    format="%.4f"
)

# Sensor 11
sensor_11 = st.sidebar.number_input(
    "Nc — سرعة دوران المحور الرئيسي / الضاغط العالي (Core Speed / Physical High-Pressure Spool Speed)",
    value=0.0,
    format="%.4f"
)

# Sensor 12
sensor_12 = st.sidebar.number_input(
    "W36 — معدل تدفق الوقود كلياً إلى غرفة الاحتراق (Fuel Flow Rate)",
    value=0.0,
    format="%.4f"
)

# Sensor 13
sensor_13 = st.sidebar.number_input(
    "FAR — نسبة الوقود إلى الهواء أثناء الاحتراق (Burner Fuel-Air Ratio)",
    value=0.0,
    format="%.4f"
)

# Sensor 14
sensor_14 = st.sidebar.number_input(
    "htBleed — كمية الهواء المسحوب للتبريد الداخلي (Enthalpy Bleed / Cooling Air Flow)",
    value=0.0,
    format="%.4f"
)

# معلومات النموذج
st.divider()

c1, c2, c3 = st.columns(3)

c1.metric(
    "نوع المشكلة",
    "RUL Regression"
)

c2.metric(
    "عدد الحساسات",
    "14"
)

c3.metric(
    "الموديل النهائي",
    "Bagging"
)


st.divider()

col_left, col_right = st.columns([6, 4])


with col_left:

    st.subheader("🤖 Supervisor Agent")

    st.write(
        "يقوم المشرف بتنسيق عملية الصيانة التنبؤية من خلال "
        "تحليل قراءات الحساسات ثم تنفيذ أدوات النظام."
    )

    if st.button(
        "🚀 تشغيل Supervisor Agent",
        type="primary",
        use_container_width=True
    ):

        import time

        with st.status(
            "🧠 جاري تنفيذ نظام الصيانة التنبؤية...",
            expanded=True
        ) as status:

            res = run_supervisor(
                engine_id,
                cycle,
                sensor_1,
                sensor_2,
                sensor_3,
                sensor_4,
                sensor_5,
                sensor_6,
                sensor_7,
                sensor_8,
                sensor_9,
                sensor_10,
                sensor_11,
                sensor_12,
                sensor_13,
                sensor_14
            )

            for log in res["logs"]:
                st.write(log)
                time.sleep(0.5)

            status.update(
                label="✅ اكتمل التحليل والتنفيذ بنجاح",
                state="complete"
            )


with col_right:

    st.subheader("📊 معلومات النموذج")

    st.write("**Dataset:** NASA C-MAPSS 2")
    st.write("**Task:** Remaining Useful Life (RUL) Prediction")
    st.write("**Features:** 14 Sensors + Cycle")
    st.write("**Models:** 8 Regression Models")
    st.write("**Optimization:** BayesSearchCV")
    st.write("**Cross Validation:** GroupKFold")
    st.write("**Ensemble:** Bagging / Stacking / Boosting")

    st.info(
        "الموديل النهائي المحفوظ حاليًا هو Bagging Regressor "
        "بناءً على نتائج الاختبار الحالية."
    )