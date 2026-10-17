
import sys
import os
import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from secure_ingestion.csv_validator import validate_csv
from supervisor_agent import run_supervisor


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="رَصِين | Turbofan Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ منصة رَصِين | Turbofan Predictive Maintenance")
st.caption(
    "نظام صيانة تنبؤية يعتمد على Machine Learning وLangChain "
    "لتحليل حالة المحرك والتنبؤ بالعمر التشغيلي المتبقي RUL."
)


# ============================================================
# Local Security Audit Log
# Session-only; not sent to external services
# ============================================================

if "security_audit_log" not in st.session_state:
    st.session_state.security_audit_log = []

if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = {}


# ============================================================
# Engine Inputs
# ============================================================

st.sidebar.header("🎛️ بيانات المحرك")

engine_id = st.sidebar.selectbox(
    "رقم المحرك",
    ["ENGINE-001", "ENGINE-002"],
)

cycle = st.sidebar.number_input(
    "رقم الدورة (Cycle)",
    min_value=1,
    value=1,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("قراءات الحساسات")

sensor_labels = [
    "T2 — Compressor Inlet Temperature",
    "T24 — LPC Outlet Temperature",
    "T30 — HPC Outlet Temperature",
    "T48 — HPT Outlet Temperature",
    "T50 — LPT Outlet Temperature",
    "P2 — Compressor Inlet Pressure",
    "P21 — Fan Outlet Pressure",
    "P30 — HPC Outlet Total Pressure",
    "P48 — HPT Outlet Pressure",
    "Nf — Fan Speed",
    "Nc — Core Speed",
    "W36 — Fuel Flow Rate",
    "FAR — Burner Fuel-Air Ratio",
    "htBleed — Cooling Air Flow",
]

sensor_values = []

for index, label in enumerate(sensor_labels, start=1):
    value = st.sidebar.number_input(
        label,
        value=0.0,
        format="%.4f",
        key=f"sensor_{index}",
    )
    sensor_values.append(value)


# ============================================================
# Model Information
# ============================================================

st.divider()

c1, c2, c3 = st.columns(3)

c1.metric("نوع المشكلة", "RUL Regression")
c2.metric("عدد الحساسات", "14")
c3.metric("الموديل النهائي", "Bagging")

st.divider()

col_left, col_right = st.columns([6, 4])


# ============================================================
# Supervisor Agent
# ============================================================

with col_left:
    st.subheader("🤖 Supervisor Agent")

    st.write(
        "ينسق المشرف تحليل قراءات الحساسات وتقييم المخاطر "
        "وإنشاء تنبيه صيانة محلي."
    )

    if st.button(
        "🚀 تشغيل Supervisor Agent",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.status(
                "🧠 جاري تنفيذ تحليل الصيانة التنبؤية...",
                expanded=True,
            ) as status:
                result = run_supervisor(
                    engine_id,
                    int(cycle),
                    *sensor_values,
                )

                for log in result.get("logs", []):
                    st.write(log)

                status.update(
                    label="✅ اكتمل التحليل",
                    state="complete",
                )

        except Exception as exc:
            st.error(
                "تعذر إكمال التحليل. راجعي تفاصيل الخطأ في "
                "Terminal الخاص بالتطبيق."
            )
            print(f"Raseen analysis error: {type(exc).__name__}: {exc}")


# ============================================================
# Model Details
# ============================================================

with col_right:
    st.subheader("📊 معلومات النموذج")

    st.write("**Dataset:** NASA C-MAPSS")
    st.write("**Task:** Remaining Useful Life (RUL) Prediction")
    st.write("**Features:** 14 Sensors + Cycle")
    st.write("**Final Model:** Bagging Regressor")

    st.info(
        "يستخدم المشروع نموذج Bagging Regressor المحفوظ "
        "للتنبؤ بالعمر التشغيلي المتبقي."
    )


# ============================================================
# Secure Historical Data Import
# ============================================================

st.divider()
st.header("🔐 Secure Historical Data Import")

st.caption(
    "يُفحص ملف CSV محليًا قبل السماح بمعاينته. "
    "هذا الفحص أولي ولا يثبت موثوقية مصدر البيانات."
)

uploaded_file = st.file_uploader(
    "ارفعي ملف بيانات الحساسات بصيغة CSV",
    type=["csv"],
    key="secure_csv_upload",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # Identify the current upload to avoid duplicate audit entries
    upload_key = (
        uploaded_file.name,
        len(file_bytes),
        hash(file_bytes),
    )

    if upload_key not in st.session_state.processed_uploads:
        result = validate_csv(file_bytes, uploaded_file.name)

        if not result["ok"]:
            audit_entry = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "filename": uploaded_file.name,
                "status": "REJECTED",
                "reason": result["error"],
            }

            st.session_state.security_audit_log.append(audit_entry)
            st.session_state.processed_uploads[upload_key] = result

        else:
            audit_entry = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "filename": uploaded_file.name,
                "status": "ACCEPTED_INITIAL_CHECK",
                "reason": "Initial CSV validation passed",
            }

            st.session_state.security_audit_log.append(audit_entry)
            st.session_state.processed_uploads[upload_key] = result

    result = st.session_state.processed_uploads[upload_key]

    if not result["ok"]:
        st.error(f"تم رفض الملف: {result['error']}")

    else:
        st.success(
            "اجتاز الملف الفحص الأولي — "
            f"عدد الصفوف: {result['rows']}"
        )

        if st.button(
            "معاينة البيانات بعد التحقق",
            key="preview_verified_csv",
        ):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                )

            except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
                st.error("تعذرت قراءة الملف للمعاينة.")


# ============================================================
# Local Security Audit Viewer
# ============================================================

st.divider()
st.header("🛡️ Local Security Audit Log")

st.caption(
    "سجل مؤقت داخل جلسة التطبيق. لا يُرسل إلى خدمة خارجية، "
    "وقد يُفقد عند انتهاء الجلسة أو إعادة تشغيل التطبيق."
)

if st.session_state.security_audit_log:
    audit_df = pd.DataFrame(
        st.session_state.security_audit_log
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "مسح السجل المحلي",
        key="clear_local_audit",
    ):
        st.session_state.security_audit_log = []
        st.session_state.processed_uploads = {}
        st.rerun()

else:
    st.info("لا توجد محاولات رفع مسجلة في هذه الجلسة.")


# ============================================================
# One-Way Design Notice
# ============================================================

st.divider()
st.subheader("🔒 Receive-Only Design")

st.info(
    "هذا النموذج الأولي يفحص ملفات CSV ويحلل البيانات محليًا "
    "داخل مسار الاستيراد. هذا لا يفرض عزلًا شبكيًا فعليًا. "
    "يجب تطبيق واختبار Data Diode أو بوابة أحادية الاتجاه "
    "مناسبة قبل استخدام النظام مع شبكة OT صناعية."
)
