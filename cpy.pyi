import os
import jieba
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

# ===========================
# 🌟 Streamlit 页面设置
# ===========================
st.set_page_config(page_title="AI 智能食堂助手", page_icon="🍱", layout="wide")

# ===========================
# 📂 路径定义
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_FILE = os.path.join(BASE_DIR, "menu_data.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "user_feedback.csv")
MODEL_FILE = os.path.join(BASE_DIR, "user_model.pkl")

# ===========================
# 📊 加载菜单数据（带 BOM/列检查）
# ===========================
@st.cache_data
def load_menu():
    if not os.path.exists(MENU_FILE):
        st.error("❌ 未找到 menu_data.csv，请确保它与 app.py 在同一目录下。")
        st.stop()

    # 尝试不同编码读取 CSV（支持 Excel 导出的 UTF-8-BOM）
    try:
        df = pd.read_csv(MENU_FILE, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(MENU_FILE, encoding="utf-8")
        except Exception:
            df = pd.read_csv(MENU_FILE, encoding="gbk")

    # 确保必要列存在
    required_cols = ["name", "canteen", "price", "calories"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ menu_data.csv 缺少必要列：'{col}'。请检查文件内容。")
            st.stop()

    # 处理 tags 列
    if "tags" not in df.columns:
        df["tags"] = ""
    df["tags"] = df["tags"].apply(lambda x: x.split(";") if isinstance(x, str) else [])

    # 转换数值列
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    return df

menu_data = load_menu()

# ===========================
# 🧠 文本相似度计算
# ===========================
def compute_similarity(df, text):
    corpus = [" ".join([str(row["name"])] + row["tags"]) for _, row in df.iterrows()]
    user_cut = " ".join(jieba.lcut(text))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus + [user_cut])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    df["similarity"] = sim
    return df

# ===========================
# 🔧 特征工程
# ===========================
def prepare_features(df, text):
    df = compute_similarity(df, text)
    df["price_norm"] = MinMaxScaler().fit_transform(df[["price"]])
    df["cal_norm"] = MinMaxScaler().fit_transform(df[["calories"]])
    return df, df[["similarity", "price_norm", "cal_norm"]].values

# ===========================
# 💾 模型加载 / 初始化
# ===========================
def load_or_init_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return LGBMClassifier(n_estimators=80, learning_rate=0.1, random_state=42)

model = load_or_init_model()

# ===========================
# 📝 反馈保存
# ===========================
def record_feedback(dish_name, liked):
    new_row = pd.DataFrame([[dish_name, int(liked)]], columns=["dish", "liked"])
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8")

# ===========================
# 🧩 模型重新训练
# ===========================
def retrain_model(df):
    if not os.path.exists(FEEDBACK_FILE):
        st.warning("暂无用户反馈，无法训练模型。")
        return None
    fb = pd.read_csv(FEEDBACK_FILE)
    merged = df.merge(fb, left_on="name", right_on="dish", how="inner")
    if merged.empty:
        st.warning("反馈数据为空，模型未更新。")
        return None

    X = merged[["similarity", "price_norm", "cal_norm"]]
    y = merged["liked"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
    new_model.fit(X_train, y_train)
    joblib.dump(new_model, MODEL_FILE)
    st.success("✅ 模型已重新训练成功！")
    return new_model

# ===========================
# 🍽️ 推荐逻辑
# ===========================
from sklearn.exceptions import NotFittedError

def smart_recommend(df, text, model=None, canteen=None):
    # 特征计算
    df, X = prepare_features(df, text)

    # 检查模型是否可用
    model_ready = False
    if model is not None:
        try:
            _ = model.predict_proba([[0, 0, 0]])  # 尝试虚拟预测
            model_ready = True
        except NotFittedError:
            st.warning("⚠️ 模型尚未训练，使用基础相似度推荐。")
        except Exception:
            model_ready = False

    # 使用模型预测（如果模型可用）
    if model_ready:
        try:
            df["predict"] = model.predict_proba(X)[:, 1]
            df["score"] = 0.6 * df["predict"] + 0.4 * df["similarity"]
        except Exception as e:
            st.warning(f"⚠️ 模型预测失败：{e}")
            df["score"] = 0.7 * df["similarity"] + 0.3 * (1 - df["price_norm"])
    else:
        # 无模型时使用相似度+价格综合推荐
        df["score"] = 0.7 * df["similarity"] + 0.3 * (1 - df["price_norm"])

    # 按食堂过滤
    if canteen and canteen != "所有食堂":
        df = df[df["canteen"] == canteen]

    return df.sort_values(by="score", ascending=False).head(5)

# ===========================
# 🌟 页面主体
# ===========================
st.title("🍱 北京师范大学 · AI 智能食堂助手（自学习版）")
st.markdown("> 💬 示例：`清淡低脂`、`想吃辣的`、`高蛋白`")

canteen = st.selectbox("选择食堂：", ["所有食堂"] + sorted(menu_data["canteen"].unique().tolist()))
text = st.text_input("请输入你的需求：", placeholder="例如：清淡低脂、15元以内、增肌餐...")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("🍽️ 智能推荐")
with col2:
    retrain_btn = st.button("🧠 重新训练模型")

# ===========================
# 🔍 推荐展示
# ===========================
if run_btn:
    if not text.strip():
        st.warning("请输入饮食偏好～")
        st.stop()
    recs = smart_recommend(menu_data, text, model, canteen)
    if recs.empty:
        st.warning("😅 当前没有匹配的菜品。")
    else:
        st.subheader("✅ 智能推荐结果")
        for _, row in recs.iterrows():
            with st.expander(f"🍛 {row['name']}（{row['price']} 元）", expanded=True):
                st.markdown(f"""
                - 🔥 热量：**{row['calories']} kcal**
                - 🏷️ 标签：{', '.join(row['tags']) if row['tags'] else '暂无标签'}
                - 🏫 食堂：{row['canteen']}
                - 🤖 推荐得分：{row['score']:.3f}
                """)
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button(f"👍 喜欢 {row['name']}", key=f"like_{row['name']}"):
                        record_feedback(row['name'], True)
                        st.success(f"已记录喜欢 {row['name']}")
                with c2:
                    if st.button(f"👎 不喜欢 {row['name']}", key=f"dislike_{row['name']}"):
                        record_feedback(row['name'], False)
                        st.info(f"已记录不喜欢 {row['name']}")

# ===========================
# 🔁 重新训练按钮
# ===========================
if retrain_btn:
    model = retrain_model(menu_data)

# ===========================
# 📘 算法说明
# ===========================
st.markdown("---")
st.markdown("""
### 🧠 算法原理说明
- **TF-IDF + 余弦相似度**：理解用户输入语义；
- **LightGBM 模型**：根据用户反馈学习口味；
- **多目标排序**：平衡“相似度 + 价格 + 热量”；
- **自学习机制**：你反馈得越多，推荐越精准。
""")
