# =======================================================
# 🍱 BNU 智能食堂助手（自学习 + 反馈 + 可训练 修正版）
# =======================================================
import os
import pandas as pd
import jieba
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------- 路径设置 -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_FILE = os.path.join(BASE_DIR, "menu_data.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "user_feedback.csv")
MODEL_FILE = os.path.join(BASE_DIR, "user_model.pkl")

# ------------------- 加载菜单 -------------------
def load_menu():
    try:
        df = pd.read_csv(MENU_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(MENU_FILE, encoding="gbk")

    df["tags"] = df["tags"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    return df

# ------------------- 特征构造 -------------------
def prepare_features(df, text):
    corpus = [" ".join([row["name"]] + row["tags"]) for _, row in df.iterrows()]
    user_cut = " ".join(jieba.lcut(text))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus + [user_cut])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    df["similarity"] = sim

    scaler = MinMaxScaler()
    df["price_norm"] = scaler.fit_transform(df[["price"]])
    df["cal_norm"] = scaler.fit_transform(df[["calories"]])
    return df, df[["similarity", "price_norm", "cal_norm"]]

# ------------------- 保存反馈 -------------------
def save_feedback(dish_name, liked):
    try:
        if not os.path.exists(FEEDBACK_FILE):
            pd.DataFrame(columns=["dish", "liked"]).to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")

        new = pd.DataFrame([[dish_name, int(liked)]], columns=["dish", "liked"])
        new.to_csv(FEEDBACK_FILE, mode="a", index=False, header=False, encoding="utf-8-sig")

        st.toast(f"✅ 已记录反馈：{'喜欢' if liked else '不喜欢'} {dish_name}")
    except Exception as e:
        st.error(f"保存反馈失败：{e}")

# ------------------- 模型训练 -------------------
def retrain_model(df):
    """重新训练模型"""
    if not os.path.exists(FEEDBACK_FILE):
        st.warning("暂无用户反馈，无法训练模型。")
        return None

    fb = pd.read_csv(FEEDBACK_FILE)
    if fb.empty:
        st.warning("反馈数据为空，请多点几次喜欢/不喜欢。")
        return None

    # ---- 准备特征（随便给个文本，用于生成特征列）----
    df, _ = prepare_features(df, "辣")

    # ---- 合并反馈 ----
    merged = df.merge(fb, left_on="name", right_on="dish", how="inner")
    if merged.empty:
        st.warning("反馈菜品与菜单不匹配。请检查菜名是否一致。")
        return None

    # ---- 检查特征列 ----
    required_cols = ["similarity", "price_norm", "cal_norm"]
    if not all(col in merged.columns for col in required_cols):
        st.error(f"训练失败：缺少特征列 {required_cols}")
        return None

    X = merged[required_cols]
    y = merged["liked"]

    if len(y.unique()) < 2:
        st.warning("反馈样本类别过少（仅有喜欢或不喜欢一种），无法训练模型。")
        return None

    # ---- 模型训练 ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    acc = model.score(X_test, y_test)
    st.success("✅ 模型已成功训练并保存！")
    st.info(f"模型在测试集准确率：{acc:.2%}")

    # 更新 session_state 里的模型
    st.session_state.model = model
    return model

# ------------------- 加载模型 -------------------
def load_or_init_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            _ = model.predict([[0, 0, 0]])  # 验证是否可用
            return model
        except Exception:
            return None
    return None

# ------------------- 推荐算法 -------------------
def smart_recommend(df, text, model=None, canteen=None):
    df, X = prepare_features(df, text)

    if model is not None:
        try:
            df["predict"] = model.predict_proba(X)[:, 1]
            df["score"] = 0.6 * df["predict"] + 0.4 * df["similarity"]
        except Exception:
            df["score"] = df["similarity"]
    else:
        df["score"] = 0.7 * df["similarity"] + 0.3 * (1 - df["price_norm"])

    if canteen and canteen != "所有食堂":
        df = df[df["canteen"] == canteen]

    return df.sort_values("score", ascending=False).head(5)

# ------------------- Streamlit 界面 -------------------
st.set_page_config(page_title="BNU 智能食堂助手", page_icon="🍱", layout="centered")
st.title("🍱 北京师范大学 · 智能食堂助手（自学习修正版）")
st.caption("💬 输入你的口味，让系统推荐最适合你的菜！")

# 输入与选择
text = st.text_input("请输入需求：", placeholder="例如：清淡低脂 或 想吃辣的")
canteen = st.selectbox("选择食堂：", ["所有食堂", "学一食堂", "学二食堂", "学三食堂", "学四食堂"])

menu_data = load_menu()
model = st.session_state.get("model", load_or_init_model())

# 推荐按钮
if st.button("🍽️ 开始推荐"):
    if not text.strip():
        st.warning("请输入需求再试！")
    else:
        recs = smart_recommend(menu_data, text, model, canteen)
        if recs.empty:
            st.info("暂无符合条件的菜。")
        else:
            st.subheader("推荐菜品 🍜")
            for i, (_, row) in enumerate(recs.iterrows()):
                with st.expander(f"{row['name']} | {row['canteen']}"):
                    st.write(f"💰 价格：¥{row['price']}")
                    st.write(f"🔥 热量：{row['calories']} kcal")
                    st.write(f"🏷️ 标签：{'、'.join(row['tags'])}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("👍 喜欢", key=f"like_{i}_{row['name']}",
                                  on_click=lambda d=row["name"]: save_feedback(d, 1))
                    with col2:
                        st.button("👎 不喜欢", key=f"dislike_{i}_{row['name']}",
                                  on_click=lambda d=row["name"]: save_feedback(d, 0))

# 重新训练按钮
if st.button("🧠 重新训练模型"):
    retrain_model(menu_data)

# 调试部分
st.divider()
if st.checkbox("显示反馈数据"):
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        st.write(df.tail(10))
        st.info(f"📊 当前反馈样本数：{len(df)} 条")
    else:
        st.write("暂无反馈记录。")

st.caption("📘 系统使用 TF-IDF + LightGBM 实现智能自学习推荐。")
