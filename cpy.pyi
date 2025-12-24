# =======================================================
# 🍱 BNU 智能食堂助手（语音输入 + 自学习 + 反馈可视化版）
# =======================================================
import os
import pandas as pd
import jieba
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr  # 🎙️ 语音输入支持

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
            pd.DataFrame(columns=["dish", "liked", "time"]).to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")

        new = pd.DataFrame([[dish_name, int(liked), datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                           columns=["dish", "liked", "time"])
        new.to_csv(FEEDBACK_FILE, mode="a", index=False, header=False, encoding="utf-8-sig")

        if liked:
            st.success(f"👍 已记录反馈：你喜欢『{dish_name}』")
        else:
            st.warning(f"👎 已记录反馈：你不喜欢『{dish_name}』")

    except Exception as e:
        st.error(f"保存反馈失败：{e}")

# ------------------- 模型训练 -------------------
def retrain_model(df):
    if not os.path.exists(FEEDBACK_FILE):
        st.warning("暂无用户反馈，无法训练模型。")
        return None

    fb = pd.read_csv(FEEDBACK_FILE)
    if fb.empty:
        st.warning("反馈数据为空，请多点几次喜欢/不喜欢。")
        return None

    df, _ = prepare_features(df, "辣")
    merged = df.merge(fb, left_on="name", right_on="dish", how="inner")

    if merged.empty:
        st.warning("反馈菜品与菜单不匹配，请检查菜名是否一致。")
        return None

    X = merged[["similarity", "price_norm", "cal_norm"]]
    y = merged["liked"]

    if len(y.unique()) < 2:
        st.warning("反馈样本类别过少，无法训练模型。")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    acc = model.score(X_test, y_test)
    st.success("✅ 模型已成功训练并保存！")
    st.info(f"模型在测试集准确率：{acc:.2%}")

    st.session_state.model = model
    return model

# ------------------- 加载模型 -------------------
def load_or_init_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            _ = model.predict([[0, 0, 0]])
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

# ------------------- 🎙️ 语音识别功能 -------------------
def record_and_recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 请开始说话（最多5秒）...")
        audio = r.listen(source, phrase_time_limit=5)
        st.info("🕓 识别中，请稍候...")
        try:
            text = r.recognize_google(audio, language="zh-CN")
            st.success(f"✅ 识别结果：{text}")
            return text
        except sr.UnknownValueError:
            st.warning("❌ 没听清，请再试一次。")
        except sr.RequestError:
            st.error("⚠️ 网络问题，无法连接到语音识别服务。")
    return ""

# ------------------- Streamlit 界面 -------------------
st.set_page_config(page_title="BNU 智能食堂助手", page_icon="🍱", layout="centered")
st.title("🍱 北京师范大学 · 智能食堂助手（语音输入修正版）")
st.caption("💬 说出或输入你的口味，让系统推荐最适合你的菜！")

# 输入区域
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_input("请输入需求：", placeholder="例如：清淡低脂 或 想吃辣的")
with col2:
    if st.button("🎤 语音输入"):
        spoken = record_and_recognize()
        if spoken:
            text = spoken

canteen = st.selectbox("选择食堂：", ["所有食堂", "学一食堂", "学二食堂", "学三食堂", "学四食堂"])

menu_data = load_menu()
model = st.session_state.get("model", load_or_init_model())

# 推荐按钮
if st.button("🍽️ 开始推荐"):
    if not text.strip():
        st.warning("请输入或语音输入你的需求再试！")
    else:
        recs = smart_recommend(menu_data, text, model, canteen)
        if recs.empty:
            st.info("暂无符合条件的菜。")
        else:
            st.subheader("🍜 推荐菜品")
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

# 重新训练模型
if st.button("🧠 重新训练模型"):
    retrain_model(menu_data)

# ------------------- 📊 可视化反馈 -------------------
st.divider()
st.header("📊 用户反馈分析")

if os.path.exists(FEEDBACK_FILE):
    fb = pd.read_csv(FEEDBACK_FILE)
    if not fb.empty:
        tab1, tab2, tab3 = st.tabs(["🥧 喜好比例", "🍱 热门菜品", "🕒 趋势分析"])

        # 喜好比例
        with tab1:
            liked_counts = fb["liked"].value_counts().rename({1: "喜欢", 0: "不喜欢"})
            fig1, ax1 = plt.subplots()
            ax1.pie(liked_counts, labels=liked_counts.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)
            st.info(f"总反馈数：{len(fb)} 条")

        # 热门菜品 TOP5
        with tab2:
            top_liked = fb[fb["liked"] == 1]["dish"].value_counts().head(5)
            if not top_liked.empty:
                fig2, ax2 = plt.subplots()
                ax2.barh(top_liked.index, top_liked.values)
                ax2.set_xlabel("喜欢次数")
                ax2.set_title("🍲 用户最喜欢的菜品 TOP5")
                st.pyplot(fig2)
            else:
                st.info("暂无喜欢反馈数据。")

        # 趋势分析
        with tab3:
            if "time" in fb.columns:
                fb["date"] = pd.to_datetime(fb["time"]).dt.date
                trend = fb.groupby(["date", "liked"]).size().unstack(fill_value=0)
                fig3, ax3 = plt.subplots()
                trend.plot(ax=ax3, marker="o")
                ax3.set_title("📅 每日反馈趋势")
                ax3.set_xlabel("日期")
                ax3.set_ylabel("反馈数量")
                st.pyplot(fig3)
            else:
                st.info("反馈数据缺少时间戳。")
    else:
        st.info("暂无反馈记录，请进行菜品反馈后查看结果。")
else:
    st.info("暂无反馈文件，请先提交一次反馈。")

st.caption("📘 本系统基于 TF-IDF + LightGBM + 语音识别 实现智能自学习推荐。")
