# =======================================================
# 🍱 BNU 智能食堂助手（语音输入 + 自学习版，无图表）
# =======================================================
import os
import pandas as pd
import jieba
import joblib
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr  # 🎙️ 语音输入支持
import altair as alt  # 📊 可视化支持

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
    corpus = [" ".join([str(row["name"]), str(row["category"])] + row["tags"]) for _, row in df.iterrows()]
    user_cut = " ".join(jieba.lcut(text))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus + [user_cut])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()

    df["similarity"] = sim
    scaler = MinMaxScaler()
    df["price_norm"] = scaler.fit_transform(df[["price"]])
    df["cal_norm"] = scaler.fit_transform(df[["calories"]])
    return df, df[["similarity", "price_norm", "cal_norm"]]

# ------------------- 加载反馈数据 -------------------
def load_feedback_data():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["dish", "liked", "time"])
    try:
        # 尝试正常读取
        return pd.read_csv(FEEDBACK_FILE)
    except pd.errors.ParserError:
        # 如果遇到列数不一致（如旧数据2列，新数据3列），尝试容错读取
        try:
            df = pd.read_csv(FEEDBACK_FILE, header=None, skiprows=1, 
                             names=["dish", "liked", "time"], engine='python')
            # 修复文件头，统一为3列
            df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
            return df
        except Exception:
            return pd.DataFrame(columns=["dish", "liked", "time"])

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

# ------------------- 清空反馈 -------------------
def clear_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        try:
            pd.DataFrame(columns=["dish", "liked", "time"]).to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
            st.success("✅ 反馈记录已清空！")
            return True
        except Exception as e:
            st.error(f"清空失败：{e}")
            return False
    return False

# ------------------- 模型训练 -------------------
def retrain_model(df):
    fb = load_feedback_data()
    if fb.empty:
        st.warning("暂无用户反馈，无法训练模型。")
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

# ------------------- Streamlit 主界面配置 -------------------
st.set_page_config(page_title="BNU 智能食堂助手", page_icon="🍱", layout="centered")

# 加载自定义 CSS
def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ------------------- 页面定义 -------------------

def landing_page():
    # 导航页专属 CSS
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #00519E 0%, #003366 100%) !important;
    }
    .big-title {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 100px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: "Microsoft YaHei", sans-serif;
    }
    .subtitle {
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 50px;
        color: #E0E0E0;
        font-weight: 300;
        letter-spacing: 2px;
    }
    /* 覆盖 Streamlit 默认按钮样式，使其更突出 */
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background-color: white;
        color: #00519E;
        font-size: 24px;
        font-weight: bold;
        padding: 15px 50px;
        border-radius: 40px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        background-color: #f8f9fa;
        color: #003366;
    }
    /* 隐藏 footer 等干扰元素 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-title">北京师范大学 · 智慧食堂</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">个性化推荐 · 语音交互 · 智能学习</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 进入点餐系统"):
            st.session_state.page = "app"
            st.rerun()

def main_app():
    # 恢复或加载主应用 CSS
    try:
        local_css(os.path.join(BASE_DIR, "assets", "style.css"))
    except FileNotFoundError:
        pass
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 导航")
        if st.button("🏠 返回首页"):
            st.session_state.page = "landing"
            st.rerun()

    st.title("🍱 北京师范大学 · 智能食堂助手")
    st.caption("💬 说出或输入你的口味，让系统推荐最适合你的菜！")

    # ------------------- 输入区域 -------------------
    # 初始化 session_state
    if "search_text" not in st.session_state:
        st.session_state.search_text = ""

    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # 增加一些垂直间距，使按钮与输入框对齐
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            if st.button("🎤 语音输入"):
                spoken = record_and_recognize()
                if spoken:
                    st.session_state.search_text = spoken
                    st.rerun()

        with col1:
            # 绑定到 session_state.search_text，允许手动修改
            text = st.text_input("请输入需求：", key="search_text", placeholder="例如：清淡低脂 或 想吃辣的")

        canteen = st.selectbox("选择食堂：", ["所有食堂", "学一食堂", "学二食堂", "学三食堂", "学四食堂"])

    menu_data = load_menu()
    model = st.session_state.get("model", load_or_init_model())

    # 推荐功能
    st.markdown("---")
    if st.button("🍽️ 开始推荐", use_container_width=True):
        if not text.strip():
            st.warning("请输入或语音输入你的需求再试！")
        else:
            recs = smart_recommend(menu_data, text, model, canteen)
            if recs.empty:
                st.info("暂无符合条件的菜。")
            else:
                st.subheader("🍜 推荐菜品")
                for i, (_, row) in enumerate(recs.iterrows()):
                    with st.expander(f"{row['name']} | {row['category']} | {row['canteen']}"):
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
                
                # ------------------- 📊 可视化分析区域 -------------------
                st.markdown("---")
                st.subheader("🔍 算法可视化解析")
                
                # 1. 推荐分数构成分析
                st.markdown("#### 1. 推荐依据：为什么推荐这些菜？")
                st.caption("基于 TF-IDF 文本相似度与 LightGBM 模型预测的综合得分")
                
                # 准备绘图数据
                chart_data = recs.copy()
                # 归一化价格和热量以便展示（反向，因为越低越好）
                chart_data["价格优势"] = 1 - chart_data["price_norm"]
                chart_data["低卡优势"] = 1 - chart_data["cal_norm"]
                chart_data["文本匹配"] = chart_data["similarity"]
                
                # 如果有模型预测分
                if "predict" in chart_data.columns:
                    chart_data["模型偏好"] = chart_data["predict"]
                    cols_to_plot = ["name", "文本匹配", "模型偏好", "价格优势", "低卡优势"]
                else:
                    cols_to_plot = ["name", "文本匹配", "价格优势", "低卡优势"]
                    
                chart_df = chart_data[cols_to_plot].melt("name", var_name="指标", value_name="得分")
                
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("得分:Q", stack="zero"),
                    y=alt.Y("name:N", sort="-x", title="菜品名称"),
                    color=alt.Color("指标:N", scale=alt.Scale(scheme="set2")),
                    tooltip=["name", "指标", alt.Tooltip("得分", format=".2f")]
                ).properties(height=300)
                
                st.altair_chart(chart, use_container_width=True)

                # 2. 价格与热量分布图
                st.markdown("#### 2. 性价比分析：价格 vs 热量")
                
                # 基础图表
                base = alt.Chart(recs).encode(
                    x=alt.X("price:Q", title="价格 (元)", scale=alt.Scale(zero=False, padding=1)),
                    y=alt.Y("calories:Q", title="热量 (kcal)", scale=alt.Scale(zero=False, padding=1)),
                    tooltip=["name", "category", "price", "calories", alt.Tooltip("score", format=".2f")]
                )

                # 散点图
                points = base.mark_circle(size=150, opacity=0.8, stroke='black', strokeWidth=1).encode(
                    color=alt.Color("category:N", legend=alt.Legend(title="类别"), scale=alt.Scale(scheme="category10")),
                    size=alt.Size("score:Q", legend=None, scale=alt.Scale(range=[100, 300]))
                )

                # 文字标签
                text_labels = base.mark_text(align='left', dx=12, dy=-5, fontSize=12).encode(
                    text="name",
                    color=alt.value("black")
                )

                # 组合图表
                final_chart = (points + text_labels).interactive().properties(
                    title="推荐菜品分布 (点越大推荐分越高)"
                )
                
                st.altair_chart(final_chart, use_container_width=True)

    # 重新训练模型
    with st.expander("⚙️ 高级选项：重新训练模型 & 模型透视"):
        st.info("当反馈数据积累较多时，点击下方按钮更新推荐模型。")
        if st.button("🧠 重新训练模型", use_container_width=True):
            retrain_model(menu_data)
        
        # 模型特征重要性可视化
        if model is not None and hasattr(model, "feature_importances_"):
            st.markdown("#### 🧠 模型特征重要性")
            st.caption("模型认为哪些因素最影响你的喜好？")
            feat_imp = pd.DataFrame({
                "Feature": ["文本相似度", "价格因素", "热量因素"],
                "Importance": model.feature_importances_
            })
            imp_chart = alt.Chart(feat_imp).mark_bar().encode(
                x="Importance:Q",
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.Color("Feature:N", legend=None)
            )
            st.altair_chart(imp_chart, use_container_width=True)

    # ------------------- 用户反馈简表 -------------------
    st.divider()
    st.subheader("📋 用户反馈记录")

    fb = load_feedback_data()
    if not fb.empty:
        with st.expander("📄 展开查看最近反馈", expanded=False):
            st.dataframe(fb.tail(10), use_container_width=True)
            if st.button("🗑️ 清空所有反馈记录"):
                if clear_feedback_data():
                    st.rerun()
        st.caption(f"共 {len(fb)} 条反馈记录。")
    else:
        st.info("暂无反馈记录，请先点赞或点踩菜品。")

    st.caption("📘 本系统基于 TF-IDF + LightGBM + 语音识别 实现智能自学习推荐。")

# ------------------- 主程序入口 -------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    landing_page()
else:
    main_app()
