import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 페이지 기본 설정
st.set_page_config(page_title="배송 지연 예측 프로그램", page_icon="📦", layout="wide")
st.title("📦 이커머스 배송 지연 예측 프로그램")
st.markdown("주문 및 배송 조건 데이터를 분석하여 **배송 지연 여부**를 예측하는 머신러닝 웹 앱입니다.")

# 데이터 불러오기 (인터넷에서 자동으로 다운로드)
@st.cache_data
def load_data():
    # 인터넷상의 공개된 데이터 깃허브 Raw 링크를 통해 파일을 자동으로 불러옵니다.
    url = "https://raw.githubusercontent.com/salsmardhiyah/EcommerceShippingData/master/E-Commerce%20Shipping%20Data-Train.csv"
    
    try:
        df = pd.read_csv(url)
    except:
        # 혹시 인터넷 링크 접속이 원활하지 않을 경우를 대비해 로컬 파일도 탐색
        df = pd.read_csv("Train.csv")
        
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3차시 분량 탭 구성
tab1, tab2, tab3 = st.tabs([
    "📊 1차시: 데이터 탐색 및 시각화",
    "🤖 2차시: 머신러닝 모델 학습 및 평가",
    "🚀 3차시: 실시간 배송 지연 예측"
])

# ==========================================
# 1차시: 데이터 탐색 및 시각화
# ==========================================
with tab1:
    st.header("1차시: 데이터 확인 및 시각화")
    st.write(f"전체 데이터 크기: **{df.shape[0]}행 × {df.shape[1]}열**")
    
    st.subheader("1) 데이터 미리보기")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2) 배송 지연 여부 분포 (타겟 변수)")
        st.caption("0: 정시 도착(정상) / 1: 배송 지연")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df["Reached.on.Time_Y.N"].value_counts().sort_index().plot(
            kind="bar", color=["#4CAF50", "#FF5252"], ax=ax1
        )
        ax1.set_xticklabels(["정상(0)", "지연(1)"], rotation=0)
        ax1.set_ylabel("주문 건수")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("3) 배송 수단별 주문 건수")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        df["Mode_of_Shipment"].value_counts().plot(
            kind="bar", color="#2196F3", ax=ax2
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.set_ylabel("주문 건수")
        st.pyplot(fig2)

# ==========================================
# 2차시: 머신러닝 모델 학습 및 평가
# ==========================================
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=["Reached.on.Time_Y.N"])
y = df_encoded["Reached.on.Time_Y.N"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model(X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf

model = train_model(X_train, y_train)

with tab2:
    st.header("2차시: 분류 모델 학습 및 성능 평가")
    st.write("사용 모델: **랜덤 포레스트 분류기 (RandomForestClassifier)**")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric(label="테스트 데이터 예측 정확도 (Accuracy)", value=f"{acc * 100:.2f}%")
    
    st.subheader("어떤 조건이 배송 지연에 가장 큰 영향을 미칠까? (특성 중요도)")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    importances.tail(8).plot(kind="barh", color="#FF9800", ax=ax3)
    ax3.set_xlabel("중요도")
    st.pyplot(fig3)

# ==========================================
# 3차시: 실시간 배송 지연 예측 웹 앱
# ==========================================
with tab3:
    st.header("3차시: 주문 조건 입력 및 지연 여부 예측")
    st.write("아래에 주문 정보를 입력한 뒤 예측 버튼을 눌러보세요.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        warehouse = st.selectbox("창고 블록 (Warehouse_block)", ["A", "B", "C", "D", "F"])
        shipment = st.selectbox("배송 수단 (Mode_of_Shipment)", ["Ship", "Flight", "Road"])
        gender = st.selectbox("고객 성별 (Gender)", ["F", "M"])
    with c2:
        calls = st.slider("고객센터 문의 횟수 (Customer_care_calls)", 2, 7, 4)
        rating = st.slider("고객 평점 (Customer_rating)", 1, 5, 3)
        prior_purchases = st.number_input("이전 구매 횟수 (Prior_purchases)", 2, 10, 3)
    with c3:
        cost = st.number_input("상품 가격 ($)", 90, 350, 200)
        importance = st.selectbox("상품 중요도 (Product_importance)", ["low", "medium", "high"])
        discount = st.slider("할인율 (%)", 1, 65, 10)
        weight = st.number_input("상품 무게 (g)", 1000, 8000, 3000)

    if st.button("🔍 배송 지연 여부 예측하기", use_container_width=True):
        input_raw = pd.DataFrame([{
            "Warehouse_block": warehouse,
            "Mode_of_Shipment": shipment,
            "Customer_care_calls": calls,
            "Customer_rating": rating,
            "Cost_of_the_Product": cost,
            "Prior_purchases": prior_purchases,
            "Product_importance": importance,
            "Gender": gender,
            "Discount_offered": discount,
            "Weight_in_gms": weight
        }])
        
        input_encoded = pd.get_dummies(input_raw)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        pred = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]
        
        st.divider()
        if pred == 1:
            st.error(f"⚠️ **배송 지연이 예상됩니다!** (지연 확률: {proba[1]*100:.1f}%)")
            st.write("할인율이 높거나 무게/배송 수단 등의 영향으로 예정일보다 늦게 도착할 가능성이 높습니다.")
        else:
            st.success(f"✅ **정시 도착(정상 배송)이 예상됩니다.** (정상 도착 확률: {proba[0]*100:.1f}%)")
