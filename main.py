import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025       # 수업 기준 기간의 끝
MIN_DAYS = 300         # 이보다 관측일이 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온에 직선(선형 회귀)을 맞추고, 고른 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(끝 연도 기준 달력상 20년)만으로 구한 기울기
RECENT_YEARS = 20
recent_start = end_year - RECENT_YEARS + 1
recent = yearly[yearly["연도"] >= recent_start]
if len(recent) >= 3:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_recent, intercept_recent = np.polyfit(xr, yr, 1)
    r_recent = float(np.corrcoef(xr, yr)[0, 1])
else:
    slope_recent = intercept_recent = r_recent = None

# --- 직선 정보 ---
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.write(
    f"기준 기간은 {LAST_YEAR}년까지이며, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# --- 기울기: 100년에 몇 도 오르는가 ---
per100 = slope * 100
st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0 0.4rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">직선의 기울기 (전체 기간 {start_year}~{end_year}년)</div>
        <div style="font-size:3.2rem; font-weight:800; line-height:1.1;">100년에 {per100:+.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.8rem; border:1px solid rgba(128,128,128,0.4); border-radius:12px;">
            <div style="opacity:0.7;">전체 기간 ({start_year}~{end_year}년, {n_years}개 해)</div>
            <div style="font-size:2.4rem; font-weight:800;">100년에 {per100:+.2f} ℃</div>
            <div style="opacity:0.7;">상관계수 {r:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_recent:
    if slope_recent is not None:
        per100_recent = slope_recent * 100
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.8rem; border:1px solid rgba(128,128,128,0.4); border-radius:12px;">
                <div style="opacity:0.7;">최근 20년 ({int(recent['연도'].min())}~{end_year}년, {len(recent)}개 해)</div>
                <div style="font-size:2.4rem; font-weight:800;">100년에 {per100_recent:+.2f} ℃</div>
                <div style="opacity:0.7;">상관계수 {r_recent:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("최근 20년 안에 데이터가 3개 해 미만이라 기울기를 구할 수 없습니다.")

if slope_recent is not None:
    diff = (slope_recent - slope) * 100
    st.caption(
        f"최근 20년 기울기는 전체 기간보다 100년당 {diff:+.2f}℃ 차이가 납니다. "
        "20년은 표본이 적고 해마다 변동이 커서, 최근 기울기는 흔들림이 큽니다."
    )

# --- 슬라이더와 예상 기온 ---
year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 상관계수 ---
st.metric("상관계수 (연도 vs 연평균기온)", f"{r:.3f}")
st.caption(f"전체 기간 회귀식: 기온 = {slope:.4f} × 연도 + ({intercept:.2f})  (예상 기온은 이 직선으로 계산)")

# --- 그래프 ---
x_line = np.array([min(1900, start_year), 2100], dtype=float)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        hovertemplate="%{x}년<br>%{y:.2f}℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="회귀 직선 (전체 기간)",
        line=dict(width=3, dash="solid"),
    )
)
if slope_recent is not None:
    x_recent = np.array([recent_start, 2100], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=x_recent,
            y=slope_recent * x_recent + intercept_recent,
            mode="lines",
            name="회귀 직선 (최근 20년)",
            line=dict(width=3, dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년<br>예상 %{y:.2f}℃<extra></extra>",
    )
)
fig.update_layout(
    title="서울 연평균기온 산점도와 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(℃)"}).round({"연평균기온(℃)": 2}),
        use_container_width=True,
        hide_index=True,
    )
