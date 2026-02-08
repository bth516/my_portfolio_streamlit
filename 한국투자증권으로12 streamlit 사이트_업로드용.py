#실행#
#streamlit run 9_Allweather_portfolio_auto\"한국투자증권으로12 streamlit 사이트_업로드용.py"
# streamlit_portfolio_app.py
import streamlit as st
import requests, json, yaml, os, time, warnings
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import mplfinance as mpf
from fear_and_greed import get as get_cnn_index
import plotly.express as px #

# ✅ 1. 세션 상태 초기화
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "mode" not in st.session_state:
    st.session_state.mode = None 
if "overseas" not in st.session_state:
    st.session_state.overseas = []
if "domestic" not in st.session_state:
    st.session_state.domestic = []
if "usd_cash" not in st.session_state:
    st.session_state.usd_cash = 0.0

if "krw_cash" not in st.session_state:
    st.session_state.krw_cash = 0.0
if "macro_view" not in st.session_state:
    st.session_state.macro_view = None

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================
# Streamlit 기본 설정
# ========================
st.set_page_config(layout="wide")
top_l, top_r = st.columns([5, 1])
with top_l:
    st.title("📊 투자 포트폴리오")
with top_r:
    account_type = st.selectbox("계좌", ["종합계좌", "ISA계좌"], label_visibility="collapsed")

#API불러오기 **SCRET을사용**
try:
    if account_type == "종합계좌":
        APP_KEY = st.secrets["APP_KEY"]
        APP_SECRET = st.secrets["APP_SECRET"]
        CANO = st.secrets["CANO"]
        ACNT_PRDT_CD = st.secrets["ACNT_PRDT_CD"]
    else:  # ISA 계좌
        APP_KEY = st.secrets["ISA_APP_KEY"]
        APP_SECRET = st.secrets["ISA_APP_SECRET"]
        CANO = st.secrets["ISA_CANO"]
        ACNT_PRDT_CD = st.secrets["ISA_ACNT_PRDT_CD"]
    
    # 공통 항목
    URL_BASE = st.secrets["URL_BASE"]

except KeyError as e:
    st.error(f"Streamlit Secrets 설정이 누락되었습니다: {e}")
    st.info("Advanced Settings의 Secrets 항목에 해당 키가 등록되어 있는지 확인하세요.")
    st.stop()
except Exception as e:
    st.error(f"설정 정보를 가져오는 중 오류가 발생했습니다: {e}")
    st.stop()
    
# ========================
# 유틸리티 및 보조지표 함수 (원복)
# ========================
def get_access_token():
    token_file = f"token_{APP_KEY}_{CANO}.json"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            data = json.load(f)
            if time.time() - data["created_at"] < 4 * 60 * 60:
                return data["access_token"]
    res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers={"content-type": "application/json"}, 
                        json={"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET})
    token = res.json()["access_token"]
    with open(token_file, "w") as f:
        json.dump({"access_token": token, "created_at": time.time()}, f)
    return token

def load_weights(file_name, default_data):
    if os.path.exists(file_name):
        return pd.read_csv(file_name).set_index('Asset')['Weight'].to_dict()
    pd.DataFrame(list(default_data.items()), columns=['Asset', 'Weight']).to_csv(file_name, index=False)
    return default_data

OVERSEAS_WEIGHT_FILE, DOMESTIC_WEIGHT_FILE = "weights_overseas.csv", "weights_domestic.csv"
DEFAULT_OV = {"QQQM": 35.0, "SPYM": 35.0, "XYLD": 30.0}
DEFAULT_DOM = {"0072R0": 20.0, "486290": 20.0, "458730": 20.0, "379810": 20.0, "379800": 20.0}

TARGET_WEIGHTS = load_weights(OVERSEAS_WEIGHT_FILE, DEFAULT_OV)
TARGET_WEIGHTS2 = load_weights(DOMESTIC_WEIGHT_FILE, DEFAULT_DOM)

@st.cache_data(ttl=3600)
def get_usdkrw():
    df = yf.download("KRW=X", period="5d", progress=False)
    return float(df["Close"].iloc[-1])

usdkrw = get_usdkrw()

def money_fmt(val):
    if st.session_state.mode == "domestic": return f"₩{val:,.0f}"
    return f"${val:,.2f}"

def signal_up_down(val, standard=0): return "상승신호" if val > standard else "하락신호"
def cci_state(val):
    if val > 100: return "과매수구간"
    elif val < -100: return "과매도구간"
    return "중립"
def willr_state(val):
    if val > -20: return "과매수"
    elif val < -80: return "과매도"
    return "중립"

@st.cache_data(ttl=3600)
def load_price_data(ticker, days=500):
    y_ticker = f"{ticker}.KS" if st.session_state.mode == "domestic" else ticker
    df = yf.download(tickers=y_ticker, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

def calc_indicators(df):
    close, high, low = df["Close"].squeeze(), df["High"].squeeze(), df["Low"].squeeze()
    macd_ind = ta.trend.MACD(close=close)
    dmi = ta.trend.ADXIndicator(high=high, low=low, close=close)
    psar_val = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().iloc[-1]
    return {
        "macd": macd_ind.macd().iloc[-1], "macd_signal": macd_ind.macd_signal().iloc[-1],
        "cci": ta.trend.CCIIndicator(high=high, low=low, close=close).cci().iloc[-1],
        "willr": ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().iloc[-1],
        "plus_di": dmi.adx_pos().iloc[-1], "minus_di": dmi.adx_neg().iloc[-1],
        "psar": "상승추세" if psar_val < close.iloc[-1] else "하락추세"
    }

def calc_ma_trend(df):
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1]
    close = df["Close"].iloc[-1]
    trend = "🔥 강한 상승" if ma5 > ma200 and close > ma200 else "⬇️ 약세/조정"
    return ma5, ma200, trend

# ✅ 종목코드와 종목명을 연결할 전역 맵 (국내용)
if "ticker_name_map" not in st.session_state:
    st.session_state.ticker_name_map = {}

# ========================
# API 데이터 처리
# ========================
@st.cache_data(ttl=300)
def get_overseas_balance(access_token):
    url = f"{URL_BASE}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers = {"authorization": f"Bearer {access_token}", "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": "TTTS3012R"}
    all_items = []
    ctx_fk100, ctx_nk100 = "", ""
    while True:
        params = {
            "CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD, "INQR_DVSN": "02", "AFHR_FLPR_YN": "N",
            "UNPR_DVSN": "01", "PRCS_DVSN": "01", "OVRS_EXCG_CD": "", "TR_CRCY_CD": "USD",
            "CTX_AREA_FK100": ctx_fk100, "CTX_AREA_NK100": ctx_nk100,
            "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
        }
        res = requests.get(url, headers=headers, params=params).json()
        if res.get("rt_cd") != "0":
            st.error("❌ 해외잔고 조회 실패")
            return []
        output1 = res.get("output1", [])

        if isinstance(output1, list): all_items.extend(output1)
        ctx_fk100, ctx_nk100 = res.get("ctx_area_fk100"), res.get("ctx_area_nk100")
        if not ctx_fk100 and not ctx_nk100: break

    return all_items

@st.cache_data(ttl=300)
def get_overseas_cash(access_token):
    url = f"{URL_BASE}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers = {"authorization": f"Bearer {access_token}", "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": "TTTS3007R"}
    params = {"CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD, "OVRS_EXCG_CD": "NAS", "OVRS_ORD_UNPR": "1", "ITEM_CD": "AAPL", "ORD_DVSN": "00"}
    res = requests.get(url, headers=headers, params=params).json()
    return float(res.get("output", {}).get("ord_psbl_frcr_amt", 0))

def get_domestic_balance(access_token):
    url = f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {"authorization": f"Bearer {access_token}", "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": "TTTC8434R"}
    params = {"CANO": CANO, "ACNT_PRDT_CD": ACNT_PRDT_CD, "INQR_DVSN": "02", "AFHR_FLPR_YN": "N", "FUND_STTL_ICLD_YN": "N", "OFL_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "UNPR_DVSN": "01", "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
    res = requests.get(url, headers=headers, params=params).json()
    
    stocks = res.get("output1", [])
    # 💡 로드 시점에 종목코드(pdno)와 종목명(prdt_name) 매핑 저장
    for s in stocks:
        st.session_state.ticker_name_map[s['pdno']] = s['prdt_name']
        
    return stocks, res.get("output2", [{}])[0]

# ========================
# 🔘 컨트롤 및 지표 (UI)
# ========================
st.subheader("🔘 컨트롤")
c1, c2, c3, c4 = st.columns(4)

def reset_all():
    st.session_state.overseas, st.session_state.domestic = [], []
    st.session_state.usd_cash, st.session_state.krw_cash = 0.0, 0.0
    st.session_state.mode, st.session_state.macro_view = None, None

with c1:
    if st.button("🌎 해외 종목 Load", use_container_width=True):
        # ✅ ISA 계좌일 경우 해외 로직 실행 전 차단
        if account_type == "ISA계좌":
            st.warning("⚠️ ISA 계좌는 해외 잔고가 없습니다. (해외 주식 거래 불가 계좌)")
        else:
            reset_all()
            token = get_access_token()
            # 해외 잔고 불러오기 시도
            st.session_state.overseas = get_overseas_balance(token)
            st.session_state.usd_cash = get_overseas_cash(token)
            
            # 종목이 하나도 없을 경우 처리
            if not st.session_state.overseas and st.session_state.usd_cash == 0:
                st.info("조회된 해외 자산이 없습니다.")
            else:
                st.session_state.mode = "overseas"
                st.success("해외 로드 완료")

with c2:
    if st.button("🇰🇷 국내 종목 Load", use_container_width=True):
        # ✅ 종합계좌인데 국내 로드를 시도할 경우 안내 멘트 출력
        if account_type == "종합계좌":
            reset_all() # 기존 데이터 초기화
            st.info("💡 현재 종합계좌는 국내 주식을 사용하고 있지 않습니다. 종합계좌는 해외자산만 운용 중입니다.")
        else:
            # ISA계좌 등 국내 자산이 있는 경우 정상 로직 실행
            reset_all()
            token = get_access_token()
            dom_stocks, dom_summary = get_domestic_balance(token)
            
            # 종목이 실제로 있는지 확인
            if not dom_stocks and float(dom_summary.get("dnca_tot_amt", 0)) == 0:
                st.warning("조회된 국내 자산이 없습니다.")
            else:
                st.session_state.domestic = dom_stocks
                st.session_state.krw_cash = float(dom_summary.get("dnca_tot_amt", 0))
                st.session_state.mode = "domestic"
                st.success("국내 로드 완료")

with c3:
    selected_macro = st.selectbox("지표 선택", ["선택하세요", "VIX (6개월)", "CNN Fear & Greed"], label_visibility="collapsed")
    if st.button("📊 지표 불러오기", use_container_width=True):
        st.session_state.macro_view = selected_macro

if st.session_state.macro_view and st.session_state.macro_view != "선택하세요":
    st.divider()
    if "VIX" in st.session_state.macro_view:
        vix_df = yf.Ticker("^VIX").history(period="6mo")
        st.metric(label="현재 VIX 지수", value=f"{vix_df['Close'].iloc[-1]:.2f}", delta=f"{vix_df['Close'].iloc[-1] - vix_df['Close'].iloc[-2]:.2f}")
        st.line_chart(vix_df["Close"])
    elif "CNN" in st.session_state.macro_view:
        try:
            cnn_data = get_cnn_index()
            st.metric(label="CNN Fear & Greed", value=f"{int(cnn_data.value)}", delta=cnn_data.description)
            st.progress(int(cnn_data.value) / 100)
        except: st.error("CNN 지수 로드 실패")

with c4:
    if st.button("🧹 초기화", use_container_width=True):
        reset_all()
        st.rerun()

if st.session_state.mode is None:
    st.info("상단의 버튼을 눌러 자산을 불러주세요")
    st.stop()

# ========================
# 🥧 포트폴리오 계산
# ========================
portfolio_value = {}
ticker_to_name = {} # 코드 -> 이름 매핑용
name_to_ticker = {} # 이름 -> 코드 매핑용

if st.session_state.mode == "overseas":
    portfolio_value = {"USD Cash": st.session_state.usd_cash}
    total_eval, total_buy = 0, 0
    for s in st.session_state.overseas:
        code = s.get("ovrs_pdno")
        eval_amt = float(s.get("ovrs_stck_evlu_amt", "0").replace(",", "") or 0)
        buy_amt = float(s.get("frcr_pchs_amt1", "0").replace(",", "") or 0)
        if eval_amt > 0:
            portfolio_value[code] = eval_amt
            total_eval += eval_amt
            total_buy += buy_amt
            ticker_to_name[code] = code
            name_to_ticker[code] = code
    summary = {"eval": total_eval, "buy": total_buy, "profit": total_eval - total_buy, "cash": st.session_state.usd_cash}
else:
    portfolio_value = {"KRW Cash": st.session_state.krw_cash}
    total_eval, total_buy, total_profit = st.session_state.krw_cash, 0, 0
    for s in st.session_state.domestic:
        code = s.get("pdno")
        name = s.get("prdt_name")
        eval_amt, buy_amt, profit = float(s.get("evlu_amt", 0)), float(s.get("pchs_amt", 0)), float(s.get("evlu_pfls_amt", 0))
        if eval_amt > 0:
            portfolio_value[name] = eval_amt
            ticker_to_name[code] = name
            name_to_ticker[name] = code
            total_eval += eval_amt
            total_buy += buy_amt
            total_profit += profit
    summary = {"eval": total_eval, "buy": total_buy, "profit": total_profit, "cash": st.session_state.krw_cash}

# ========================
# 💰 포트폴리오 요약 
# ========================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("🥧 비중")
    if portfolio_value:
        df_pie = pd.DataFrame(list(portfolio_value.items()), columns=['종목', '금액'])
        fig = px.pie(df_pie, values='금액', names='종목', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("💰 포트폴리오 요약")
    
    asset_details = []
    total_eval_for_pct = summary['eval']
    
    if st.session_state.mode == "overseas":
        for s in st.session_state.overseas:
            code = s.get("ovrs_pdno")
            # 평가금액
            eval_amt = float(str(s.get("ovrs_stck_evlu_amt", "0")).replace(",", "") or 0)
            
            if eval_amt > 0:
                # 💡 사용자가 확인한 필드명으로 정확히 매핑
                pchs_amt = float(str(s.get("frcr_pchs_amt1", "0")).replace(",", "") or 0)     # 매입금액
                avg_price = float(str(s.get("pchs_avg_pric", "0")).replace(",", "") or 0)    # 매입단가
                now_price = float(str(s.get("now_pric2", "0")).replace(",", "") or 0)       # 현재가 (now_pric2)
                qty = float(str(s.get("ovrs_cblc_qty", "0")).replace(",", "") or 0)         # 보유수량 (ovrs_cblc_qty)
                
                # 수익 및 비중 계산
                profit = eval_amt - pchs_amt
                rate = (profit / pchs_amt * 100) if pchs_amt > 0 else 0
                pct = (eval_amt / total_eval_for_pct * 100) if total_eval_for_pct > 0 else 0
                
                asset_details.append({
                    "name": code, "pct": pct, "rate": rate, "profit": profit,
                    "avg": avg_price, "now": now_price, "qty": qty
                })
        
        # 해외 요약 출력 (기존 유지)
        s = summary
        total_eval_sum = s['eval'] + s['cash']
        st.metric("총 평가자산 (USD)", f"${total_eval_sum:,.2f}")
        st.caption(f"(₩{total_eval_sum * usdkrw:,.0f})")
        st.metric("총 투자원금 (USD)", f"${s['buy']:,.2f}")
        st.caption(f"(₩{s['buy'] * usdkrw:,.0f})")
        st.metric("현금 (USD)", f"${s['cash']:,.2f}")
        st.caption(f"(₩{s['cash'] * usdkrw:,.0f})")
        st.metric("총 평가손익 (USD)", f"${s['profit']:,.2f}")
        st.caption(f"(₩{s['profit'] * usdkrw:,.0f})")
        st.metric("수익률", f"{(s['profit']/s['buy']*100 if s['buy']>0 else 0):+.2f}%")

    else:
        # 국내 종목 상세 (기존 로직 유지)
        for s in st.session_state.domestic:
            name = s.get("prdt_name")
            eval_amt = float(s.get("evlu_amt", 0))
            if eval_amt > 0:
                asset_details.append({
                    "name": name,
                    "pct": (eval_amt / total_eval_for_pct * 100) if total_eval_for_pct > 0 else 0,
                    "rate": float(s.get("evlu_pfls_rt", 0)),
                    "profit": float(s.get("evlu_pfls_amt", 0)),
                    "avg": float(s.get("pchs_avg_pric", 0)),
                    "now": float(s.get("prpr", 0)),
                    "qty": float(s.get("hldg_qty", 0))
                })

        s = summary
        st.metric("총 투자원금", money_fmt(s["buy"]))
        st.metric("총 평가자산", money_fmt(s["eval"]))
        st.metric("현금", money_fmt(s["cash"]))
        st.metric("총 평가손익", money_fmt(s["profit"]))
        st.metric("수익률", f"{(s['profit']/s['buy']*100 if s['buy']>0 else 0):+.2f}%")

    # --- 🟢 '개별종목현황' 버튼 ---
    st.divider()
    with st.expander("📜 개별종목현황 확인하기"):
        if not asset_details:
            st.warning("보유 종목 정보를 불러올 수 없습니다.")
        else:
            for item in asset_details:
                t_dict = TARGET_WEIGHTS if st.session_state.mode == "overseas" else TARGET_WEIGHTS2
                target_w = t_dict.get(item['name'], 0)
                
                st.markdown(f"""
**{item['name']}**
- 비중 : {item['pct']:.1f}% (목표 {target_w:.1f}%)
- 수익률 : {item['rate']:+.2f}%
- 평가손익 : {money_fmt(item['profit'])}
- 평균단가 : {money_fmt(item['avg'])}
- 현재가 : {money_fmt(item['now'])}
- 보유수량 : {item['qty']:.1f}주
""")
                st.write("---")

# ========================
# 🧭 리밸런싱 제안 (종목명 출력 고정)
# ========================
st.divider()
st.subheader("🧭 리밸런싱 제안")
target_dict = TARGET_WEIGHTS if st.session_state.mode == "overseas" else TARGET_WEIGHTS2
total_all = sum(portfolio_value.values())
for asset_key, target in target_dict.items():
    # 💡 설정에 코드가 있든 이름이 있든, 화면에는 이름으로 표시
    display_name = ticker_to_name.get(asset_key, asset_key)
    cur_val = portfolio_value.get(display_name, 0)
    cur_pct = (cur_val / total_all * 100) if total_all > 0 else 0
    if abs(cur_pct - target) > 1.0:
        real_ticker = name_to_ticker.get(asset_key, asset_key)
        df_p = load_price_data(real_ticker, 5)
        if not df_p.empty:
            now_p = df_p["Close"].iloc[-1]
            diff_qty = (total_all * target / 100 - cur_val) / now_p
            color = "🟢 매수" if diff_qty > 0 else "🔴 매도"
            st.write(f"{color} **{display_name}**: 약 {abs(diff_qty):.2f}주 (현재 {cur_pct:.1f}% → 목표 {target}%)")

# ========================
# 📌 분석 도구 (보조지표 & 캔들차트 원상복구!)
# ========================
st.divider()
st.subheader("📌 분석 도구")
b1, b2 = st.columns(2)

with b1:
    if st.button("🔍 보조지표 보기", use_container_width=True):
        tickers = [t for t in target_dict.keys() if "Cash" not in t]
        for t_key in tickers:
            real_ticker = name_to_ticker.get(t_key, t_key)
            display_name = ticker_to_name.get(real_ticker, t_key)
            df = load_price_data(real_ticker, 500)
            if df is None or df.empty: continue
            ind = calc_indicators(df)
            ma5, ma200, trend = calc_ma_trend(df)
            st.markdown(f"""
**<{display_name}>** ({real_ticker}) MACD : {ind['macd']:.2f} ({signal_up_down(ind['macd'])})  
MACD SIGNAL : {ind['macd_signal']:.2f} ({signal_up_down(ind['macd_signal'])})  
CCI : {ind['cci']:.2f} ({cci_state(ind['cci'])})  
WILL%R : {ind['willr']:.2f} ({willr_state(ind['willr'])})  
DMI : +DI {ind['plus_di']:.2f} / -DI {ind['minus_di']:.2f} ({"상승추세" if ind['plus_di'] > ind['minus_di'] else "하락추세"})  
PSAR : {ind['psar']}

MA200 : {ma200:.2f}  MA5 : {ma5:.2f}  **추세 : {trend}**
""")
            st.divider()

with b2:
    days = st.selectbox("📆 캔들 기간 선택", [500, 400, 300, 200, 100, 50])
    if st.button("📈 캔들차트 보기", use_container_width=True):
        tickers = [t for t in target_dict.keys() if "Cash" not in t]
        for t_key in tickers:
            real_ticker = name_to_ticker.get(t_key, t_key)
            display_name = ticker_to_name.get(real_ticker, t_key)
            df = load_price_data(real_ticker, days)
            if df is None or df.empty:
                st.warning(f"{display_name} 데이터 없음")
                continue
            st.write(f"📊 **{display_name}** ({real_ticker}) - 최근 {days}일")
            fig, _ = mpf.plot(df, type="candle", volume=True, style="yahoo", returnfig=True)
            st.pyplot(fig)

# ========================
# ⚙️ 목표 설정 (Dynamic)
# ========================
st.divider()
st.subheader("⚙️ 포트폴리오 종목 및 비중 설정")
st.info("💡 표 하단의 (+) 버튼으로 채권(TLT 등)을 추가할 수 있습니다.")
ec1, ec2 = st.columns(2)
with ec1:
    st.write("🌎 해외 포트폴리오")
    ov_df = pd.DataFrame(list(TARGET_WEIGHTS.items()), columns=['Asset', 'Weight'])
    new_ov = st.data_editor(ov_df, key="ed_ov", hide_index=True, use_container_width=True, num_rows="dynamic")
    if st.button("해외 설정 저장"):
        new_ov.to_csv(OVERSEAS_WEIGHT_FILE, index=False)
        st.success("해외 설정 저장 완료! (새로고침 시 반영)")
with ec2:
    st.write("🇰🇷 국내 포트폴리오")
    dm_df = pd.DataFrame(list(TARGET_WEIGHTS2.items()), columns=['Asset', 'Weight'])
    new_dm = st.data_editor(dm_df, key="ed_dom", hide_index=True, use_container_width=True, num_rows="dynamic")
    if st.button("국내 설정 저장"):
        new_dm.to_csv(DOMESTIC_WEIGHT_FILE, index=False)

        st.success("국내 설정 저장 완료! (새로고침 시 반영)")
