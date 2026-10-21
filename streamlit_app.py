"""Korean single-candidate swing scanner. Research only; never places orders."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
import pandas as pd
import streamlit as st

APP_TITLE = "알려줘 · 주식 스윙 분석기"
DEFAULT_CAPITAL = 200_000
RISK_FRACTION = 0.01
MAX_DATA_AGE_DAYS = 5
MIN_HISTORY_ROWS = 60
MIN_TURNOVER_WON = 1_000_000_000
MIN_TURNOVER_RATIO = 1.5
MIN_VOLUME_RATIO = 1.2
MAX_DAILY_CHANGE_PCT = 8.0

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="centered")
st.title("📈 주식 스윙 분석기")
st.caption("국내 주식 · 규칙 기반 분석 · 자동 주문 없음")
st.warning(
    "교육·조사용 도구입니다. 수익을 보장하지 않으며, 데이터 지연·가격 급변·갭 하락·수수료·세금 때문에 "
    "실제 결과가 계산과 달라질 수 있습니다. 주문 전 최신 호가와 공시를 직접 확인하세요."
)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@st.cache_data(ttl=900, show_spinner=False)
def get_listing(market: str) -> pd.DataFrame:
    """Return a cleaned listing. Raises a useful error if the provider format changes."""
    df = fdr.StockListing(market)
    if df is None or df.empty:
        raise RuntimeError(f"{market} 상장 종목 목록이 비어 있습니다. 데이터 제공처 응답을 확인하세요.")
    df = df.copy()
    # FinanceDataReader versions/providers differ: KRX listings often use Code,
    # while other listing formats use Symbol. Normalize both to Symbol.
    code_col = next((c for c in ("Symbol", "Code", "종목코드", "단축코드") if c in df.columns), None)
    name_col = next((c for c in ("Name", "종목명", "회사명") if c in df.columns), None)
    if code_col is None or name_col is None:
        raise RuntimeError(
            "상장 목록의 열 이름이 예상과 다릅니다. "
            f"받은 열: {', '.join(map(str, df.columns[:15]))}"
        )
    if code_col != "Symbol":
        df["Symbol"] = df[code_col]
    if name_col != "Name":
        df["Name"] = df[name_col]
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Symbol"].str.fullmatch(r"\d{6}", na=False)]
    df = df[(df["Name"] != "") & (df["Name"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["Symbol"])
    if "Marcap" in df.columns:
        df["Marcap"] = pd.to_numeric(df["Marcap"], errors="coerce")
        df = df.sort_values("Marcap", ascending=False, na_position="last")
    return df.reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def get_history(code: str, start: str, end_exclusive: str) -> pd.DataFrame:
    """Fetch daily OHLCV; normalize numeric columns and remove malformed rows."""
    df = fdr.DataReader(code, start, end_exclusive)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])
    df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Volume"] >= 0)]
    # Duplicate timestamps can make rolling calculations ambiguous.
    return df[~df.index.duplicated(keep="last")]


def analyze_one(code: str, name: str, hist: pd.DataFrame, today: date) -> dict[str, Any] | None:
    """Calculate indicators, freshness, signal, risk levels and quantity for one ticker."""
    if hist is None or hist.empty or len(hist) < MIN_HISTORY_ROWS:
        return None

    d = hist.copy().sort_index()
    d["SMA20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["SMA50"] = d["Close"].rolling(50, min_periods=50).mean()
    prev_close = d["Close"].shift(1)
    true_range = pd.concat(
        [d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR14"] = true_range.rolling(14, min_periods=14).mean()
    d["Turnover"] = d["Close"] * d["Volume"]  # estimate, not official investor flow
    d["TurnoverAvg20"] = d["Turnover"].shift(1).rolling(20, min_periods=20).mean()
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20, min_periods=20).mean()

    last = d.iloc[-1]
    last_date = pd.Timestamp(d.index[-1]).date()
    age_days = (today - last_date).days
    fresh = 0 <= age_days <= MAX_DATA_AGE_DAYS
    close = _as_float(last["Close"])
    atr = _as_float(last["ATR14"])
    sma20 = _as_float(last["SMA20"])
    sma50 = _as_float(last["SMA50"])
    turnover_avg = _as_float(last["TurnoverAvg20"])
    volume_avg = _as_float(last["VolAvg20"])
    turnover = _as_float(last["Turnover"])
    volume = _as_float(last["Volume"])
    previous_close = _as_float(d["Close"].iloc[-2])

    if not all(math.isfinite(x) and x > 0 for x in [close, atr, sma20, sma50, turnover, previous_close]):
        return None
    if not math.isfinite(turnover_avg) or turnover_avg <= 0 or not math.isfinite(volume_avg) or volume_avg <= 0:
        return None

    # Exclude the current row from the breakout reference to avoid look-ahead bias.
    prior_20 = d.iloc[-21:-1]
    if len(prior_20) != 20:
        return None
    prior_20_high = _as_float(prior_20["High"].max())
    if not math.isfinite(prior_20_high) or prior_20_high <= 0:
        return None

    daily_pct = (close / previous_close - 1.0) * 100.0
    turnover_ratio = turnover / turnover_avg
    volume_ratio = volume / volume_avg
    trend = close > sma20 > sma50
    breakout = close > prior_20_high
    liquid = turnover >= MIN_TURNOVER_WON
    momentum = (
        turnover_ratio >= MIN_TURNOVER_RATIO
        and volume_ratio >= MIN_VOLUME_RATIO
        and 0 <= daily_pct < MAX_DAILY_CHANGE_PCT
    )
    signal = bool(fresh and trend and breakout and liquid and momentum)

    entry = close if breakout else prior_20_high
    stop = entry - 2.0 * atr
    target = entry + 3.0 * atr
    stop_gap = entry - stop
    capital = DEFAULT_CAPITAL
    risk_budget = capital * RISK_FRACTION
    if entry <= 0 or stop <= 0 or stop_gap <= 0:
        return None
    risk_shares = math.floor(risk_budget / stop_gap)
    capital_shares = math.floor(capital / entry)
    shares = min(risk_shares, capital_shares) if signal else 0

    return {
        "code": str(code).zfill(6), "name": name, "date": last_date.isoformat(),
        "age_days": age_days, "fresh": fresh, "close": close, "sma20": sma20, "sma50": sma50,
        "atr": atr, "prior_20_high": prior_20_high, "turnover": turnover,
        "turnover_ratio": turnover_ratio, "volume_ratio": volume_ratio, "daily_pct": daily_pct,
        "trend": trend, "breakout": breakout, "liquid": liquid, "momentum": momentum,
        "signal": signal, "entry": entry, "stop": stop, "target": target,
        "risk_budget": risk_budget, "shares": max(0, shares),
        "estimated_cost": max(0, shares) * entry,
        "planned_risk": max(0, shares) * stop_gap,
    }


def scan_market(market: str, universe_size: int, progress_bar) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan the chosen universe. Returns only fresh, technically valid rows and error samples."""
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = today - timedelta(days=180)
    end_exclusive = today + timedelta(days=1)
    listing = get_listing(market).head(universe_size)
    if listing.empty:
        raise RuntimeError("선택한 시장에서 분석할 종목을 찾지 못했습니다.")

    found: list[dict[str, Any]] = []
    errors: list[str] = []
    empty_history = 0
    short_history = 0
    total = len(listing)
    for i, row in enumerate(listing.itertuples(index=False), start=1):
        code = str(getattr(row, "Symbol")).zfill(6)
        name = str(getattr(row, "Name"))
        try:
            hist = get_history(code, start.isoformat(), end_exclusive.isoformat())
            if hist.empty:
                empty_history += 1
            elif len(hist) < MIN_HISTORY_ROWS:
                short_history += 1
            item = analyze_one(code, name, hist, today) if not hist.empty else None
            if item is not None and item["fresh"]:
                found.append(item)
        except Exception as exc:
            # Keep a few concise examples for troubleshooting without flooding the UI.
            if len(errors) < 5:
                errors.append(f"{name}({code}): {type(exc).__name__}")
        if i % 5 == 0 or i == total:
            progress_bar.progress(i / total, text=f"종목 분석 중: {i}/{total}")
    found.sort(
        key=lambda x: (
            x["signal"], x["trend"], x["breakout"], x["turnover_ratio"], x["volume_ratio"], x["daily_pct"]
        ),
        reverse=True,
    )
    diagnostics = [f"종목 목록: {total}개", f"가격 데이터 없음: {empty_history}개", f"거래 이력 부족: {short_history}개"]
    return found, errors, diagnostics


with st.expander("분석 설정", expanded=True):
    market = st.selectbox("분석 시장", ["KRX", "KOSPI", "KOSDAQ"], index=0)
    universe_size = st.select_slider("분석 종목 수", options=[50, 100, 200], value=100)
    st.caption("기본 투자금 200,000원 · 거래당 계획 손실 예산 2,000원(1%) · 데이터 허용 최대 나이 5일")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []
if "scan_settings" not in st.session_state:
    st.session_state.scan_settings = None
if "scan_diagnostics" not in st.session_state:
    st.session_state.scan_diagnostics = []

if st.button("알려줘", type="primary", use_container_width=True):
    st.session_state.scan_results = None
    st.session_state.scan_errors = []
    st.session_state.scan_diagnostics = []
    st.session_state.scan_settings = (market, universe_size)
    progress = st.progress(0, text="분석을 준비하고 있습니다...")
    try:
        with st.spinner("최신 주가 데이터를 가져와 분석하고 있습니다. 종목 수에 따라 시간이 걸릴 수 있어요."):
            results, errors, diagnostics = scan_market(market, universe_size, progress)
        st.session_state.scan_results = results
        st.session_state.scan_errors = errors
        st.session_state.scan_diagnostics = diagnostics
    except Exception as exc:
        st.session_state.scan_results = []
        st.session_state.scan_errors = [f"전체 분석: {type(exc).__name__}: {exc}"]
        st.session_state.scan_diagnostics = []
    finally:
        progress.empty()

results = st.session_state.scan_results
if results is None:
    st.info("분석을 시작하려면 위의 **알려줘** 버튼을 한 번 누르세요.")
elif not results:
    st.warning("이번 분석에서 신선한 데이터와 모든 진입 조건을 충족한 종목을 찾지 못했습니다. 매수 신호가 없는 것이 정상일 수 있습니다.")
    if st.session_state.scan_diagnostics:
        st.caption("분석 진단: " + " · ".join(st.session_state.scan_diagnostics))
    if st.session_state.scan_errors:
        with st.expander("오류 확인"):
            for err in st.session_state.scan_errors:
                st.write(f"- {err}")
    st.caption("조건을 충족하는 종목이 없을 때는 억지로 종목을 추천하지 않습니다.")
else:
    # Show exactly one actionable candidate. Other rows are not presented as recommendations.
    r = next((item for item in results if item["signal"]), None)
    if r is None:
        st.warning("분석 가능한 데이터는 있었지만 모든 진입 조건을 충족한 종목은 없습니다. 대기하세요.")
    else:
        st.success("진입 조건 충족 후보가 발견되었습니다. 아래 가격은 규칙 기반 참고값이며 주문 권고가 아닙니다.")
        st.caption(f"기준 데이터: {r['date']} · 데이터 나이: {r['age_days']}일 · 종목 코드: {r['code']}")
        st.subheader(f"{r['name']} ({r['code']})")
        st.metric("최근 종가", f"{r['close']:,.0f}원", f"{r['daily_pct']:+.2f}%")
        a, b, c = st.columns(3)
        a.metric("진입 기준가", f"{r['entry']:,.0f}원")
        b.metric("손절 기준가", f"{r['stop']:,.0f}원", f"-{(r['entry']-r['stop'])/r['entry']*100:.2f}%")
        c.metric("목표 기준가", f"{r['target']:,.0f}원", f"+{(r['target']-r['entry'])/r['entry']*100:.2f}%")
        st.subheader("20만 원 기준 매수 수량 계산")
        x, y, z = st.columns(3)
        x.metric("계산 수량", f"{r['shares']}주")
        y.metric("예상 사용 금액", f"{r['estimated_cost']:,.0f}원")
        z.metric("계획 손실 예산", f"{r['risk_budget']:,.0f}원")
        st.caption(
            f"계산상 손절 폭 기준 손실 예상액: {r['planned_risk']:,.0f}원(수수료·세금·슬리피지 제외). "
            "갭 하락 등으로 실제 손실은 더 커질 수 있습니다."
        )
        st.subheader("진입 조건 점검")
        checks = [
            ("종가 > 20일 이동평균 > 50일 이동평균", r["trend"]),
            ("종가가 직전 20거래일 고점을 돌파", r["breakout"]),
            ("추정 당일 거래대금 10억 원 이상", r["liquid"]),
            (f"거래대금이 직전 20일 평균의 {MIN_TURNOVER_RATIO:.1f}배 이상", r["turnover_ratio"] >= MIN_TURNOVER_RATIO),
            (f"거래량이 직전 20일 평균의 {MIN_VOLUME_RATIO:.1f}배 이상", r["volume_ratio"] >= MIN_VOLUME_RATIO),
            (f"당일 상승률 0% 이상 {MAX_DAILY_CHANGE_PCT:.1f}% 미만", 0 <= r["daily_pct"] < MAX_DAILY_CHANGE_PCT),
            (f"데이터가 {MAX_DATA_AGE_DAYS}일 이내", r["fresh"]),
        ]
        for label, ok in checks:
            st.write(("✅ " if ok else "❌ ") + label)
        st.subheader("보조 지표")
        d1, d2, d3 = st.columns(3)
        d1.metric("거래대금 증가", f"{r['turnover_ratio']:.2f}배")
        d2.metric("거래량 증가", f"{r['volume_ratio']:.2f}배")
        d3.metric("ATR(14)", f"{r['atr']:,.0f}원")
        export = pd.DataFrame([{
            "기준일": r["date"], "종목명": r["name"], "종목코드": r["code"],
            "최근종가(원)": round(r["close"]), "진입기준가(원)": round(r["entry"]),
            "손절기준가(원)": round(r["stop"]), "목표기준가(원)": round(r["target"]),
            "계산매수수량(주)": r["shares"], "예상사용금액(원)": round(r["estimated_cost"]),
            "계획손실예산(원)": round(r["risk_budget"]), "거래대금증가배수": round(r["turnover_ratio"], 2),
            "거래량증가배수": round(r["volume_ratio"], 2), "당일등락률(%)": round(r["daily_pct"], 2),
        }])
        st.download_button(
            "분석 결과 CSV 저장", export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"스윙분석_{r['code']}_{r['date']}.csv", mime="text/csv", use_container_width=True,
        )
        if st.session_state.scan_errors:
            with st.expander("일부 종목 데이터 오류 확인"):
                st.write("일부 종목의 데이터 요청은 실패했지만, 유효한 종목 데이터는 계속 분석했습니다.")
                for err in st.session_state.scan_errors:
                    st.write(f"- {err}")

st.divider()
st.caption(
    "데이터는 FinanceDataReader를 통해 제공되는 일봉 데이터입니다. 추정 거래대금은 종가×거래량이며 "
    "공식 투자자별 순매수 자료가 아닙니다. 앱은 주문을 넣지 않습니다."
)
