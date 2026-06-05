# ===== BƯỚC 2: Import =====
import langcodes
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.tokenize import TreebankWordDetokenizer, wordpunct_tokenize
from spellchecker import SpellChecker
import nltk

nltk.download('punkt_tab', quiet=True)  # thêm dòng này
DetectorFactory.seed = 0
MIN_INPUT_LENGTH = 3

# ===== BƯỚC 3: Cấu hình =====
SPELL_LANGS = {
    "en", "es", "fr", "pt", "de",
    "ru", "ar", "eu", "lv", "nl"
}

TARGET_LANGS = {
    "Vietnamese": "vi",
    "English": "en",
    "French": "fr",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Korean": "ko",
    "Spanish": "es",
    "German": "de",
}

EXAMPLES_T = [
    "Every morning, I drink a cup of coffee.",
    "Bonjour, comment allez-vous?",
    "Xin chao, hom nay troi dep qua.",
]

EXAMPLES_S = [
    "Yesturday, I recieveed a mesage from my freind.",
    "Definately a great oppurtunity.",
    "Je voudraiis allerr au marchee.",
]

# ===== BƯỚC 4: Hàm hỗ trợ =====
@st.cache_resource(show_spinner=False)
def get_spellchecker(code):
    # Tạo và cache đối tượng SpellChecker theo mã ngôn ngữ,
    # tránh khởi tạo lại mỗi lần gọi
    return SpellChecker(language=code)

def language_name(code):
    try:
        # Chuyển mã ngôn ngữ (vd: "en") thành tên hiển thị đầy đủ (vd: "English")
        return langcodes.Language.get(code).display_name()
    except Exception:
        # Trả về mã gốc nếu không tra cứu được, hoặc "Unknown" nếu mã rỗng
        return code or "Unknown"

def detect_language(raw):
    try:
        # Dùng langdetect để nhận diện ngôn ngữ từ chuỗi văn bản
        return detect(raw)
    except LangDetectException:
        # Trả về None khi không đủ dữ liệu để nhận diện
        return None

# ===== BƯỚC 5: Hàm sửa lỗi chính tả =====
def fix_typos(text, code):
    # Lấy đối tượng spell checker cho ngôn ngữ tương ứng (có cache)
    spell = get_spellchecker(code)
    # Tách văn bản thành danh sách token (từ và dấu câu)
    tokens = wordpunct_tokenize(text)
    fixed = []

    for token in tokens:
        # Chỉ kiểm tra token là chữ cái thuần túy và có độ dài > 1
        if token.isalpha() and len(token) > 1:
            # Tra gợi ý sửa cho từ viết thường; giữ nguyên nếu không có gợi ý
            suggestion = spell.correction(token.lower()) or token
            # Khôi phục kiểu viết hoa đầu từ nếu token ban đầu là Title Case
            suggestion = suggestion.title() if token.istitle() else suggestion
            # Khôi phục kiểu ALL CAPS nếu token ban đầu viết hoa toàn bộ
            suggestion = suggestion.upper() if token.isupper() else suggestion
            fixed.append(suggestion)
        else:
            # Giữ nguyên dấu câu, số, và ký tự đặc biệt
            fixed.append(token)

    # Ghép lại token thành chuỗi; trả thêm cờ báo có sửa hay không
    return TreebankWordDetokenizer().detokenize(fixed), fixed != tokens

# ===== BƯỚC 6: Pipeline dịch =====
def run_translation(text, target_code):
    # Loại bỏ khoảng trắng thừa ở đầu và cuối chuỗi
    raw = text.strip()

    # Kiểm tra độ dài tối thiểu để tránh gửi chuỗi rỗng hoặc quá ngắn
    if len(raw) < MIN_INPUT_LENGTH:
        return {"ok": False, "error": f"Nhập tối thiểu {MIN_INPUT_LENGTH} ký tự."}

    # Nhận diện ngôn ngữ nguồn từ nội dung văn bản
    source = detect_language(raw)

    # Không tiến hành nếu không xác định được ngôn ngữ
    if source is None:
        return {"ok": False, "error": "Không nhận diện được ngôn ngữ."}

    # Nếu ngôn ngữ nguồn trùng với đích, trả về nguyên văn và thông báo
    if source == target_code:
        return {
            "ok": True,
            "source": language_name(source),
            "target": language_name(target_code),
            "translated": raw,
            "note": "Câu đã ở ngôn ngữ đích, không cần dịch.",
        }

    try:
        # Gọi Google Translate để dịch từ ngôn ngữ nguồn sang đích
        translated = GoogleTranslator(source=source, target=target_code).translate(raw)
    except Exception as e:
        return {"ok": False, "error": f"Lỗi dịch: {e}"}

    # Trả về kết quả cùng thông tin ngôn ngữ nguồn và đích
    return {
        "ok": True,
        "source": language_name(source),
        "target": language_name(target_code),
        "translated": translated,
    }

# ===== BƯỚC 7: Pipeline chính tả =====
def run_spellcheck(text):
    # Loại bỏ khoảng trắng thừa ở đầu và cuối chuỗi
    raw = text.strip()

    # Kiểm tra độ dài tối thiểu trước khi xử lý
    if len(raw) < MIN_INPUT_LENGTH:
        return {"ok": False, "error": f"Nhập tối thiểu {MIN_INPUT_LENGTH} ký tự."}

    # Nhận diện ngôn ngữ để chọn đúng từ điển chính tả
    code = detect_language(raw)

    # Không tiến hành nếu không xác định được ngôn ngữ
    if code is None:
        return {"ok": False, "error": "Không nhận diện được ngôn ngữ."}

    # Kiểm tra ngôn ngữ có nằm trong danh sách hỗ trợ của pyspellchecker
    if code not in SPELL_LANGS:
        return {
            "ok": False,
            "error": f"pyspellchecker chưa hỗ trợ {language_name(code)} ({code}).",
        }

    # Thực hiện sửa lỗi chính tả; nhận lại chuỗi đã sửa và cờ có thay đổi
    fixed, changed = fix_typos(raw, code)

    # Trả về kết quả cùng tên ngôn ngữ và trạng thái có/không có sửa đổi
    return {
        "ok": True,
        "language": language_name(code),
        "fixed": fixed,
        "changed": changed,
    }

# ===== BƯỚC 8: Giao diện chính =====
# Cấu hình tiêu đề trang và layout hiển thị giữa màn hình
st.set_page_config(page_title="NLP Pipeline Demo", layout="centered")
# Hiển thị tiêu đề và mô tả ngắn cho toàn bộ ứng dụng
st.title("Streamlit NLP Pipeline Demo")
st.caption("Hai ứng dụng: Dịch văn bản - Sửa lỗi chính tả")

# Tạo hai tab chính, mỗi tab tương ứng một tính năng NLP
tab_t, tab_s = st.tabs(["Dịch văn bản", "Sửa lỗi chính tả"])

# ===== BƯỚC 9: Tab dịch văn bản =====
with tab_t:
    # Khởi tạo key lưu kết quả dịch trong session nếu chưa tồn tại
    st.session_state.setdefault("res_t", None)

    # Hiển thị các câu ví dụ để người dùng tham khảo
    with st.expander("Ví dụ"):
        for ex in EXAMPLES_T:
            st.markdown(f"- {ex}")

    # Form nhập liệu; Streamlit chỉ xử lý khi người dùng bấm submit
    with st.form("form_translate"):
        text_t = st.text_area(
            "Câu cần dịch",
            height=90,
            placeholder="Nhập câu ở bất kỳ ngôn ngữ nào..."
        )
        # Dropdown chọn ngôn ngữ đích từ danh sách TARGET_LANGS
        target = st.selectbox("Dịch sang", list(TARGET_LANGS.keys()))
        submitted_t = st.form_submit_button("Dịch", type="primary")

    # Chỉ chạy pipeline khi form được submit
    if submitted_t:
        st.session_state.res_t = run_translation(text_t, TARGET_LANGS[target])

    # Đọc kết quả từ session để hiển thị (kể cả sau khi rerun)
    res = st.session_state.res_t

    if res:
        if res["ok"]:
            # Hiển thị cặp ngôn ngữ nguồn - đích và bản dịch
            st.caption(f"Nguồn: {res['source']} -> Đích: {res['target']}")
            st.success(res["translated"])
            # Hiển thị ghi chú bổ sung nếu có (vd: đã ở ngôn ngữ đích)
            if res.get("note"):
                st.info(res["note"])
        else:
            # Hiển thị thông báo lỗi nếu pipeline thất bại
            st.warning(res["error"])

# ===== BƯỚC 10: Tab sửa lỗi chính tả =====
with tab_s:
    # Khởi tạo key lưu kết quả kiểm tra chính tả trong session nếu chưa tồn tại
    st.session_state.setdefault("res_s", None)

    # Hiển thị các câu ví dụ có lỗi chính tả để người dùng thử nghiệm
    with st.expander("Ví dụ"):
        for ex in EXAMPLES_S:
            st.markdown(f"- {ex}")

    # Thông báo rõ danh sách ngôn ngữ được hỗ trợ bởi pyspellchecker
    st.caption(f"Hỗ trợ: {', '.join(sorted(SPELL_LANGS))}")

    # Form nhập liệu; Streamlit chỉ xử lý khi người dùng bấm submit
    with st.form("form_spell"):
        text_s = st.text_area(
            "Câu cần kiểm tra",
            height=90,
            placeholder="Nhập câu để kiểm tra chính tả..."
        )
        submitted_s = st.form_submit_button("Kiểm tra", type="primary")

    # Chỉ chạy pipeline khi form được submit
    if submitted_s:
        st.session_state.res_s = run_spellcheck(text_s)

    # Đọc kết quả từ session để hiển thị (kể cả sau khi rerun)
    res = st.session_state.res_s

    if res:
        if res["ok"]:
            # Hiển thị ngôn ngữ nhận diện được và văn bản đã sửa
            st.caption(f"Ngôn ngữ: {res['language']}")
            st.success(res["fixed"])
            # Thông báo có/không có lỗi được sửa dựa trên cờ changed
            st.caption("Có sửa lỗi chính tả" if res["changed"] else "Không phát hiện lỗi")
        else:
            # Hiển thị thông báo lỗi nếu pipeline thất bại
            st.warning(res["error"])