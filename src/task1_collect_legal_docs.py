"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.

Nguồn thực tế dùng trong script này (RMIT Vietnam, trang công khai):
    - Student fees & charges guide (học phí & các khoản phí)
    - Scholarship terms & conditions (điều khoản học bổng)
    - HCM accommodation advice list (hỗ trợ chỗ ở TP.HCM)
    - Hanoi accommodation advice support list (hỗ trợ chỗ ở Hà Nội)
    - Wellbeing external service provider directory (danh sách dịch vụ hỗ trợ sinh viên)

Tất cả là PDF gốc được trường xuất bản công khai, không phải HTML convert.
"""

from pathlib import Path
import urllib.request
import urllib.error

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

LEGAL_DOCS = [
    {
        "url": "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/"
               "documents/pdfs/study-at-rmit/tuition-fees/"
               "student-fees-and-charges-guide-06-2026.pdf",
        "filename": "tuition-fees-and-charges-rmit-2026.pdf",
        "topic": "Học phí & các khoản phí (Tuition Fees & Charges)",
    },
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
               "assets-for-production/documents/pdfs/study-at-rmit/scholarships/"
               "english-pdf/rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
        "filename": "scholarship-terms-and-conditions-rmit.pdf",
        "topic": "Điều khoản & điều kiện học bổng (Scholarship Terms & Conditions)",
    },
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
               "assets-for-production/documents/pdfs/students/accommodation/"
               "hcm-accommodation-advice-list.pdf",
        "filename": "accommodation-hcm-advice-rmit.pdf",
        "topic": "Hỗ trợ chỗ ở TP.HCM (Accommodation Services — HCM)",
    },
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
               "assets-for-production/documents/pdfs/students/accommodation/"
               "hanoi-accommodation-advice-support-list.pdf",
        "filename": "accommodation-hanoi-advice-rmit.pdf",
        "topic": "Hỗ trợ chỗ ở Hà Nội (Accommodation Services — Hanoi)",
    },
    {
        "url": "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/"
               "documents/pdfs/students/wellbeing/"
               "external-service-provider-directory-25.pdf",
        "filename": "wellbeing-external-service-provider-directory-rmit.pdf",
        "topic": "Danh sách dịch vụ hỗ trợ sinh viên (Wellbeing Services Directory)",
    },
]

MIN_FILE_BYTES = 1024


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")
    return DATA_DIR


def download_file(url: str, filename: str, dest_dir: Path = DATA_DIR) -> Path | None:
    """Tải một file từ URL về dest_dir/filename. Trả về path nếu thành công."""
    filepath = dest_dir / filename
    if filepath.exists() and filepath.stat().st_size > MIN_FILE_BYTES:
        print(f"✓ Đã tồn tại ({filepath.stat().st_size} bytes): {filename}")
        return filepath

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
            if len(data) <= MIN_FILE_BYTES:
                print(f"✗ File quá nhỏ ({len(data)} bytes), có thể bị lỗi: {filename}")
                return None
            filepath.write_bytes(data)
            print(f"✓ Đã tải ({len(data)} bytes): {filename}")
            return filepath
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP {e.code} khi tải {filename}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"✗ Lỗi mạng khi tải {filename}: {e.reason}")
    except Exception as e:
        print(f"✗ Lỗi không xác định khi tải {filename}: {e}")
    return None


def collect_all() -> list[Path]:
    """Tải tất cả văn bản trong LEGAL_DOCS về DATA_DIR. Trả về danh sách file thành công."""
    setup_directory()
    downloaded = []
    for doc in LEGAL_DOCS:
        print(f"\n→ {doc['topic']}")
        result = download_file(doc["url"], doc["filename"])
        if result:
            downloaded.append(result)
    return downloaded


def main():
    print("=" * 70)
    print("Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học")
    print("Nguồn: RMIT Vietnam (rmit.edu.vn) — công khai")
    print("=" * 70)

    files = collect_all()

    print("\n" + "=" * 70)
    print(f"Tổng kết: tải thành công {len(files)}/{len(LEGAL_DOCS)} văn bản")
    print("=" * 70)
    for f in files:
        print(f"  • {f.name}  ({f.stat().st_size} bytes)")

    if len(files) < 3:
        print("\n⚠ Cảnh báo: cần tối thiểu 3 file cho test Task 1.")
        return 1
    print("\n✓ Đã đủ ≥3 văn bản cho data/landing/legal/ — Task 1 hoàn thành.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())