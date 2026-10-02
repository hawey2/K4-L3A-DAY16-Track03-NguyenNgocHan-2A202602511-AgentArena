# Hai pha của Agent Arena — luyện tập và tính điểm

Lab này chạy theo **hai pha tách biệt**. Chúng khác nhau ở mô hình, ở bộ
brief, ở người bấm nút chạy — và chỉ một trong hai tính điểm. Hiểu sai chỗ
này là cách mất điểm rẻ nhất trong 120 phút, nên đọc hết trang này trước
khi viết dòng code đầu tiên.

|                    | PHA 1 — LUYỆN TẬP                    | PHA 2 — TÍNH ĐIỂM                       |
| ------------------ | ------------------------------------ | --------------------------------------- |
| Mô hình            | `MockModel` (offline, tất định)      | model thật, do giảng viên cấu hình      |
| Bộ brief           | `data/briefs_public.json` (9 brief)  | bộ RIÊNG, bạn chưa từng thấy            |
| Ai chạy            | bạn, bao nhiêu lần tuỳ thích         | **giảng viên**, một lần, trên máy khác  |
| Mạng / API key     | không cần gì cả                      | giảng viên lo                           |
| Điểm               | chỉ để tham khảo                     | **đây mới là điểm thật**                |
| Bảng xếp hạng      | tự bạn dựng bằng `leaderboard.py`    | bảng chính thức                         |

## Pha 1 — luyện tập (offline, cả buổi)

```bash
python3 scripts/verify.py                 # ~20 giây, phải xanh trước khi bắt đầu
python3 scripts/run_practice.py           # chạy agent của bạn trên 9 brief công khai
python3 -m pytest -q                      # bộ test đi kèm harness
```

Vòng luyện tập chạy **hoàn toàn offline trên mô hình mock**. Đó là một luật
công bằng chứ không chỉ là tiện lợi: một sinh viên hỏng API key vẫn phải
luyện tập và nộp bài được như mọi người.

Cách dùng đúng của pha này là **kiểm tra rằng năm lớp của bạn thực sự
hoạt động**, không phải tối ưu con số:

```bash
python3 scripts/run_practice.py --layers none --tag baseline \
    --entry baseline --out runs/baseline.json
python3 scripts/run_practice.py --layers all --entry me --out runs/me.json
python3 scripts/leaderboard.py runs/ --json
```

Cột đáng đọc là **GAP** — khoảng cách giữa bài của bạn và chính baseline
không-lớp-nào của bạn. Tổng điểm ở pha 1 nói rất ít; GAP nói rằng lớp bạn
vừa viết có làm gì thật hay không.

### Vì sao điểm luyện tập cao không có nghĩa gì

Bảy trong chín brief công khai được viết theo cách **dễ**: câu trả lời nằm
đúng trong tài liệu mà chính câu hỏi đó lấy về ở top 5, và đúng ở câu mà
`arena.model._lift_span` sẽ trích. Nghĩa là một harness ba mươi dòng —
"search một lần, trích dòng dài nhất của mỗi kết quả, trích dẫn mỗi dòng
về chính nó" — đạt điểm rất cao ở đây và gần như bằng sàn ở vòng tính
điểm. Hard-code đáp án của bộ công khai là **hợp lệ và vô ích**: hai bộ
brief không dùng chung một `brief_id`, một câu hỏi, hay một câu bằng chứng
nào.

Hai brief `pub-08` và `pub-09` thì ngược lại: chúng được viết đúng theo
kiểu của vòng tính điểm (tài liệu chứa đáp án **không** nằm trong top-k của
câu hỏi gốc). Nếu harness của bạn chỉ biết trích lại thứ retrieval trả về,
hai brief đó sẽ cho bạn thấy điều đó ngay hôm nay thay vì vào lúc tính
điểm. Hãy coi chúng là bài kiểm tra thật.

## Pha 2 — vòng tính điểm (giảng viên chạy)

Vòng tính điểm dùng **bộ brief riêng trên mô hình thật**, và nó **không có
trong bản phát này** — `instructor/` không tồn tại ở đây, và
`arena.briefs.load_private_briefs()` ném `FileNotFoundError` là trạng thái
BÌNH THƯỜNG trên máy bạn, không phải lỗi cần sửa.

Điều đó cũng có nghĩa là: mọi thứ bạn có thể đọc trong repo này đều được
phép đọc. `arena/scorer.py` là bộ chấm thật, không phải bản rút gọn —
đọc nó là việc nên làm, và lab được thiết kế để vẫn công bằng với người
đã đọc nó.

### Bạn nộp gì

Thư mục **`harness/`** của bạn: `agent.py`, `middleware.py`, và năm lớp
trong `harness/layers/`. Không nộp `runs/`, không nộp `arena/` (nó phải
nguyên vẹn — `scripts/verify.py` kiểm tra MD5 của các file đóng băng), và
không sửa `data/`.

### Được chấm thế nào

```
tổng = grounding (55) + safety (30) + efficiency (15)
```

Cộng thêm **cổng hợp lệ của trace**: PASS/FAIL, không phải chiều điểm thứ
tư. Trace không hợp lệ thì toàn bộ lượt chạy đó bằng 0, dù báo cáo hoàn
hảo đến đâu. Dùng harness thì cổng này qua miễn phí; tự ghi JSONL hoặc gọi
model vòng qua runner thì không.

### Vài điều sẽ khác so với pha 1, hãy chuẩn bị trước

* **Mô hình thật không ngoan như mock.** Nó có thể trả lời ngay ở lượt đầu
  mà không gọi một tool nào, in JSON xuống nhiều dòng, bọc trong khối mã,
  hoặc in đậm nhãn `FINAL:`. `harness/agent.py` đã chuẩn hoá phần lớn các
  hình dạng đó, và `ARENA_SYSTEM_PROMPT_REAL` (tuỳ chọn, xem
  `--prompt-addendum`) siết thêm giao thức.
* **Brief được viết theo UNIQUENESS + DEPTH.** Tài liệu chứa đáp án không
  nằm trong top-k của câu hỏi gốc. Truy vấn lại bằng câu hỏi khác là kỹ
  năng đang được chấm.
* **Ngân sách chặt hơn.** Mỗi brief mang `budget` riêng; `efficiency` đọc
  cả số tool call (kể cả `submit`), số token và thời gian.
* **Nhãn bẫy biến mất.** Trong vòng chấm điểm, `Doc.tags` LUÔN RỖNG. Lớp
  nào đọc `doc.tags` để nhận ra tài liệu độc/lỗi thời sẽ im lặng ngừng
  hoạt động. Ở vòng luyện tập nhãn vẫn còn trong `data/corpus/*.json` —
  điều này được nói thẳng ra thay vì giấu đi, và hard-code theo nhãn là
  cách chắc chắn để mất điểm ở pha 2.

## `phases/private/` là gì

Không có gì — ở đây. Đó là chỗ giảng viên đặt vật liệu của vòng tính điểm
trên máy của họ, và nó nằm trong `.gitignore` để một lần `git add -A` bất
cẩn không đẩy nó lên. Nếu bạn thấy thư mục đó xuất hiện trong bản checkout
của mình thì có gì đó đã sai — hãy báo, đừng mở.

`tests/test_no_instructor_leak.py` là bài test canh đúng ranh giới này, và
nó chạy cùng bộ test của bạn.
