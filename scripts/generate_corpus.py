"""Synthetic document corpus generator for ECM AI Module.

Generates 5 document types × N samples as:
  corpus/raw/      — clean vector PDFs
  corpus/scanned/  — simulated scanned versions (rotation, noise, JPEG)
  corpus/ground_truth.json — ground-truth attributes

Usage:
  uv run python scripts/generate_corpus.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from faker import Faker
from fpdf import FPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Font paths — Times New Roman ships with macOS and covers Cyrillic
# ---------------------------------------------------------------------------
_FONT_DIR = Path("/System/Library/Fonts/Supplemental")
_FONT_REG = str(_FONT_DIR / "Times New Roman.ttf")
_FONT_BOLD = str(_FONT_DIR / "Times New Roman Bold.ttf")


# ===========================================================================
# STEP 0 — Calibrate scanner parameters from real documents
# ===========================================================================

def _ssim_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified global SSIM over two grayscale float32 arrays (no scikit-image)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2, sigma_b2 = a.var(), b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(num / den) if den != 0 else 1.0


def calibrate_scanner_params(real_docs_dir: Path) -> dict:
    """Analyze up to 10 real scanned PDFs to extract scanner quality parameters.

    Returns:
        dict with keys: noise_sigma, jpeg_quality_min, jpeg_quality_max
    """
    fallback = {"noise_sigma": 2.0, "jpeg_quality_min": 80, "jpeg_quality_max": 92}

    if not real_docs_dir.exists():
        print("  [calibrate] Real docs directory not found — using fallback params")
        return fallback

    pdf_paths: list[Path] = []
    for subdir in ["платежное поручение", "УПД"]:
        sd = real_docs_dir / subdir
        if sd.exists():
            pdf_paths.extend(sorted(sd.glob("*.pdf"))[:5])
    pdf_paths = pdf_paths[:10]

    if not pdf_paths:
        print("  [calibrate] No PDF files found — using fallback params")
        return fallback

    noise_sigmas: list[float] = []
    quality_mins: list[int] = []
    quality_maxes: list[int] = []

    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[0]

            # Detect scan: sparse text (<10 words) AND has embedded images
            words = page.get_text("words")
            images = page.get_images()
            if len(words) >= 10 or not images:
                doc.close()
                continue

            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()

            # noise_sigma: std of pixel values in a blank 50 px top margin strip
            top_strip = np.array(img.convert("L"), dtype=np.float32)[:50, :]
            noise_sigmas.append(float(top_strip.std()))

            # JPEG quality estimate: compress → decompress → measure SSIM
            orig_gray = np.array(img.convert("L"), dtype=np.float32)
            found_q: int | None = None
            for q in range(65, 96, 5):
                buf = BytesIO()
                img.save(buf, "JPEG", quality=q)
                buf.seek(0)
                arr2 = np.array(Image.open(buf).convert("L"), dtype=np.float32)
                if _ssim_numpy(orig_gray, arr2) >= 0.97:
                    found_q = q
                    break

            if found_q is not None:
                quality_mins.append(max(65, found_q - 5))
                quality_maxes.append(min(95, found_q + 5))

        except Exception as exc:
            print(f"  [calibrate] Skipping {pdf_path.name}: {exc}")
            continue

    if not noise_sigmas:
        print("  [calibrate] No usable scanned pages found — using fallback params")
        return fallback

    return {
        "noise_sigma": round(min(float(np.mean(noise_sigmas)), 2.5), 2),
        "jpeg_quality_min": max(int(np.mean(quality_mins)) if quality_mins else fallback["jpeg_quality_min"], 78),
        "jpeg_quality_max": max(int(np.mean(quality_maxes)) if quality_maxes else fallback["jpeg_quality_max"], 88),
    }


# ===========================================================================
# STEP 1 — PDF template helpers
# ===========================================================================

def _new_pdf() -> FPDF:
    """Return an A4 FPDF with Times New Roman (regular + bold) loaded."""
    pdf = FPDF()
    pdf.add_font("TNR", fname=_FONT_REG)
    pdf.add_font("TNR", style="B", fname=_FONT_BOLD)
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    return pdf


# ---------------------------------------------------------------------------
# INVOICE — счёт-фактура
# ---------------------------------------------------------------------------

def generate_invoice(type_index: int, doc_index: int) -> tuple[bytes, dict]:
    fake = Faker("ru_RU")
    fake.seed_instance(type_index * 100 + doc_index)
    rnd = random.Random(type_index * 100 + doc_index)

    num = str(rnd.randint(1, 999))
    d_str = fake.date_between(start_date="-3y", end_date="today").strftime("%d.%m.%Y")
    d_iso = datetime.strptime(d_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    seller_name = fake.company()
    seller_inn = fake.numerify("77########")
    buyer_name = fake.company()
    buyer_inn = fake.numerify("77########")
    amount_with_vat = round(rnd.uniform(15_000, 2_000_000), 2)
    vat_amount = round(amount_with_vat / 6, 2)
    amount_no_vat = round(amount_with_vat - vat_amount, 2)

    goods = [
        (fake.bs().split()[0].capitalize()[:38], rnd.randint(1, 100), round(rnd.uniform(500, 50_000), 2))
        for _ in range(rnd.randint(2, 3))
    ]

    pdf = _new_pdf()
    pdf.set_font("TNR", "B", 13)
    pdf.cell(0, 9, f"СЧЁТ-ФАКТУРА № {num} от {d_str}", align="C")
    pdf.ln(10)

    pdf.set_font("TNR", size=11)
    pdf.cell(90, 7, f"Продавец: {seller_name}")
    pdf.cell(0, 7, f"ИНН: {seller_inn}")
    pdf.ln()
    pdf.cell(90, 7, f"Покупатель: {buyer_name}")
    pdf.cell(0, 7, f"ИНН: {buyer_inn}")
    pdf.ln(8)

    col_w = [80, 25, 35, 40]
    pdf.set_font("TNR", "B", 10)
    for label, w in zip(["Наименование", "Кол-во", "Цена, руб.", "Сумма, руб."], col_w):
        pdf.cell(w, 7, label, border=1, align="C")
    pdf.ln()

    pdf.set_font("TNR", size=10)
    for name, qty, price in goods:
        row_sum = round(qty * price, 2)
        for val, w in zip([name, str(qty), f"{price:,.2f}", f"{row_sum:,.2f}"], col_w):
            pdf.cell(w, 7, val, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("TNR", size=11)
    pdf.cell(
        0, 7,
        f"Итого без НДС: {amount_no_vat:,.2f} руб.   "
        f"НДС 20%: {vat_amount:,.2f} руб.   "
        f"Итого: {amount_with_vat:,.2f} руб.",
    )

    attrs: dict = {
        "invoice_number": num,
        "invoice_date": d_iso,
        "seller_name": seller_name,
        "seller_inn": seller_inn,
        "buyer_name": buyer_name,
        "buyer_inn": buyer_inn,
        "amount_with_vat": amount_with_vat,
        "vat_amount": vat_amount,
    }
    return bytes(pdf.output()), attrs


# ---------------------------------------------------------------------------
# ACT — акт выполненных работ
# ---------------------------------------------------------------------------

def generate_act(type_index: int, doc_index: int) -> tuple[bytes, dict]:
    fake = Faker("ru_RU")
    fake.seed_instance(type_index * 100 + doc_index)
    rnd = random.Random(type_index * 100 + doc_index)

    num = str(rnd.randint(1, 999))
    d_str = fake.date_between(start_date="-3y", end_date="today").strftime("%d.%m.%Y")
    d_iso = datetime.strptime(d_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    contractor_name = fake.company()
    contractor_inn = fake.numerify("77########")
    client_name = fake.company()
    client_inn = fake.numerify("77########")
    total_amount = round(rnd.uniform(15_000, 2_000_000), 2)
    services_description = (
        f"{rnd.choice(['Разработка', 'Техническое обслуживание', 'Монтаж', 'Пусконаладка'])} "
        f"{rnd.choice(['программного обеспечения', 'оборудования', 'системы вентиляции', 'сети передачи данных'])}"
    )

    pdf = _new_pdf()
    pdf.set_font("TNR", "B", 12)
    pdf.multi_cell(0, 8, f"АКТ СДАЧИ-ПРИЁМКИ ВЫПОЛНЕННЫХ РАБОТ № {num} от {d_str}", align="C", new_x="LMARGIN")
    pdf.ln(6)

    pdf.set_font("TNR", size=11)
    pdf.multi_cell(0, 7, f"Исполнитель: {contractor_name}, ИНН {contractor_inn}", new_x="LMARGIN")
    pdf.multi_cell(0, 7, f"Заказчик: {client_name}, ИНН {client_inn}", new_x="LMARGIN")
    pdf.ln(4)
    pdf.multi_cell(
        0, 7,
        f"Исполнитель выполнил, а Заказчик принял следующие работы (услуги): "
        f"{services_description}.",
        new_x="LMARGIN",
    )
    pdf.ln(4)
    pdf.set_font("TNR", "B", 11)
    pdf.cell(0, 7, f"Общая стоимость работ: {total_amount:,.2f} руб.")
    pdf.ln(14)

    pdf.set_font("TNR", size=11)
    pdf.cell(90, 7, "Исполнитель: ______________")
    pdf.cell(0, 7, "Заказчик: ______________")

    attrs: dict = {
        "act_number": num,
        "act_date": d_iso,
        "contractor_name": contractor_name,
        "contractor_inn": contractor_inn,
        "client_name": client_name,
        "client_inn": client_inn,
        "services_description": services_description,
        "total_amount": total_amount,
    }
    return bytes(pdf.output()), attrs


# ---------------------------------------------------------------------------
# ORDER — приказ
# ---------------------------------------------------------------------------

def generate_order(type_index: int, doc_index: int) -> tuple[bytes, dict]:
    fake = Faker("ru_RU")
    fake.seed_instance(type_index * 100 + doc_index)
    rnd = random.Random(type_index * 100 + doc_index)

    num = str(rnd.randint(1, 999))
    d_str = fake.date_between(start_date="-3y", end_date="today").strftime("%d.%m.%Y")
    d_iso = datetime.strptime(d_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    org_name = fake.company() + " (ООО)"
    order_subject = rnd.choice([
        "проведении инвентаризации",
        "назначении ответственных лиц",
        "утверждении графика отпусков",
        "организации документооборота",
        "внесении изменений в штатное расписание",
    ])
    signatory_position = rnd.choice([
        "Генеральный директор", "Директор",
        "Исполнительный директор", "Руководитель",
    ])
    signatory_name = (
        f"{fake.last_name()} {fake.first_name()[0]}.{fake.middle_name()[0]}."
    )
    clauses = [f"{i + 1}. {fake.bs().capitalize()}." for i in range(3)]

    pdf = _new_pdf()
    pdf.set_font("TNR", size=11)
    pdf.cell(0, 7, org_name, align="C")
    pdf.ln()
    pdf.set_font("TNR", "B", 13)
    pdf.cell(0, 9, f"ПРИКАЗ № {num}", align="C")
    pdf.ln()
    pdf.set_font("TNR", size=11)
    pdf.cell(0, 7, f"от {d_str}", align="C")
    pdf.ln(8)
    pdf.set_font("TNR", "B", 12)
    pdf.multi_cell(0, 7, f"О {order_subject}", new_x="LMARGIN")
    pdf.ln(4)
    pdf.set_font("TNR", "B", 11)
    pdf.cell(0, 7, "ПРИКАЗЫВАЮ:")
    pdf.ln(4)
    pdf.set_font("TNR", size=11)
    for clause in clauses:
        pdf.multi_cell(0, 7, clause, new_x="LMARGIN")
        pdf.ln(2)
    pdf.ln(10)
    pdf.cell(0, 7, f"{signatory_position}    ________    {signatory_name}")

    attrs: dict = {
        "order_number": num,
        "order_date": d_iso,
        "organization_name": org_name,
        "order_subject": order_subject,
        "signatory_name": signatory_name,
        "signatory_position": signatory_position,
    }
    return bytes(pdf.output()), attrs


# ---------------------------------------------------------------------------
# WAYBILL — требование-накладная, форма М-11
# ---------------------------------------------------------------------------

def generate_waybill(type_index: int, doc_index: int) -> tuple[bytes, dict]:
    fake = Faker("ru_RU")
    fake.seed_instance(type_index * 100 + doc_index)
    rnd = random.Random(type_index * 100 + doc_index)

    num = str(rnd.randint(1, 999))
    d_str = fake.date_between(start_date="-3y", end_date="today").strftime("%d.%m.%Y")
    d_iso = datetime.strptime(d_str, "%d.%m.%Y").strftime("%Y-%m-%d")

    def _dept() -> str:
        choice = rnd.randint(0, 3)
        if choice == 0:
            return f"Склад № {rnd.randint(1, 9)}"
        if choice == 1:
            return f"Цех № {rnd.randint(1, 5)}"
        if choice == 2:
            return "Отдел снабжения"
        return f"Производственный участок № {rnd.randint(1, 4)}"

    sender_dept = _dept()
    receiver_dept = _dept()
    op_code = fake.numerify("#######")

    units = ["шт", "м", "кг", "л", "уп"]
    materials = [
        (fake.bs().split()[0].capitalize()[:30], rnd.choice(units), rnd.randint(1, 500))
        for _ in range(rnd.randint(3, 5))
    ]
    items_description = ", ".join(m[0] for m in materials)

    pdf = _new_pdf()
    pdf.set_font("TNR", "B", 12)
    pdf.cell(0, 8, f"ТРЕБОВАНИЕ-НАКЛАДНАЯ № {num}  Форма № М-11")
    pdf.ln(8)

    pdf.set_font("TNR", size=11)
    pdf.cell(90, 7, f"Дата: {d_str}")
    pdf.cell(0, 7, f"Код операции: {op_code}")
    pdf.ln()
    pdf.cell(90, 7, f"Отправитель: {sender_dept}")
    pdf.cell(0, 7, f"Получатель: {receiver_dept}")
    pdf.ln(8)

    col_w = [100, 25, 45]
    pdf.set_font("TNR", "B", 10)
    for label, w in zip(["Наименование", "Ед.изм.", "Количество"], col_w):
        pdf.cell(w, 7, label, border=1, align="C")
    pdf.ln()

    pdf.set_font("TNR", size=10)
    for name, unit, qty in materials:
        for val, w in zip([name, unit, str(qty)], col_w):
            pdf.cell(w, 7, val, border=1)
        pdf.ln()

    attrs: dict = {
        "document_number": num,
        "document_date": d_iso,
        "sender_department": sender_dept,
        "receiver_department": receiver_dept,
        "operation_type_code": op_code,
        "items_description": items_description,
    }
    return bytes(pdf.output()), attrs


# ---------------------------------------------------------------------------
# PAYMENT — платёжное поручение
# ---------------------------------------------------------------------------

def generate_payment(type_index: int, doc_index: int) -> tuple[bytes, dict]:
    fake = Faker("ru_RU")
    fake.seed_instance(type_index * 100 + doc_index)
    rnd = random.Random(type_index * 100 + doc_index)

    num = str(rnd.randint(1, 999))
    d_str = fake.date_between(start_date="-3y", end_date="today").strftime("%d.%m.%Y")
    d_iso = datetime.strptime(d_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    payer_name = fake.company()
    payer_inn = fake.numerify("77########")
    receiver_name = fake.company()
    receiver_inn = fake.numerify("77########")
    amount = round(rnd.uniform(15_000, 2_000_000), 2)
    purpose_date = fake.date_between(start_date="-2y", end_date="today").strftime("%d.%m.%Y")
    purpose_goods = rnd.choice(["услуги", "поставку товаров", "выполненные работы"])
    payment_purpose = (
        f"Оплата по договору № {rnd.randint(1, 200)} "
        f"от {purpose_date} за {purpose_goods} без НДС"
    )

    pdf = _new_pdf()
    pdf.set_font("TNR", "B", 13)
    pdf.cell(0, 9, f"ПЛАТЁЖНОЕ ПОРУЧЕНИЕ № {num} от {d_str}")
    pdf.ln(10)

    w_lbl, w_val, w_key, w_inn = 40, 85, 22, 33
    pdf.set_font("TNR", size=11)
    pdf.cell(w_lbl, 8, "Плательщик:", border=1)
    pdf.cell(w_val, 8, payer_name[:45], border=1)
    pdf.cell(w_key, 8, "ИНН:", border=1, align="R")
    pdf.cell(w_inn, 8, payer_inn, border=1)
    pdf.ln()
    pdf.cell(w_lbl, 8, "Получатель:", border=1)
    pdf.cell(w_val, 8, receiver_name[:45], border=1)
    pdf.cell(w_key, 8, "ИНН:", border=1, align="R")
    pdf.cell(w_inn, 8, receiver_inn, border=1)
    pdf.ln()
    pdf.cell(w_lbl + w_val + w_key + w_inn, 8, f"Сумма: {amount:,.2f} руб.", border=1)
    pdf.ln(8)
    pdf.multi_cell(0, 7, f"Назначение платежа: {payment_purpose}", new_x="LMARGIN")

    attrs: dict = {
        "document_number": num,
        "document_date": d_iso,
        "payer_name": payer_name,
        "payer_inn": payer_inn,
        "receiver_name": receiver_name,
        "receiver_inn": receiver_inn,
        "amount": amount,
        "payment_purpose": payment_purpose,
    }
    return bytes(pdf.output()), attrs


# ===========================================================================
# STEP 2 — Scanner effect
# ===========================================================================

def apply_scanner_effect(
    pdf_bytes: bytes,
    params: dict,
    rng: np.random.Generator,
) -> bytes:
    """Render one PDF page and apply simulated scanner degradation."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    # 1. Slight random rotation (±1.5°)
    rotation = float(rng.uniform(-1.5, 1.5))
    img = img.rotate(rotation, expand=False, fillcolor=(255, 255, 255))

    # 2. Gaussian noise
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, params["noise_sigma"], arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 3. JPEG compression at random quality within calibrated range
    quality = int(rng.integers(params["jpeg_quality_min"], params["jpeg_quality_max"] + 1))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).copy()

    # 4. Wrap back into a single-page PDF at 150 DPI
    out = BytesIO()
    img.save(out, "PDF", resolution=150)
    return out.getvalue()


# ===========================================================================
# STEP 3 — Main corpus generator
# ===========================================================================

def generate_corpus(output_dir: str = "corpus", n_per_type: int = 20) -> None:
    output = Path(output_dir)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "scanned").mkdir(parents=True, exist_ok=True)

    GENERATORS = [
        ("INVOICE", generate_invoice),
        ("ACT", generate_act),
        ("ORDER", generate_order),
        ("WAYBILL", generate_waybill),
        ("PAYMENT", generate_payment),
    ]

    real_docs_dir = Path(
        "/Users/rustamakhmedzianov/Downloads/мага/вкр/ВКР/образцы документов"
    )
    scan_params = calibrate_scanner_params(real_docs_dir)
    print(f"Scanner params: {scan_params}")

    ground_truth: list[dict] = []

    for type_index, (doc_type, gen_func) in enumerate(GENERATORS):
        for i in range(n_per_type):
            doc_num = i + 1
            raw_name = f"{doc_type}_{doc_num:03d}_raw.pdf"
            scan_name = f"{doc_type}_{doc_num:03d}.pdf"

            pdf_bytes, attrs = gen_func(type_index=type_index, doc_index=i)
            (output / "raw" / raw_name).write_bytes(pdf_bytes)

            rng = np.random.default_rng(seed=type_index * 1000 + i)
            scanned = apply_scanner_effect(pdf_bytes, scan_params, rng)
            (output / "scanned" / scan_name).write_bytes(scanned)

            ground_truth.append({
                "filename": scan_name,
                "document_type": doc_type,
                "attributes": attrs,
            })
            print(f"  {scan_name} ✓")

    gt_path = output / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2))

    # Validation
    assert len(ground_truth) == len(GENERATORS) * n_per_type
    for entry in ground_truth:
        assert all(v is not None for v in entry["attributes"].values()), \
            f"None value in {entry['filename']}"

    print(f"\n✅ Generated {len(ground_truth)} documents")
    print(f"   Scanner params used: {scan_params}")
    print(f"   Ground truth: {gt_path} ({len(ground_truth)} entries)")


if __name__ == "__main__":
    generate_corpus(output_dir="corpus", n_per_type=20)
