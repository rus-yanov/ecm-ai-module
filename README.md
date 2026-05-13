<div align="center">

# ECM AI Module

### Интеллектуальная обработка входящих документов в ECM-системе

**OCR · Классификация документов · Извлечение реквизитов · Confidence Routing**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Qwen_2.5_7B-Ollama-FF6B35)](https://ollama.com)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-кириллица-0062B1)](https://github.com/PaddlePaddle/PaddleOCR)
[![Лицензия](https://img.shields.io/badge/Лицензия-MIT-green)](LICENSE)

---

*Магистерская выпускная квалификационная работа*  
**НИУ ВШЭ · Высшая Школа Бизнеса · 2026**  
Автор: Ахмедзянов Р.Р.

</div>

---

## Описание

Модуль автоматизирует этап первичной обработки входящих документов в корпоративной ECM-системе:

- принимает документ (PDF, изображение или текстовый файл) через REST API
- распознаёт текст (OCR)
- определяет тип документа (акт, накладная, приказ, счёт-фактура, платёжное поручение)
- извлекает ключевые реквизиты (номер, дата, стороны, суммы и др.)
- оценивает уверенность в каждом результате и направляет сомнительные случаи на ручную верификацию
- возвращает структурированный JSON для автозаполнения карточки ECM

Модуль является **внешним дополнением** к ECM-платформе и не заменяет её базовую функциональность.

---

## Архитектура

```
HTTP Request (PDF/image/txt)
│
▼
┌─────────────────┐
│   FastAPI (API) │  ← точка входа, координация
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────────┐
│  OCR  │  │  ECM-адаптер     │  ← справочник контрагентов
└───┬───┘  └──────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  LLM-сервис (Qwen 2.5 7B)     │  ← классификация + извлечение
│  + постпроцессоры             │  ← нормализация + rule-based fixes
└────────────────┬──────────────┘
                 │
                 ▼
┌────────────────────────────┐
│  Confidence Router         │  ← пороги из config/thresholds.yaml
└────────────────┬───────────┘
                 │
                 ▼
           JSON-ответ → ECM
```

**Компоненты:**

| Компонент | Файл | Описание |
|-----------|------|----------|
| API-шлюз | `app/api/main.py` | FastAPI, приём файлов, координация |
| OCR-сервис | `app/services/ocr/service.py` | PaddleOCR, кириллическая модель, deskew |
| LLM-сервис | `app/services/llm/service.py` | Qwen 2.5 7B через Ollama, постпроцессоры |
| Confidence Router | `app/services/router/confidence_router.py` | Пороги уверенности per-атрибут |
| ECM-адаптер | `app/adapters/ecm/adapter.py` | Запрос схем и контрагентов из ECM |
| Mock ECM | `mock_ecm/main.py` | Заглушка ECM REST API для локальной разработки |

---

## Технический стек

- **Python 3.11+** · FastAPI · Uvicorn
- **OCR:** PaddleOCR (`PP-OCRv4_mobile_det` + `eslav_PP-OCRv5_mobile_rec`) · OpenCV · PyMuPDF · pypdfium2
- **LLM:** Qwen 2.5 7B Instruct Q4\_K\_M · [Ollama](https://ollama.com) · Ollama Structured Output (JSON Schema)
- **Инфраструктура:** structlog · rapidfuzz · pydantic-settings
- **Сборка:** [uv](https://github.com/astral-sh/uv)

---

## Требования

- Python 3.11 или новее
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — менеджер пакетов
- [Ollama](https://ollama.com/download) — для запуска LLM
- Модель: `ollama pull qwen2.5:7b-instruct-q4_K_M`
- RAM: минимум 16 ГБ (рекомендуется 32 ГБ для комфортной работы)

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/rus-yanov/ecm-ai-module.git
cd ecm-ai-module
```

### 2. Установить зависимости

```bash
uv sync
```

### 3. Загрузить LLM-модель

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### 4. Настроить конфигурацию

```bash
cp .env.example .env
# При необходимости отредактируй .env
```

### 5. Запустить сервисы

```bash
# Mock ECM (заглушка, имитирует ECM REST API)
uv run uvicorn mock_ecm.main:app --port 8001 &

# Основной модуль
uv run uvicorn app.api.main:app --port 8000
```

### 6. Проверить работоспособность

```bash
curl http://localhost:8000/healthz
curl http://localhost:8001/healthz
```

---

## API

### Входящие запросы (модуль принимает)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/healthz` | Проверка работоспособности |
| `POST` | `/api/v1/documents/process` | Обработка документа |

**POST /api/v1/documents/process** — принимает `multipart/form-data`:

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | файл | PDF, изображение или текстовый файл |

Тип документа определяется автономно — модуль классифицирует документ на основе текста и few-shot примеров; передавать тип заранее не требуется.

```bash
curl -X POST http://localhost:8000/api/v1/documents/process \
  -F "file=@/path/to/document.pdf"
```

**Пример ответа:**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_type": "ACT",
  "type_confidence": 0.98,
  "attributes": [
    {"name": "act_number", "value": "15",         "confidence": 0.99, "requires_verification": false},
    {"name": "act_date",   "value": "2026-02-12", "confidence": 0.97, "requires_verification": false},
    {"name": "executor",   "value": "ООО «А-Сервис»", "confidence": 0.95, "requires_verification": false}
  ],
  "requires_manual_review": false,
  "total_processing_time_sec": 31.4
}
```

### Исходящие запросы (модуль отправляет в ECM)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/dictionaries/{name}` | Получить справочник (контрагенты) для fuzzy-нормализации |
| `POST` | `/cards` | Создать карточку документа в ECM |

> `GET /schemas/{schema_id}` реализован в Mock ECM и сохранён в `EcmAdapter`, но в текущей версии пайплайна не вызывается: тип документа определяется автономно LLM-сервисом, входящий `schema_id` из запроса удалён.

Все запросы к ECM отправляются с заголовком `X-API-Key` и имеют retry-логику: до 3 повторных попыток с задержками 1 → 3 → 9 с.

---

## Поддерживаемые типы документов

| Тип | Название | Основные извлекаемые реквизиты |
|-----|----------|-------------------------------|
| `ACT` | Акт выполненных работ | Номер, дата, исполнитель, ИНН, заказчик |
| `WAYBILL` | Требование-накладная (М-11) | Номер, дата составления, отправитель, получатель |
| `ORDER` | Приказ | Номер, дата, заголовок, подписант |
| `INVOICE` | Счёт-фактура | Номер, дата, продавец/покупатель, ИНН, сумма с НДС |
| `PAYMENT` | Платёжное поручение | Плательщик, получатель, ИНН, сумма, назначение |

---

## Результаты эксперимента

Эксперимент проводился на **27 реальных PDF-документах** (все 5 типов) в двух режимах.  
Температура инференса LLM: 0.1.

### Track A — text fast-path (без OCR, 12 документов)

Применяется когда PDF содержит встроенный текстовый слой; OCR пропускается.

| Тип | Docs | Accuracy | Attr accuracy |
|-----|------|----------|---------------|
| ACT | 2 | 100% | — (rule override типа, атрибуты не извлекаются) |
| WAYBILL | 5 | 100% | **100%** |
| ORDER | 4 | 100% | 82% |
| INVOICE | 1 | 100% | 60% |
| **ИТОГО** | **12** | **100%** | **~83%** |

**Производительность:** среднее время обработки 34 с (CPU, без GPU).

### Track B — OCR pipeline (сканы, 15 документов)

Применяется для отсканированных PDF без текстового слоя; фильтры уверенности и когерентности OCR отсеивают нечитаемые страницы.

| Тип | Docs | Accuracy | Причины отказов |
|-----|------|----------|----------------|
| ORDER | 11 | 27% | 5 garbled OCR (когерентность < 35%), 2 timeout (37/69 стр.), 1 low-conf |
| PAYMENT | 4 | 100% | — |
| **ИТОГО** | **15** | **33%** | — |

**Производительность:** среднее время обработки 51 с на документ (CPU).

### Сводка

| | Docs | Type accuracy |
|---|---|---|
| Track A (text) | 12 | **100%** |
| Track B (scans) | 15 | 33% |
| **Всего** | **27** | **63%** |

> Незначительные колебания per-type метрик между прогонами — следствие ненулевой температуры LLM (temperature=0.1); общая точность 63% воспроизводима.

Подробные результаты: [`experiment/metrics_real.json`](experiment/metrics_real.json) · [`experiment/results_real.json`](experiment/results_real.json)

---

## Структура проекта

```
ecm-ai-module/
├── app/
│   ├── api/              # FastAPI роутеры и main
│   ├── services/
│   │   ├── ocr/          # OCR-сервис (PaddleOCR)
│   │   ├── llm/          # LLM-сервис (Qwen + постпроцессоры)
│   │   └── router/       # Confidence Router
│   ├── adapters/ecm/     # ECM-адаптер
│   ├── domain/           # Доменные модели (Pydantic)
│   └── config/           # Настройки приложения
├── config/
│   └── thresholds.yaml   # Пороги уверенности по атрибутам
├── mock_ecm/             # Заглушка ECM REST API
├── experiment/
│   ├── real_ground_truth.json     # Разметка (27 документов: text + scan)
│   ├── metrics_real.json          # Итоговые метрики
│   └── results_real.json          # Per-document результаты
├── scripts/              # Вспомогательные скрипты
│   ├── run_experiment.py
│   └── build_real_ground_truth.py
├── tests/                # Тесты (pytest-asyncio)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Конфигурация

Основные параметры в `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
OLLAMA_TIMEOUT_SEC=180
ECM_BASE_URL=http://localhost:8001
OCR_CONFIDENCE_THRESHOLD=0.25
MAX_FILE_SIZE_MB=20
```

Пороги уверенности по атрибутам — в `config/thresholds.yaml`.

---

## Запуск эксперимента

```bash
# Track A — только текстовые документы (без OCR, серверы уже запущены)
uv run python scripts/run_experiment.py --real --text-only --no-servers

# Track A + B — текст и сканы (требует запущенного сервера с OCR)
uv run python scripts/run_experiment.py --real --no-servers

# Просмотр результатов
cat experiment/metrics_real.json
```

---

## Ограничения прототипа

- OCR на реальных сканах: точность классификации 33% — основные потери на нечитаемых/многостраничных сканах; текстовые документы обрабатываются со 100% точностью
- Атрибуты PAYMENT (сумма, ИНН) плохо извлекаются из зашумлённого OCR-текста платёжных форм
- Производительность: text fast-path ≈ 34 с, OCR-скан ≈ 51 с (CPU); GPU даст 5–10× ускорение
- Контракт API принимает только `file`; тип документа определяется автономно (schema_id удалён)
- Выборка (27 документов) — разведочный эксперимент, достаточный для прототипа ВКР

---

## Академический контекст

Проект разработан в рамках ВКР по теме:  
**«Интеллектуализация обработки документов в ECM-системе средствами OCR и искусственного интеллекта»**

НИУ ВШЭ · Высшая Школа Бизнеса · Электронный бизнес и цифровые инновации · 2026

---

## Лицензия

MIT
