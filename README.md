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
- распознаёт текст (OCR) при необходимости — с поддержкой кириллицы
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
│  OCR  │  │  ECM-адаптер     │  ← схема + контрагенты
└───┬───┘  └──────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  LLM-сервис (Qwen 2.5 7B)    │  ← классификация + извлечение
│  + постпроцессоры             │  ← нормализация + rule-based fixes
└────────────────┬──────────────┘
                 │
                 ▼
┌────────────────────────────┐
│  Confidence Router          │  ← пороги из config/thresholds.yaml
└────────────────┬────────────┘
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
- RAM: минимум 8 ГБ (рекомендуется 16 ГБ для комфортной работы)

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

## Использование

### Обработка документа через API

```bash
curl -X POST http://localhost:8000/api/v1/documents/process \
  -F "file=@/path/to/document.pdf" \
  -F "schema_id=act"
```

### Пример ответа

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

Эксперимент проводился на 12 реальных текстовых PDF-документах (4 типа из 5).  
Режим: text fast-path (без OCR), температура инференса: 0.1.

| Тип | Docs | Accuracy | Avg F1 (атрибуты) |
|-----|------|----------|-------------------|
| ACT | 2 | 100% | — (override типа) |
| WAYBILL | 5 | 100% | **1.000** |
| ORDER | 4 | 100% | 0.702 |
| INVOICE | 1 | 100% | 0.600 |
| **ИТОГО** | **12** | **100%** | **~0.73** |

**Производительность:** среднее время обработки 34.1 с (CPU, без GPU).

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
│   ├── real_ground_truth.json     # Разметка (12 документов)
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
# Эксперимент на реальных документах (text fast-path, серверы уже запущены)
uv run python scripts/run_experiment.py --real --text-only --no-servers

# Просмотр результатов
cat experiment/metrics_real.json
```

---

## Ограничения прототипа

- OCR на реальных сканах реализован, но не верифицирован экспериментально
- PAYMENT не проверен экспериментально (нет текстовых образцов в корпусе)
- Производительность: mean ≈ 34 с на CPU; GPU даст 5–10× ускорение
- Выборка (12 документов) — разведочный эксперимент, достаточный для прототипа ВКР

---

## Академический контекст

Проект разработан в рамках ВКР по теме:  
**«Интеллектуализация обработки документов в ECM-системе средствами OCR и искусственного интеллекта»**

НИУ ВШЭ · Высшая Школа Бизнеса · Эллектронный бизнес и цифровые инновации · 2026

---

## Лицензия

MIT
