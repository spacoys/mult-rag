# Multimodal RAG для русскоязычных PDF

Система поиска и ответов на вопросы по PDF-документам с поддержкой текста, изображений и таблиц. Гибридный поиск (dense + BM25), cross-encoder реранкер, дедупликация результатов по страницам.

---

## Возможности

- **Гибридный поиск**: dense-эмбеддинги (`multilingual-e5`) + BM25 (лемматизация через `pymorphy3`)
- **Реранкер**: cross-encoder для точной пересортировки кандидатов
- **Дедупликация по страницам**: одна страница — один результат
- **Мультимодальность**: извлечение текста и изображений из PDF
- **Расширение запроса синонимами**: `умеет` → `уметь`, `умение`, `навык`, `способность`
- **Streamlit UI**: загрузка PDF и поиск в браузере

- ### Пайплайн индексации

1. **Extract** — PyMuPDF читает PDF, извлекает текст по страницам и изображения
2. **Chunk** — текст режется на чанки по границам абзацев и предложений (`chunk_size=1000`, `overlap=150`)
3. **Embed** — текстовые чанки → `multilingual-e5-base` (768-мерные), изображения → `open_clip` (512-мерные)
4. **Store** — векторы сохраняются в Qdrant (локальный режим, две именованные коллекции: `text` и `image`)
5. **BM25** — токенизированный корпус сохраняется в pickle для гибридного поиска

### Пайплайн поиска

1. **Dense search** — запрос → эмбеддинг → top-100 из Qdrant
2. **BM25 search** — запрос → лемматизация + синонимы → top-100 по BM25
3. **RRF** — Reciprocal Rank Fusion объединяет два списка
4. **Rerank** — cross-encoder пересчитывает близость для топ-50
5. **Dedupe** — оставляем лучший чанк на каждую страницу


### Запуск

```bash
streamlit run ui/app.py
```
Пример поиска по resume.pdf


<img width="1757" height="236" alt="image" src="https://github.com/user-attachments/assets/6c755fe1-8628-4afe-a6e3-4733b3688129" />

<img width="1445" height="617" alt="image" src="https://github.com/user-attachments/assets/20f70391-1c61-49af-b51a-926843707dac" />

<img width="1790" height="623" alt="image" src="https://github.com/user-attachments/assets/1991dc1d-1231-4ac8-b1bd-8f7a05485c88" />

<img width="1798" height="265" alt="image" src="https://github.com/user-attachments/assets/006b00fa-7248-4c21-874b-b4c74ef1d842" />


