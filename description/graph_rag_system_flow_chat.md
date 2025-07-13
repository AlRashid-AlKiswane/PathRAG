
```mermaid
flowchart TD

    %% Uploading & Preprocessing
    A[1️⃣ Upload Document(s)] --> B[🔗 Chunk Documents]
    B --> C[💾 Save to DB: 'chunks']
    C --> D[📐 Embed Chunks]
    D --> E[💾 Save to DB: 'embed_vector']
    E --> F[🧠 Extract NER Entities]
    F --> G[💾 Save to DB: 'ner_entities']

    %% Embedding Route → FAISS Init
    G --> H[2️⃣ 🔁 /embedding-chunks API Called]
    H --> I[⚙️ Initialize FaissRAG]
    I --> J[🏗️ Build FAISS Index from 'embed_vector']

    %% Retrieval
    J --> K[3️⃣ 🔍 /retrieve API Called]
    K --> L{Retrieval Mode}
    L -->|faiss_only| M[🔎 FAISS Semantic Search]
    L -->|entity_only| N[🧬 Entity-Level Search]
    L -->|intersection| O[FAISS ∩ Entity Results]
    L -->|union| P[FAISS ∪ Entity Results]

    %% Chatbot Flow
    P --> Q[4️⃣ 💬 /chatbot API Called]
    Q --> R{Cache Available?}
    R -->|Yes| S[✅ Fetch LLM Response from DB: 'query_requests']
    R -->|No| T[❌ Perform Dual-Level Retrieval]
    T --> U[🧠 Embed Query → FAISS & NER Retrieval]
    U --> V[🗂️ Combine Results (by Mode)]
    V --> W[🗣️ Generate LLM Response]
    W --> X[💾 Save to 'query_requests']

    style A fill:#e0f7fa,stroke:#26c6da
    style H fill:#f1f8e9,stroke:#8bc34a
    style K fill:#fff3e0,stroke:#ffb300
    style Q fill:#ede7f6,stroke:#7e57c2
```