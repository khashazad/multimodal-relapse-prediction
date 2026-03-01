# Multimodal Relapse Prediction

Multimodal transformer fusion model for non-psychotic relapse detection from wearable sensor data (Track 1). Binary per-day classification (relapse vs. stable) for 9 patients using 5 sensor modalities under Leave-One-Subject-Out (LOSO) cross-validation.

## Model Architecture

```mermaid
graph TB
    subgraph Input["Input Window (B, 7 days)"]
        A1["Accelerometer<br/>(B, 7, 38)"]
        A2["Gyroscope<br/>(B, 7, 38)"]
        A3["Heart Rate<br/>(B, 7, 26)"]
        A4["Steps<br/>(B, 7, 10)"]
        A5["Sleep<br/>(B, 7, 9)"]
        PM["Padding Mask<br/>(B, 7)"]
        MM1["Accel Mask<br/>(B, 7)"]
        MM2["Gyr Mask<br/>(B, 7)"]
        MM3["HR Mask<br/>(B, 7)"]
        MM4["Step Mask<br/>(B, 7)"]
        MM5["Sleep Mask<br/>(B, 7)"]
    end

    subgraph Stage1["Stage 1 — Modality Encoders (independent weights)"]
        subgraph ME1["ModalityEncoder: accel"]
            P1["Linear(38→64) + LayerNorm + GELU"]
            CLS1["Prepend [CLS] token"]
            PE1["+ Positional Embeddings"]
            TE1["TransformerEncoder<br/>1 layer, 4 heads, d=64<br/>pre-norm, GELU, FFN=256"]
            C1["Extract [CLS]<br/>(B, 64)"]
            P1 --> CLS1 --> PE1 --> TE1 --> C1
        end

        subgraph ME2["ModalityEncoder: gyr"]
            P2["Linear(38→64) + LayerNorm + GELU"]
            CLS2["Prepend [CLS] token"]
            PE2["+ Positional Embeddings"]
            TE2["TransformerEncoder<br/>1 layer, 4 heads, d=64"]
            C2["Extract [CLS]<br/>(B, 64)"]
            P2 --> CLS2 --> PE2 --> TE2 --> C2
        end

        subgraph ME3["ModalityEncoder: hr"]
            P3["Linear(26→64) + LayerNorm + GELU"]
            CLS3["Prepend [CLS] token"]
            PE3["+ Positional Embeddings"]
            TE3["TransformerEncoder<br/>1 layer, 4 heads, d=64"]
            C3["Extract [CLS]<br/>(B, 64)"]
            P3 --> CLS3 --> PE3 --> TE3 --> C3
        end

        subgraph ME4["ModalityEncoder: step"]
            P4["Linear(10→64) + LayerNorm + GELU"]
            CLS4["Prepend [CLS] token"]
            PE4["+ Positional Embeddings"]
            TE4["TransformerEncoder<br/>1 layer, 4 heads, d=64"]
            C4["Extract [CLS]<br/>(B, 64)"]
            P4 --> CLS4 --> PE4 --> TE4 --> C4
        end

        subgraph ME5["ModalityEncoder: sleep"]
            P5["Linear(9→64) + LayerNorm + GELU"]
            CLS5["Prepend [CLS] token"]
            PE5["+ Positional Embeddings"]
            TE5["TransformerEncoder<br/>1 layer, 4 heads, d=64"]
            C5["Extract [CLS]<br/>(B, 64)"]
            P5 --> CLS5 --> PE5 --> TE5 --> C5
        end
    end

    subgraph Stage2["Stage 2 — Fusion Transformer"]
        STACK["Stack Modality Tokens<br/>(B, 5, 64)"]
        MEMB["+ Learnable Modality Embeddings"]
        AVAIL["Modality Availability Mask<br/>(B, 5) — any day has data?"]
        FTE["TransformerEncoder<br/>1 layer, 4 heads, d=64<br/>pre-norm, GELU, FFN=256"]
        POOL["Mean Pool over available modalities<br/>(B, 64)"]
        STACK --> MEMB --> FTE --> POOL
        AVAIL -.->|"mask unavailable<br/>modalities"| FTE
    end

    subgraph Stage3["Stage 3 — Classification Head"]
        FC1["Linear(64→64) + GELU"]
        DROP["Dropout(0.1)"]
        FC2["Linear(64→1)"]
        SIG["Sigmoid"]
        OUT["P(relapse)<br/>scalar logit per sample"]
        FC1 --> DROP --> FC2 --> SIG --> OUT
    end

    A1 --> P1
    A2 --> P2
    A3 --> P3
    A4 --> P4
    A5 --> P5

    PM -.->|"mask padded days"| TE1
    PM -.->|"mask padded days"| TE2
    PM -.->|"mask padded days"| TE3
    PM -.->|"mask padded days"| TE4
    PM -.->|"mask padded days"| TE5

    MM1 -.->|"mask missing sensor"| TE1
    MM2 -.->|"mask missing sensor"| TE2
    MM3 -.->|"mask missing sensor"| TE3
    MM4 -.->|"mask missing sensor"| TE4
    MM5 -.->|"mask missing sensor"| TE5

    C1 --> STACK
    C2 --> STACK
    C3 --> STACK
    C4 --> STACK
    C5 --> STACK

    MM1 -.-> AVAIL
    MM2 -.-> AVAIL
    MM3 -.-> AVAIL
    MM4 -.-> AVAIL
    MM5 -.-> AVAIL

    POOL --> FC1

    style Input fill:#1a1a2e,stroke:#e94560,color:#eee
    style Stage1 fill:#16213e,stroke:#0f3460,color:#eee
    style Stage2 fill:#1a1a2e,stroke:#e94560,color:#eee
    style Stage3 fill:#0f3460,stroke:#53a8b6,color:#eee
```

**~300K parameters** with default hyperparameters (d_model=64, 4 heads, 1 encoder layer, 1 fusion layer).
