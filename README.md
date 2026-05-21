# FIAP - IoT - CP3

Projetos de visão computacional com MediaPipe e PySide6 para monitoramento de postura e detecção de sonolência.

## Integrantes

Felipe Cerbonchini - RM 554909
Vitor Chaves - RM 557067

## Pré-requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

## Instalação

```bash
# Instalar dependências
uv sync
```

## Executando os projetos

### Ex1 - Jumping Jack Tracker

Detector de polichinelos usando MediaPipe Pose Landmarker.

```bash
uv run python ex1/main.py
```

**Funcionalidades:**
- Detecção de pose em tempo real
- Contagem automática de repetições
- Timeout de 0.75s para validar o ciclo completo
- Indicador de EXERCISING / IDLE
- Controles de debug: landmarks, skeleton, resolução

### Ex2 - Drowsiness Detection

Detector de sonolência usando MediaPipe Face Landmarker.

```bash
uv run python ex2/main.py
```

**Funcionalidades:**
- Cálculo de EAR (Eye Aspect Ratio) com suavização exponencial
- Detecção de head nod (cabeça caindo)
- Histerese para evitar flickering entre estados
- Alerta visual DROWSY no vídeo
- Controles de debug: EAR, head tilt, landmarks, resolução