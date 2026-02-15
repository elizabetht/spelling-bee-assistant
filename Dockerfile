FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr ca-certificates git g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps "pipecat-ai[elevenlabs]>=0.0.100"

# Patch nvidia-pipecat for pipecat >=0.0.100 compatibility
# (FrameSerializerType was removed; ACE serializer is always binary)
RUN NVPC=$(python -c "import nvidia_pipecat; import os; print(os.path.dirname(nvidia_pipecat.__file__))") && \
    sed -i 's/from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType/from pipecat.serializers.base_serializer import FrameSerializer/' \
        "$NVPC/transports/network/ace_fastapi_websocket.py" && \
    sed -i 's/is_binary = self._params.serializer.type == FrameSerializerType.BINARY/is_binary = True/' \
        "$NVPC/transports/network/ace_fastapi_websocket.py" && \
    python -c "import pathlib,sys;f=pathlib.Path(sys.argv[1]+'/serializers/ace_websocket.py');s=f.read_text();s=s.replace('from pipecat.serializers.base_serializer import (\n    FrameSerializer,\n    FrameSerializerType,\n)','from pipecat.serializers.base_serializer import FrameSerializer');s=s.replace('def type(self) -> FrameSerializerType:','def type(self):');s=s.replace('return FrameSerializerType.BINARY','return True');f.write_text(s)" "$NVPC"

COPY spelling_bee_agent_backend.py ./
COPY guardrails ./guardrails
COPY ui ./ui

EXPOSE 8080

CMD ["python", "spelling_bee_agent_backend.py"]
