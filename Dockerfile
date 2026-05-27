# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.4-base

ARG HF_TOKEN=""

# Garante que o utilitário curl esteja instalado no container base para os downloads dos modelos
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# IMPORTANTE: Altere esta data para forçar o Docker a re-clonar os custom nodes
# e garantir versoes atualizadas (ex: WanVideoWrapper com start_step/end_step corrigidos)
ARG CACHE_BUST=2026-05-27

# install custom nodes + dependencies
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    pip install -r requirements.txt || true

RUN git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux /comfyui/custom_nodes/comfyui_controlnet_aux && \
    cd /comfyui/custom_nodes/comfyui_controlnet_aux && \
    pip install -r requirements.txt || true

RUN git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt || true

# CRITICO: Echo do CACHE_BUST invalida o cache do Docker para re-clonar o WanVideoWrapper
# Isso garante que start_step e end_step estejam corretamente no INPUT_TYPES
RUN echo "WanVideoWrapper build date: ${CACHE_BUST}" && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper /comfyui/custom_nodes/ComfyUI-WanVideoWrapper && \
    cd /comfyui/custom_nodes/ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt || true


# CRITICO: instala dependencias Python do WanVideoWrapper (sem isso os nodes nao carregam)
RUN pip install "diffusers>=0.33.0" accelerate einops "peft>=0.17" ftfy

# PERFORMANCE: Instala o SageAttention (acelera a inferência do WanVideo em até 2x!)
RUN pip install sageattention==2.2.0 --no-build-isolation || true

# Copia o script de download/linkagem de modelos dinâmicos para dentro do container
COPY download_models.py /comfyui/download_models.py

# Quando o container inicia, roda o download_models.py primeiro (que detecta e anexa o volume da RunPod)
# e depois passa o controle para o script start.sh padrão da RunPod.
CMD ["sh", "-c", "python /comfyui/download_models.py && /start.sh"]
