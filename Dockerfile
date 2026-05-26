# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.4-base

ARG HF_TOKEN=""
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

# DWPose annotators (baixa os modelos de pose)
RUN mkdir -p /comfyui/models/annotators && \
    curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt" \
    -o /comfyui/models/annotators/yolox_l.torchscript.pt

RUN curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt" \
    -o /comfyui/models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt

# SEGURANÇA: Cria symlinks para o DWPose (evita que o node cometa o erro de fazer download fantasma na hora de rodar o vídeo)
RUN mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/ && \
    ln -s /comfyui/models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt || true && \
    ln -s /comfyui/models/annotators/yolox_l.torchscript.pt /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/yolox_l.torchscript.pt || true

# CLIP Vision
RUN mkdir -p /comfyui/models/clip_vision && \
    curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    -o /comfyui/models/clip_vision/clip_vision_h.safetensors

# VAE
RUN mkdir -p /comfyui/models/vae && \
    curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    -o /comfyui/models/vae/wan_2.1_vae.safetensors

# Text Encoder UMT5-XXL
RUN mkdir -p /comfyui/models/text_encoders && \
    curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" \
    -o /comfyui/models/text_encoders/umt5-xxl-enc-bf16.safetensors

# Modelo principal correto: Wan2.1 I2V 14B fp8 (Image-to-Video em FP8 para caber nos 24GB de VRAM)
# NOTA: O nome correto no repositório do Comfy-Org tem "_480p_" no meio!
RUN mkdir -p /comfyui/models/diffusion_models && \
    curl -L -f --connect-timeout 30 --retry 5 \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors" \
    -o /comfyui/models/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors
