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
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt' \
    --relative-path models/annotators \
    --filename 'yolox_l.torchscript.pt' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt' \
    --relative-path models/annotators \
    --filename 'dw-ll_ucoco_384_bs5.torchscript.pt' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# SEGURANÇA: Cria symlinks para o DWPose (evita que o node cometa o erro de fazer download fantasma na hora de rodar o vídeo)
RUN mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/ && \
    ln -s /comfyui/models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt || true && \
    ln -s /comfyui/models/annotators/yolox_l.torchscript.pt /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/yolox_l.torchscript.pt || true

# CLIP Vision
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors' \
    --relative-path models/clip_vision \
    --filename 'clip_vision_h.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# VAE
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors' \
    --relative-path models/vae \
    --filename 'wan_2.1_vae.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# Text Encoder UMT5-XXL
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors' \
    --relative-path models/text_encoders \
    --filename 'umt5-xxl-enc-bf16.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# Modelo principal correto: Wan2.1 I2V 14B fp8 (Image-to-Video em FP8 para caber nos 24GB de VRAM)
# NOTA: O nome correto no repositório do Comfy-Org tem "_480p_" no meio!
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors' \
    --relative-path models/diffusion_models \
    --filename 'wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done
