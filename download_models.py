import os
import sys
import subprocess
from pathlib import Path

# Pasta persistente do Volume de Rede da RunPod
VOLUME_ROOT = Path("/workspace")
COMFY_ROOT = Path("/comfyui")

MODELS_TO_DOWNLOAD = [
    {
        "name": "yolox_l.torchscript.pt",
        "url": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt",
        "dest_volume": VOLUME_ROOT / "models/annotators",
        "dest_comfy": COMFY_ROOT / "models/annotators"
    },
    {
        "name": "dw-ll_ucoco_384_bs5.torchscript.pt",
        "url": "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt",
        "dest_volume": VOLUME_ROOT / "models/annotators",
        "dest_comfy": COMFY_ROOT / "models/annotators"
    },
    {
        "name": "clip_vision_h.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        "dest_volume": VOLUME_ROOT / "models/clip_vision",
        "dest_comfy": COMFY_ROOT / "models/clip_vision"
    },
    {
        "name": "wan_2.1_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "dest_volume": VOLUME_ROOT / "models/vae",
        "dest_comfy": COMFY_ROOT / "models/vae"
    },
    {
        "name": "umt5-xxl-enc-bf16.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
        "dest_volume": VOLUME_ROOT / "models/text_encoders",
        "dest_comfy": COMFY_ROOT / "models/text_encoders"
    },
    {
        "name": "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        "dest_volume": VOLUME_ROOT / "models/diffusion_models",
        "dest_comfy": COMFY_ROOT / "models/diffusion_models"
    },
    {
        "name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo-comfy/resolve/main/Wan2_2/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "dest_volume": VOLUME_ROOT / "models/diffusion_models/WanVideo/2_2",
        "dest_comfy": COMFY_ROOT / "models/diffusion_models/WanVideo/2_2"
    }
]

def download_file(url, dest_path):
    print(f"[*] Baixando {url} para {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.with_suffix(".download")
    
    # Executa curl para fazer o download com barra de progresso
    cmd = [
        "curl", "-L", "-f",
        "--connect-timeout", "30",
        "--retry", "5",
        url,
        "-o", str(temp_dest)
    ]
    
    # Adiciona token se disponível nas variáveis de ambiente
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        cmd.extend(["-H", f"Authorization: Bearer {hf_token}"])
        
    try:
        subprocess.run(cmd, check=True)
        temp_dest.rename(dest_path)
        print(f"[OK] Download de {dest_path.name} concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Falha ao baixar {url}: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        sys.exit(1)

def main():
    print("[*] [STARTUP] Inicializando resolvedor de modelos dinâmico (Kinomuse)...")
    
    # Determina se há um volume persistente disponível
    use_volume = VOLUME_ROOT.exists() and os.access(VOLUME_ROOT, os.W_OK)
    if use_volume:
        print("[*] [STARTUP] Volume persistente /workspace detectado. Modelos serão mantidos lá!")
    else:
        print("[!] [STARTUP] Volume persistente /workspace não encontrado. Baixando na partição local temporária do container.")

    for m in MODELS_TO_DOWNLOAD:
        name = m["name"]
        url = m["url"]
        
        # Define onde o arquivo final deve estar fisicamente
        physical_dest = m["dest_volume"] if use_volume else m["dest_comfy"]
        physical_file = physical_dest / name
        
        # 1. Faz o download físico se o arquivo não existir
        if not physical_file.exists() or physical_file.stat().st_size < 1024 * 1024: # menor que 1MB é inválido
            print(f"[!] {name} não encontrado fisicamente em {physical_dest}. Iniciando download...")
            download_file(url, physical_file)
        else:
            print(f"[OK] {name} já existe fisicamente em {physical_dest}. Download ignorado.")
            
        # 2. Cria o link simbólico na pasta que o ComfyUI espera
        target_comfy_dir = m["dest_comfy"]
        target_comfy_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = target_comfy_dir / name
        
        if use_volume:
            if symlink_path.exists():
                if symlink_path.is_symlink():
                    symlink_path.unlink()
                else:
                    print(f"[!] Conflito: {symlink_path} existe e não é link simbólico. Removendo...")
                    symlink_path.unlink()
            
            symlink_path.symlink_to(physical_file)
            print(f"[OK] Link simbólico criado: {symlink_path} -> {physical_file}")

    # Cria os symlinks específicos de segurança para o DWPose
    dwpose_volume_yolox = VOLUME_ROOT / "models/annotators/yolox_l.torchscript.pt" if use_volume else COMFY_ROOT / "models/annotators/yolox_l.torchscript.pt"
    dwpose_volume_dw = VOLUME_ROOT / "models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt" if use_volume else COMFY_ROOT / "models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt"
    
    dwpose_comfy_dir = COMFY_ROOT / "custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5"
    dwpose_comfy_dir.mkdir(parents=True, exist_ok=True)
    
    sym_yolox = dwpose_comfy_dir / "yolox_l.torchscript.pt"
    sym_dw = dwpose_comfy_dir / "dw-ll_ucoco_384_bs5.torchscript.pt"
    
    for sym, src in [(sym_yolox, dwpose_volume_yolox), (sym_dw, dwpose_volume_dw)]:
        if sym.exists() or sym.is_symlink():
            sym.unlink()
        sym.symlink_to(src)
        print(f"[OK] Link DWPose criado: {sym} -> {src}")

    print("[*] [STARTUP] Todos os modelos resolvidos e linkados! Inicializando ComfyUI...")

if __name__ == "__main__":
    main()
