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
        "dest_comfy": COMFY_ROOT / "models/annotators",
        "min_size": 50 * 1024 * 1024 # 50MB
    },
    {
        "name": "dw-ll_ucoco_384_bs5.torchscript.pt",
        "url": "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt",
        "dest_volume": VOLUME_ROOT / "models/annotators",
        "dest_comfy": COMFY_ROOT / "models/annotators",
        "min_size": 50 * 1024 * 1024 # 50MB
    },
    {
        "name": "clip_vision_h.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        "dest_volume": VOLUME_ROOT / "models/clip_vision",
        "dest_comfy": COMFY_ROOT / "models/clip_vision",
        "min_size": 1000 * 1024 * 1024 # 1GB
    },
    {
        "name": "wan_2.1_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "dest_volume": VOLUME_ROOT / "models/vae",
        "dest_comfy": COMFY_ROOT / "models/vae",
        "min_size": 500 * 1024 * 1024 # 500MB
    },
    {
        "name": "umt5-xxl-enc-bf16.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
        "dest_volume": VOLUME_ROOT / "models/text_encoders",
        "dest_comfy": COMFY_ROOT / "models/text_encoders",
        "min_size": 10 * 1024 * 1024 * 1024 # 10GB
    },
    {
        "name": "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        "dest_volume": VOLUME_ROOT / "models/diffusion_models",
        "dest_comfy": COMFY_ROOT / "models/diffusion_models",
        "min_size": 10 * 1024 * 1024 * 1024 # 10GB
    },
    {
        "name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "dest_volume": VOLUME_ROOT / "models/diffusion_models/WanVideo/2_2",
        "dest_comfy": COMFY_ROOT / "models/diffusion_models/WanVideo/2_2",
        "min_size": 10 * 1024 * 1024 * 1024 # 10GB
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
        min_size = m.get("min_size", 1024 * 1024)
        
        # Define onde o arquivo final deve estar fisicamente
        physical_dest = m["dest_volume"] if use_volume else m["dest_comfy"]
        physical_file = physical_dest / name
        
        # 1. Faz o download físico se o arquivo não existir ou for menor que o tamanho mínimo
        if not physical_file.exists() or physical_file.stat().st_size < min_size:
            if physical_file.exists():
                print(f"[!] {name} existe mas tem tamanho inválido ({physical_file.stat().st_size} bytes). Removendo...")
                try:
                    physical_file.unlink()
                except Exception as ex:
                    print(f"[!] Erro ao remover arquivo inválido: {ex}")
            print(f"[!] {name} não encontrado ou inválido em {physical_dest}. Iniciando download...")
            download_file(url, physical_file)
        else:
            size_gb = physical_file.stat().st_size / (1024 * 1024 * 1024)
            print(f"[OK] {name} já existe fisicamente em {physical_dest} com tamanho válido ({size_gb:.2f} GB). Download ignorado.")
            
        # 2. Cria o link simbólico na pasta que o ComfyUI espera
        target_comfy_dir = m["dest_comfy"]
        target_comfy_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = target_comfy_dir / name
        
        if use_volume:
            if symlink_path.exists() or symlink_path.is_symlink():
                print(f"[!] Removendo link simbólico antigo/existente: {symlink_path}")
                try:
                    symlink_path.unlink()
                except Exception as ex:
                    print(f"[!] Erro ao remover link simbólico: {ex}")
            
            try:
                symlink_path.symlink_to(physical_file)
                print(f"[OK] Link simbólico criado: {symlink_path} -> {physical_file}")
            except Exception as ex:
                print(f"[ERRO] Falha ao criar link simbólico: {ex}")
                sys.exit(1)

    # 3. CRITICO: Cria symlink no nivel RAIZ do diffusion_models para o modelo Animate
    # O WanAnimateToVideo com UNETLoader procura modelos em models/diffusion_models/ no nivel raiz,
    # mas o Animate foi baixado para uma subpasta WanVideo/2_2/.
    _animate_model_name = "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
    _animate_root_symlink = COMFY_ROOT / "models/diffusion_models" / _animate_model_name
    _animate_volume_file = VOLUME_ROOT / "models/diffusion_models/WanVideo/2_2" / _animate_model_name if use_volume else COMFY_ROOT / "models/diffusion_models/WanVideo/2_2" / _animate_model_name

    if _animate_volume_file.exists():
        if _animate_root_symlink.exists() or _animate_root_symlink.is_symlink():
            print(f"[!] Removendo symlink raiz existente: {_animate_root_symlink}")
            try:
                _animate_root_symlink.unlink()
            except Exception as ex:
                print(f"[!] Erro ao remover symlink raiz: {ex}")

        try:
            _animate_root_symlink.symlink_to(_animate_volume_file)
            print(f"[OK] Symlink raiz criado: {_animate_root_symlink} -> {_animate_volume_file}")
        except Exception as ex:
            print(f"[ERRO] Falha ao criar symlink raiz: {ex}")
            sys.exit(1)
    else:
        print(f"[!] Arquivo do modelo Animate nao encontrado em {_animate_volume_file}. Symlink raiz ignorado.")

    # Cria os symlinks específicos de segurança para o DWPose
    dwpose_volume_yolox = VOLUME_ROOT / "models/annotators/yolox_l.torchscript.pt" if use_volume else COMFY_ROOT / "models/annotators/yolox_l.torchscript.pt"
    dwpose_volume_dw = VOLUME_ROOT / "models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt" if use_volume else COMFY_ROOT / "models/annotators/dw-ll_ucoco_384_bs5.torchscript.pt"
    
    dwpose_comfy_dir = COMFY_ROOT / "custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5"
    dwpose_comfy_dir.mkdir(parents=True, exist_ok=True)
    
    sym_yolox = dwpose_comfy_dir / "yolox_l.torchscript.pt"
    sym_dw = dwpose_comfy_dir / "dw-ll_ucoco_384_bs5.torchscript.pt"
    
    for sym, src in [(sym_yolox, dwpose_volume_yolox), (sym_dw, dwpose_volume_dw)]:
        if sym.exists() or sym.is_symlink():
            try:
                sym.unlink()
            except Exception as ex:
                print(f"[!] Erro ao remover link DWPose: {ex}")
        try:
            sym.symlink_to(src)
            print(f"[OK] Link DWPose criado: {sym} -> {src}")
        except Exception as ex:
            print(f"[ERRO] Falha ao criar link DWPose: {ex}")
            sys.exit(1)

    print("[*] [STARTUP] Todos os modelos resolvidos e linkados! Inicializando ComfyUI...")

if __name__ == "__main__":
    main()
