"""
teste_pipeline.py
=================
Teste completo do pipeline de animacao:
1. Detecta automaticamente resolucao do video de referencia
2. Escolhe a melhor resolucao de geracao (compativel com Wan2.1)
3. Disparo do job RunPod com parametros otimizados para animacao
4. Polling ate COMPLETED
5. Download e salvamento do video final

Uso:
    python teste_pipeline.py
"""

import asyncio
import os
import sys
import base64
import time
import subprocess
import random
import json
import math
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.runpod_serverless_service import serverless_service
from services.config import local_settings

# ---- Arquivos de teste ----
IMG_PATH = r"tests\1774549074946.png"
VID_PATH = r"tests\q90BwG7Pcih8-PB3yRodr_output.mp4"

# ---- Prompt de referencia ----
PROMPT = (
    "A woman dancing, moving dynamically, swaying her body and arms, "
    "smooth natural motion, cinematic quality"
)
NEGATIVE = (
    "blurry, low quality, distorted face, bad anatomy, flickering, "
    "watermark, extra limbs, multiple people, static, frozen, slow motion"
)

# ---- Resolucoes suportadas pelo Wan2.1 Animate (largura x altura) ----
# Todas divisiveis por 16
SUPPORTED_RESOLUTIONS = {
    "portrait": [(480, 832), (512, 768), (576, 1024), (448, 768)],
    "landscape": [(832, 480), (768, 512), (1024, 576), (768, 448)],
    "square": [(512, 512), (768, 768)],
}


def detect_video_resolution(video_path: str) -> tuple:
    """Detecta a resolucao original do video de referencia via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao detectar resolucao do video: {result.stderr}")
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def pick_best_resolution(video_width: int, video_height: int, max_pixels: int = 400000) -> tuple:
    """Escolhe a melhor resolucao suportada que cabe na GPU (max_pixels) e melhor se aproxima do aspect ratio."""
    video_ratio = video_width / video_height

    candidates = []
    for orientation, res_list in SUPPORTED_RESOLUTIONS.items():
        for w, h in res_list:
            pixels = w * h
            if pixels > max_pixels:
                continue
            gen_ratio = w / h
            ratio_diff = abs(video_ratio - gen_ratio)
            candidates.append((ratio_diff, -pixels, w, h, orientation))

    if not candidates:
        candidates = [(0, 0, 480, 832, "portrait")]

    candidates.sort()
    best = candidates[0]
    return best[2], best[3]


async def main():
    print("\n" + "="*60)
    print("  KINOMUSE PIPELINE TEST - Motor Serverless v4.0")
    print("="*60 + "\n")

    if not os.path.exists(IMG_PATH):
        print(f"[ERRO] Imagem nao encontrada: {IMG_PATH}")
        return
    if not os.path.exists(VID_PATH):
        print(f"[ERRO] Video nao encontrado: {VID_PATH}")
        return

    print(f"[OK] Imagem: {IMG_PATH} ({os.path.getsize(IMG_PATH)//1024} KB)")

    # ---- Etapa 0: Deteccao automatica do video ----
    print("\n[0/4] Detectando resolucao do video de referencia...")
    vid_w, vid_h = detect_video_resolution(VID_PATH)
    print(f"[OK] Video de referencia: {vid_w}x{vid_h}")

    gen_w, gen_h = pick_best_resolution(vid_w, vid_h)
    print(f"[OK] Resolucao de geracao escolhida: {gen_w}x{gen_h}")

    # ---- Etapa 1: Compressao do video para a resolucao alvo ----
    print("\n[1/4] Comprimindo video de referencia...")
    os.makedirs("temp", exist_ok=True)
    tag = str(random.randint(1000, 9999))
    temp_vid = f"temp/motion_test_{tag}.mp4"

    cmd = (
        f'ffmpeg -y -i "{VID_PATH}" '
        f'-t 3 '
        f'-vf "scale={gen_w}:{gen_h}:force_original_aspect_ratio=decrease,pad={gen_w}:{gen_h}:(ow-iw)/2:(oh-ih)/2" '
        f'-vcodec libx264 -crf 28 -preset fast -an '
        f'"{temp_vid}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True)

    if os.path.exists(temp_vid):
        size = os.path.getsize(temp_vid) / (1024*1024)
        print(f"[OK] Video comprimido: {size:.2f} MB para {gen_w}x{gen_h}")
    else:
        print("[AVR] Compressao falhou, usando video original")
        temp_vid = VID_PATH

    # ---- Etapa 2: Usar arquivos locais diretamente ----
    print("\n[2/4] Usando arquivos locais (base64 no payload RunPod)...")
    image_url = IMG_PATH
    video_url = temp_vid
    print(f"[OK] Imagem local: {IMG_PATH}")
    print(f"[OK] Video local: {temp_vid}")

    # ---- Etapa 3: Disparo do Job com parametros otimizados ----
    print("\n[3/4] Disparando job no RunPod Serverless...")

    fps = 16
    num_frames = 48
    steps = 20

    seed = random.randint(1, 999999)

    job_id = await serverless_service.run_inference(
        image_url=image_url,
        video_url=video_url,
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=gen_w,
        height=gen_h,
        fps=fps,
        frames=num_frames,
        length=num_frames,
        video_length=num_frames,
        steps=steps,
        cfg=3.0,
        shift=1.0,
        pose_strength=2.5,
        face_strength=1.5,
        denoise_strength=1.0,
        seed=seed,
        local_image_path=IMG_PATH,
        local_video_path=temp_vid
    )

    if not job_id:
        print("[ERRO] Falha ao disparar o job!")
        return

    print(f"[OK] Job ID: {job_id}")
    print(f"[OK] Parametros: {gen_w}x{gen_h} | steps={steps} | cfg=2.5 | shift=2.0 | seed={seed}")

    # ---- Etapa 4: Polling ----
    print("\n[4/4] Aguardando processamento...")
    print("(Isso pode levar de 10 a 30 minutos - H100 no RunPod)\n")

    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        status_data = await serverless_service.check_status(job_id)
        status = status_data.get("status")
        print(f"\r[{elapsed:4d}s] Status: {status}                    ", end="", flush=True)

        if status == "COMPLETED":
            print(f"\n\n[SUCESSO] Job concluido em {elapsed}s!")
            output = status_data.get("output", {})
            print(f"Output recebido: {str(output)[:200]}")

            os.makedirs("outputs", exist_ok=True)

            images_out = output.get("images", [])
            video_b64 = None
            if isinstance(images_out, list) and len(images_out) > 0:
                first_img = images_out[0]
                video_b64 = first_img.get("data") if isinstance(first_img, dict) else first_img

            if not video_b64:
                video_b64 = output.get("video") or output.get("video_base64") if isinstance(output, dict) else None

            video_out_url = output.get("video_url") or output.get("url") if isinstance(output, dict) else None

            if video_b64:
                out_path_webp = f"outputs/test_result_{tag}.webp"
                out_path_mp4 = f"outputs/test_result_{tag}.mp4"

                with open(out_path_webp, "wb") as f:
                    f.write(base64.b64decode(video_b64))
                print(f"[OK] Video (WEBP bruto) salvo em: {out_path_webp}")

                from PIL import Image as PILImage
                try:
                    webp = PILImage.open(out_path_webp)
                    n_frames = getattr(webp, "n_frames", 1)
                    webp.close()

                    temp_dir = f"temp/frames_{tag}"
                    os.makedirs(temp_dir, exist_ok=True)

                    webp = PILImage.open(out_path_webp)
                    for i in range(n_frames):
                        webp.seek(i)
                        frame = webp.copy()
                        if frame.mode == "RGBA":
                            bg = PILImage.new("RGB", frame.size, (0, 0, 0))
                            bg.paste(frame, mask=frame.split()[3])
                            frame = bg
                        elif frame.mode != "RGB":
                            frame = frame.convert("RGB")
                        frame.save(os.path.join(temp_dir, f"frame_{i:04d}.png"), "PNG")
                    webp.close()

                    subprocess.run(
                        ["ffmpeg", "-y", "-framerate", str(fps),
                         "-i", os.path.join(temp_dir, "frame_%04d.png"),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-crf", "18", out_path_mp4],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    size_mb = os.path.getsize(out_path_mp4) / (1024 * 1024)
                    print(f"[OK] Video (MP4) convertido: {out_path_mp4} ({size_mb:.2f} MB)")
                except Exception as e:
                    print(f"[!] Aviso: Nao foi possivel converter para MP4: {e}")

            elif video_out_url:
                import requests
                r = requests.get(video_out_url, timeout=120)
                if r.status_code == 200:
                    out_path = f"outputs/test_result_{tag}.mp4"
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    print(f"[OK] Video baixado e salvo em: {out_path}")
            else:
                print("[AVR] Formato de output nao reconhecido ou vazio:")
                print(json.dumps(output, ensure_ascii=True, indent=2)[:500])
            break

        elif status == "FAILED":
            error = status_data.get("error", "Erro desconhecido")
            error_str = str(error).encode("ascii", "ignore").decode("ascii")
            print(f"\n\n[FALHA] Job falhou: {error_str[:400]}")
            print("\nDica: Verifique os logs do worker no painel RunPod:")
            print(f"  https://www.runpod.io/console/serverless/{local_settings.RUNPOD_ENDPOINT_ID}/jobs")
            break

        await asyncio.sleep(15)

    if os.path.exists(temp_vid) and temp_vid != VID_PATH:
        os.remove(temp_vid)

    print("\n" + "="*60)
    print("  TESTE CONCLUIDO")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
