import aiohttp
import asyncio
import json
import logging
import os
import base64
from typing import Dict, Any, Optional
from pathlib import Path
from .config import local_settings
from .storage_service import storage_service

class StaticResolver(aiohttp.abc.AbstractResolver):
    """
    Ignora o DNS do sistema e usa IPs hardcoded para dominios criticos.
    Evita o erro [Could not contact DNS servers] no ambiente do usuario.
    """
    async def resolve(self, host, port=0, family=0):
        if "api.runpod.ai" in host:
            # IPs reais da Cloudflare para api.runpod.ai
            return [{'hostname': host, 'host': '104.18.9.221', 'port': port, 'family': family, 'proto': 0, 'flags': 0}]
        return [{'hostname': host, 'host': host, 'port': port, 'family': family, 'proto': 0, 'flags': 0}]
    async def close(self): pass

class RunPodServerlessService:
    """
    Service para interagir com o RunPod Serverless do Kinomuse.
    """
    def __init__(self):
        self.api_key = local_settings.RUNPOD_API_KEY
        self.endpoint_id = local_settings.RUNPOD_ENDPOINT_ID
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger("RunPodServerless")

    def _log(self, message: str):
        print(f"[*] [SERVERLESS] {message}")
        self.logger.info(message)

    def _file_to_base64(self, file_path: str) -> str:
        """Converte um arquivo local para base64 para envio direto no payload"""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def prepare_inputs(self, char_image_path: str, motion_video_path: str, job_tag: str):
        """
        Faz upload de ambos os inputs locais para o Supabase Storage.
        """
        self._log("Fazendo upload de inputs locais para o Supabase Storage...")
        loop = asyncio.get_event_loop()
        
        image_url = await loop.run_in_executor(None, storage_service.upload_file, char_image_path, "inputs")
        video_url = await loop.run_in_executor(None, storage_service.upload_file, motion_video_path, "inputs")
        
        return image_url, video_url

    async def run_inference(
        self,
        image_url: str,
        video_url: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        fps: int = 16,
        frames: int = 48,
        length: int = 48,
        video_length: int = 48,
        steps: int = 20,
        cfg: float = 5.0,
        seed: int = 42,
        **kwargs
    ) -> Optional[str]:
        """
        Envia job ao endpoint RunPod Serverless usando o template WanVideoWrapper
        com WanVideoAnimateEmbeds + WanVideoSampler (corrigido: pose resize e
        resolucao alinhada, sem BlockSwap).
        """
        self._log("Disparando run_inference com WanVideoWrapper Animate (corrigido)...")

        url = f"{self.base_url}/run"
        connector = aiohttp.TCPConnector(resolver=StaticResolver())
        
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async def get_base64(file_url_or_path: str) -> Optional[str]:
                    if not file_url_or_path:
                        return None
                    if file_url_or_path.startswith("http"):
                        self._log(f"Baixando arquivo para converter em base64: {file_url_or_path[:60]}...")
                        async with session.get(file_url_or_path) as r:
                            if r.status == 200:
                                content = await r.read()
                                return base64.b64encode(content).decode("utf-8")
                    elif os.path.exists(file_url_or_path):
                        self._log(f"Lendo e codificando arquivo local em base64: {file_url_or_path}...")
                        with open(file_url_or_path, "rb") as f:
                            return base64.b64encode(f.read()).decode("utf-8")
                    return None

                local_image_path = kwargs.get("local_image_path")
                local_video_path = kwargs.get("local_video_path")

                img_b64 = await get_base64(local_image_path or image_url)
                vid_b64 = await get_base64(local_video_path or video_url)

                if not img_b64:
                    self._log("Erro: nao foi possivel obter base64 para a imagem!")
                    return None

                if not vid_b64:
                    self._log("Erro: nao foi possivel obter base64 para o video!")
                    return None

                # 1. Seleciona e carrega o template (v1 = DWPreprocessor, v2 = WanAnimatePreprocess)
                template_version = kwargs.get("template_version", "v2")
                if template_version == "v2":
                    template_name = "wan_2.2_animate_v2_template.json"
                else:
                    template_name = "wan_2.2_animate_template.json"

                template_path = os.path.join(os.path.dirname(__file__), template_name)
                if not os.path.exists(template_path):
                    self._log(f"Erro: template {template_name} nao encontrado em {template_path}")
                    return None
                    
                with open(template_path, "r", encoding="utf-8") as f:
                    workflow = json.load(f)

                # 2. Injeta prompts no WanVideoTextEncodeCached (node 65)
                if "65" in workflow:
                    workflow["65"]["inputs"]["positive_prompt"] = prompt
                    workflow["65"]["inputs"]["negative_prompt"] = negative_prompt

                # 3. Injeta largura/altura no VHS_LoadVideo (custom_width/custom_height)
                if "63" in workflow:
                    workflow["63"]["inputs"]["custom_width"] = int(width)
                    workflow["63"]["inputs"]["custom_height"] = int(height)

                # 4. Injeta resolucao nos nos que precisam (62, 64 e nodes de pose)
                nodes_to_inject = ["62", "64"]
                if template_version == "v1":
                    nodes_to_inject.append("74")  # ImageResizeKJv2 for DWPreprocessor output
                else:
                    nodes_to_inject.extend(["76", "77"])  # PoseAndFaceDetection + DrawViTPose

                for node_id in nodes_to_inject:
                    if node_id not in workflow:
                        continue
                    inp = workflow[node_id]["inputs"]
                    if isinstance(inp.get("width"), (int, float)):
                        inp["width"] = int(width)
                    if isinstance(inp.get("height"), (int, float)):
                        inp["height"] = int(height)

                # 4. Injeta parametros de geracao no WanVideoSampler (node 27)
                if "27" in workflow:
                    workflow["27"]["inputs"]["seed"] = int(seed)
                    workflow["27"]["inputs"]["steps"] = int(steps)
                    workflow["27"]["inputs"]["cfg"] = float(cfg)
                    workflow["27"]["inputs"]["shift"] = float(kwargs.get("shift", 1.0))

                # 5. Injeta num_frames e pose/face_strength no WanVideoAnimateEmbeds (node 62)
                if "62" in workflow:
                    workflow["62"]["inputs"]["num_frames"] = int(video_length)
                    pose_strength = kwargs.get("pose_strength")
                    if pose_strength is not None:
                        workflow["62"]["inputs"]["pose_strength"] = float(pose_strength)
                    face_strength = kwargs.get("face_strength")
                    if face_strength is not None:
                        workflow["62"]["inputs"]["face_strength"] = float(face_strength)

                # 6. Ajusta tiled_vae para v2 se nao presente (node 62)
                if "62" in workflow and "tiled_vae" not in workflow["62"]["inputs"]:
                    workflow["62"]["inputs"]["tiled_vae"] = False
                if "62" in workflow and "vae_tile_size" in workflow["62"]["inputs"]:
                    del workflow["62"]["inputs"]["vae_tile_size"]

                # 7. Adiciona SaveAnimatedWEBP nativo como fallback de output
                workflow["999"] = {
                    "inputs": {
                        "filename_prefix": "Kinomuse_Output",
                        "fps": float(fps),
                        "lossless": False,
                        "quality": 85,
                        "method": "default",
                        "images": ["28", 0]
                    },
                    "class_type": "SaveAnimatedWEBP"
                }

                # 7. Monta o input_data com o workflow e imagens
                input_data = {
                    "workflow": workflow,
                    "images": [
                        {
                            "name": "refer.jpeg",
                            "image": f"data:image/jpeg;base64,{img_b64}"
                        },
                        {
                            "name": "raw.mp4",
                            "image": f"data:video/mp4;base64,{vid_b64}"
                        }
                    ],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": int(seed),
                    "width": int(width),
                    "height": int(height),
                    "fps": int(fps),
                    "steps": int(steps),
                    "cfg": float(cfg),
                }

                payload = {"input": input_data}

                self._log("Enviando payload com WanVideoWrapper Animate para o RunPod...")
                async with session.post(url, json=payload, headers=self.headers, ssl=False) as response:
                    if response.status not in [200, 201]:
                        res_text = await response.text()
                        self._log(f"Erro do endpoint RunPod status={response.status}: {res_text}")
                        return None
                    
                    res_data = await response.json()
                    job_id = res_data.get("id")
                    self._log(f"Job enviado com sucesso! Job ID: {job_id}")
                    return job_id
        except Exception as e:
            self._log(f"Excecao ao disparar inference: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    async def check_status(self, job_id: str) -> Dict[str, Any]:
        """
        Checks the status of a RunPod Serverless job.
        """
        url = f"{self.base_url}/status/{job_id}"
        connector = aiohttp.TCPConnector(resolver=StaticResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.get(url, headers=self.headers, ssl=False) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        text = await response.text()
                        return {"status": "error", "message": f"HTTP {response.status}: {text}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

    async def poll_until_complete(self, job_id: str, timeout: int = 1800, interval: int = 10) -> Optional[Dict[str, Any]]:
        """
        Polls the job status until it's 'COMPLETED' or 'FAILED'.
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            status_data = await self.check_status(job_id)
            status = status_data.get("status")
            
            if status == "COMPLETED":
                self._log(f"[DONE] Job {job_id} concluido!")
                return status_data
            elif status == "FAILED":
                self._log(f"[FAIL] Job {job_id} falhou: {status_data.get('error')}")
                return status_data
            elif status == "IN_QUEUE":
                self._log(f"[WAIT] Job {job_id} na fila...")
            elif status == "IN_PROGRESS":
                self._log(f"[RUN] Job {job_id} em processamento...")
            
            await asyncio.sleep(interval)
        
        self._log(f"[TIME] Timeout: Job {job_id} nao terminou em {timeout}s.")
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancels a running job.
        """
        url = f"{self.base_url}/cancel/{job_id}"
        connector = aiohttp.TCPConnector(resolver=StaticResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.post(url, headers=self.headers, ssl=False) as response:
                    return response.status == 200
            except Exception:
                return False

# Singleton instance
serverless_service = RunPodServerlessService()
