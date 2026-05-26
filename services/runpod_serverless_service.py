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
        Envia job ao endpoint RunPod Serverless usando o template de workflow JSON.
        Faz patches dinamicos em memoria no template e envia base64 das imagens.
        """
        self._log("Disparando run_inference com injeção de workflow e payload base64...")

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

                # 1. Carrega o template
                template_path = os.path.join(os.path.dirname(__file__), "wan_2.1_template.json")
                if not os.path.exists(template_path):
                    self._log(f"Erro: template nao encontrado em {template_path}")
                    return None
                    
                with open(template_path, "r", encoding="utf-8") as f:
                    workflow = json.load(f)

                # 2. Injeta as variaveis basicas
                if "65" in workflow:
                    workflow["65"]["inputs"]["positive_prompt"] = prompt
                    workflow["65"]["inputs"]["negative_prompt"] = negative_prompt
                if "27" in workflow:
                    workflow["27"]["inputs"]["seed"] = int(seed)
                    workflow["27"]["inputs"]["steps"] = int(steps)
                    workflow["27"]["inputs"]["cfg"] = float(cfg)
                    workflow["27"]["inputs"]["shift"] = float(kwargs.get("shift", 5.0))
                    # Usar os schedulers/samplers válidos do template WanVideoSampler
                    workflow["27"]["inputs"]["scheduler"] = str(kwargs.get("scheduler", "unipc"))
                    workflow["27"]["inputs"]["sampler"] = str(kwargs.get("sampler", "dpm++_sde"))
                    # Critico: WanVideoSampler versoes novas requerem start_step/end_step
                    # Se ausentes, ficam None e causam TypeError na comparacao '>' com int
                    workflow["27"]["inputs"]["start_step"] = 0
                    workflow["27"]["inputs"]["end_step"] = -1
                if "73" in workflow:
                    workflow["73"]["inputs"]["resolution"] = 512

                # 3. Injeta resolucoes substituindo referencias do INTConstant
                for node_id in ["62", "63", "64"]:
                    if node_id not in workflow:
                        continue
                    inp = workflow[node_id]["inputs"]
                    if inp.get("width") == ["150", 0]:
                        inp["width"] = width
                    if inp.get("height") == ["151", 0]:
                        inp["height"] = height
                    if inp.get("custom_width") == ["150", 0]:
                        inp["custom_width"] = width
                    if inp.get("custom_height") == ["151", 0]:
                        inp["custom_height"] = height

                # 4. Limpa conexoes de mascara
                if "62" in workflow:
                    for opt in ["face_images", "bg_images", "mask"]:
                        if opt in workflow["62"]["inputs"]:
                            workflow["62"]["inputs"][opt] = None

                # 5. Limpa nos nao utilizados no serverless
                nodes_to_remove = {
                    "96", "99", "100", "102", "104",
                    "107", "108", "120",
                    "110", "171", "35",
                    "150", "151",
                    "42", "75", "77", "112", "152",
                }
                for n in nodes_to_remove:
                    workflow.pop(n, None)

                if "48" in workflow:
                    workflow.pop("48")
                if "50" in workflow:
                    workflow["50"]["inputs"]["model"] = ["22", 0]
                if "22" in workflow:
                    workflow["22"]["inputs"].pop("compile_args", None)

                # 6. Monta o input_data com o workflow e imagens
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
                    "shift": float(kwargs.get("shift", 5.0)),
                    "scheduler": str(kwargs.get("scheduler", "unipc")),
                    "sampler": str(kwargs.get("sampler", "dpm++_sde")),
                    "denoise_strength": 1.0,
                    "riflex_freq_index": 0,
                    # Critico: evita que o handler injete None no WanVideoSampler
                    "start_step": 0,
                    "end_step": -1
                }

                payload = {"input": input_data}

                self._log("Enviando payload com workflow customizado para o RunPod...")
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
