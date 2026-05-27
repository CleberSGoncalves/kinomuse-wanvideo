"""
Script de validacao local do workflow transformado.
Simula o que o handler faz e verifica se restam referencias a nodes removidos.
"""
import json
import sys

with open('services/wan_2.1_template.json', 'r') as f:
    workflow = json.load(f)

# Simular os parametros
width = 512
height = 512
video_length = 48
seed = 42
steps = 20
cfg = 5.0
kwargs = {"shift": 5.0, "scheduler": "unipc", "sampler": "dpm++_sde"}

# Passo 1: Node 65
if "65" in workflow:
    workflow["65"]["inputs"]["positive_prompt"] = "test positive"
    workflow["65"]["inputs"]["negative_prompt"] = "test negative"

# Passo 2: Node 27
if "27" in workflow:
    workflow["27"]["inputs"]["seed"] = int(seed)
    workflow["27"]["inputs"]["steps"] = int(steps)
    workflow["27"]["inputs"]["cfg"] = float(cfg)
    workflow["27"]["inputs"]["shift"] = float(kwargs.get("shift", 5.0))
    workflow["27"]["inputs"]["scheduler"] = str(kwargs.get("scheduler", "unipc"))
    workflow["27"]["inputs"]["sampler"] = str(kwargs.get("sampler", "dpm++_sde"))

# Passo 3: Fixar DWPreprocessor ANTES de remover 152
if "73" in workflow:
    inp73 = workflow["73"]["inputs"]
    if isinstance(inp73.get("resolution"), list):
        print(f"[OK] Node 73: resolution era lista {inp73['resolution']}, fixando para 512")
        inp73["resolution"] = 512
    else:
        inp73["resolution"] = inp73.get("resolution", 512)

# Passo 4: Fixar width/height ANTES de remover 150/151
for node_id in ["62", "63", "64"]:
    if node_id not in workflow:
        continue
    inp = workflow[node_id]["inputs"]
    if isinstance(inp.get("width"), list):
        print(f"[OK] Node {node_id}: width era lista {inp['width']}, fixando para {width}")
        inp["width"] = int(width)
    if isinstance(inp.get("height"), list):
        print(f"[OK] Node {node_id}: height era lista {inp['height']}, fixando para {height}")
        inp["height"] = int(height)
    if isinstance(inp.get("custom_width"), list):
        print(f"[OK] Node {node_id}: custom_width era lista {inp['custom_width']}, fixando para {width}")
        inp["custom_width"] = int(width)
    if isinstance(inp.get("custom_height"), list):
        print(f"[OK] Node {node_id}: custom_height era lista {inp['custom_height']}, fixando para {height}")
        inp["custom_height"] = int(height)
    if node_id == "62" and isinstance(inp.get("num_frames"), list):
        print(f"[OK] Node {node_id}: num_frames era lista {inp['num_frames']}, fixando para {video_length}")
        inp["num_frames"] = int(video_length)

# Passo 5: Remover mascara
if "62" in workflow:
    for opt in ["face_images", "bg_images", "mask"]:
        if opt in workflow["62"]["inputs"]:
            del workflow["62"]["inputs"][opt]

# Passo 6: Desativar BlockSwap (nodes 50 e 51)
if "27" in workflow:
    workflow["27"]["inputs"]["model"] = ["22", 0]

# Passo 7: Remover nodes desnecessarios
nodes_to_remove = {
    "96", "99", "100", "102", "104",
    "107", "108", "120",
    "110", "171",
    "150", "151",
    "42", "75", "77", "112", "152",
    "50", "51", # Remover BlockSwap
}
for n in nodes_to_remove:
    workflow.pop(n, None)

if "48" in workflow:
    workflow.pop("48")
if "22" in workflow:
    workflow["22"]["inputs"].pop("compile_args", None)
    if "35" in workflow:
        workflow.pop("35", None)

# VALIDACAO: verificar se algum node ainda referencia nodes removidos
print("\n=== VALIDACAO: Referencias pendentes ===")
all_removed = nodes_to_remove | {"48", "35"}
nodes_in_workflow = set(workflow.keys())
errors = []

for nid, node in workflow.items():
    inputs = node.get("inputs", {})
    for key, val in inputs.items():
        if isinstance(val, list) and len(val) >= 2:
            ref = str(val[0])
            if ref not in nodes_in_workflow:
                errors.append(f"Node {nid} ({node.get('class_type')}).{key} -> Node {ref} (NAO ENCONTRADO!)")

if errors:
    print("[ERRO] Referencias invalidas encontradas:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("[OK] Nenhuma referencia invalida encontrada!")

print("\n=== Nodes restantes no workflow ===")
for nid in sorted(workflow.keys(), key=lambda x: int(x) if x.isdigit() else 999):
    ct = workflow[nid].get("class_type", "??")
    print(f"  Node {nid}: {ct}")

print("\n[SUCESSO] Workflow valido para execucao no ComfyUI!")
