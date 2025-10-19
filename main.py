# main.py
import os
import io
import re
import datetime
import asyncio
import cv2
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from geopy.distance import geodesic
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from ultralytics import YOLO
import zipfile

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
TASK_INTERVAL_SECONDS = int(os.getenv('CRON_INTERVAL_SECONDS', 300))
IMGSZ = int(os.getenv('YOLO_IMGSZ', 640))
CONF = float(os.getenv('YOLO_CONF', 0.25))

if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
    raise EnvironmentError("One or more Supabase environment variables are missing.")

# Initialize Supabase client
supabase: Client = None
YOLO_MODEL: YOLO = None
task_scheduler: asyncio.Task = None
DATE_PATTERN = re.compile(r'(\d{8}|\d{4}[-._/]\d{2}[-._/]\d{2})_', re.IGNORECASE)

# Raio máximo de agrupamento (em metros)
RAIO_MAX_METROS = 30

def agrupar_por_raio(pontos):
    grupos = []
    for ponto in pontos:
        coordenada = (ponto["lat"], ponto["long"])
        adicionado = False

        # tenta adicionar o ponto a um grupo existente
        for grupo in grupos:
            centro = grupo["centro"]
            distancia = geodesic(coordenada, centro).meters
            if distancia <= RAIO_MAX_METROS:
                grupo["pontos"].append(ponto)
                adicionado = True
                break

        # se não couber em nenhum grupo, cria um novo
        if not adicionado:
            grupos.append({
                "centro": coordenada,
                "pontos": [ponto],
            })
    return grupos


def contar_por_tipo(grupos):
    estatisticas = []
    for grupo in grupos:
        contagem_tipos = defaultdict(int)
        datas = []

        for ponto in grupo["pontos"]:
            contagem_tipos[ponto["tipo"]] += 1
            datas.append(datetime.fromisoformat(ponto["data"].replace("Z", "+00:00")))

        estatisticas.append({
            "centro": grupo["centro"],
            "total_pontos": len(grupo["pontos"]),
            "por_tipo": dict(contagem_tipos),
            "datas": sorted(set(d.date() for d in datas))
        })
    return estatisticas

def dms_to_dd(dms_tuple, ref):
    """
    Converte coordenadas DMS (Graus, Minutos, Segundos) para Graus Decimais (DD).
    
    Args:
        dms_tuple (tuple): Tupla (graus, minutos, segundos).
        ref (str): Referência de direção ('N', 'S', 'E', 'W').

    Returns:
        float: Coordenada em Graus Decimais.
    """
    degrees = dms_tuple[0]
    minutes = dms_tuple[1]
    seconds = dms_tuple[2]
    
    # Fórmula: DD = Graus + (Minutos / 60) + (Segundos / 3600)
    dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    # Aplica o sinal negativo para Sul ou Oeste
    if ref in ('S', 'W'):
        dd = -dd
        
    # Arredonda para 6 casas decimais para precisão padrão de GPS
    return round(dd, 6)

async def process_images_async():
    """
    Função principal que pega arquivos do storage, processa com YOLO e salva os resultados.
    """
    global supabase, YOLO_MODEL, SUPABASE_BUCKET, IMGSZ, CONF

    if not supabase or not YOLO_MODEL:
        print("[ALERTA] Recursos não inicializados. Pulando ciclo de cron.")
        return
    
    print(f"\n[INFO] Iniciando ciclo de processamento em {datetime.datetime.now().isoformat()}")

    try:
        files_list = await asyncio.to_thread(
            supabase.storage.from_(SUPABASE_BUCKET).list
        )
    except Exception as e:
        print(f"[ERRO] Erro ao listar arquivos do bucket: '{SUPABASE_BUCKET}': {e}")
        return

    for f in files_list:
        file_name = f["name"]
        
        if file_name.startswith('.') or DATE_PATTERN.search(file_name):
            continue

        base_name, file_ext = file_name.rsplit('.', 1) if '.' in file_name else (file_name, '')
        
        # --- INICIALIZAÇÃO DAS VARIÁVEIS (CORREÇÃO 1) ---
        latitude_dd = None
        longitude_dd = None
        # Data de referência segura (Postgres-friendly) se o EXIF falhar
        data_info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # 1. Baixar o arquivo original em memória (assíncrono)
            print(f"Baixando: {file_name}")
            file_obj = await asyncio.to_thread(
                supabase.storage.from_(SUPABASE_BUCKET).download, file_name
            )
            
            # Use BytesIO para manipular o objeto de bytes e rebobinar para o YOLO
            img_bytes_io = io.BytesIO(file_obj)

            # 2. Extrair metadados EXIF
            try:
                # Abrir com PIL/Pillow para EXIF
                image = Image.open(img_bytes_io) 
                
                exif_data = image._getexif()
                if exif_data:
                    exif_dict = {TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data.items()}
                    
                    # Extração e formatação da data (CORREÇÃO para Postgres)
                    original_date = exif_dict.get('DateTimeOriginal')
                    if original_date:
                        data_info = original_date.replace(':', '-', 2)

                    # Extração e conversão de GPS
                    gps_info = exif_dict.get('GPSInfo')
                    if gps_info:
                        lat_dms = gps_info.get(2)
                        lat_ref = gps_info.get(1)
                        lon_dms = gps_info.get(4)
                        lon_ref = gps_info.get(3)
                        
                        if lat_dms and lat_ref and lon_dms and lon_ref:
                            latitude_dd = dms_to_dd(lat_dms, lat_ref)
                            longitude_dd = dms_to_dd(lon_dms, lon_ref)
                            print(f"GPS Convertido: {latitude_dd} / {longitude_dd}")
                        else:
                            print("Dados GPS incompletos.")
                    else:
                        print("Informação 'GPSInfo' não encontrada.")
                else:
                    print("Nenhum metadado EXIF encontrado.")
            except Exception as e:
                # Apenas avisa e continua, usando os defaults
                print(f"Aviso: Erro ao processar EXIF: {e}") 
                pass

            # 3. EXECUTAR PREDIÇÃO YOLO (OTIMIZAÇÃO: Sem np/cv2 para leitura)
            print("Executando predição YOLO...")

            results = await asyncio.to_thread(
                YOLO_MODEL.predict,
                source=image, # <--- Passa o objeto PIL.Image diretamente
                imgsz=IMGSZ,
                conf=CONF,
                save=False, show=False, device='cpu'
            )

            detected_classes = []
            
            if results and results[0].boxes:
                r = results[0] 
                
                # Extração de classes (CORREÇÃO 3: Adiciona nomes, não IDs)
                class_names = r.names
                class_ids = r.boxes.cls.tolist()
                
                unique_class_names = set()
                for class_id_float in class_ids:
                    class_id = int(class_id_float)
                    class_name = class_names.get(class_id, "Unknown")
                    unique_class_names.add(class_name)
                    
                detected_classes = list(unique_class_names)
                print(f"Classes Detectadas: {detected_classes}")

                # --- NOME CLASSIFICADO (Com H/M/S para unicidade) ---
                current_timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 
                new_original_file_name = f"{base_name}_{current_timestamp_str}.{file_ext}"
                yolo_output_file_name = f"{base_name}_{current_timestamp_str}_YOLO.png"

                # 4. RENOMEAR O ARQUIVO ORIGINAL NO STORAGE (assíncrono)
                print(f"Renomeando ORIGINAL: {file_name} -> {new_original_file_name}")
                await asyncio.to_thread(
                    supabase.storage.from_(SUPABASE_BUCKET).move, file_name, new_original_file_name
                )
                
                # 5. SALVAR IMAGEM ANOTADA (OUTPUT YOLO)
                yolo_annotated_image_np = r.plot() # Retorna um array numpy/cv2
                
                # Certifique-se de que a variável 'cv2' esteja importada para usar imencode
                success, encoded_image = await asyncio.to_thread(
                    cv2.imencode, '.png', yolo_annotated_image_np
                )
                
                if not success:
                    raise ValueError("Falha ao codificar imagem anotada pelo YOLO para PNG.")
                    
                yolo_output_bytes = encoded_image.tobytes()
                
                # 6. SALVAR IMAGEM ANOTADA (YOLO) NO STORAGE (assíncrono)
                print(f"Salvando YOLO Output: {yolo_output_file_name}")
                await asyncio.to_thread(
                    supabase.storage.from_(SUPABASE_BUCKET).upload,
                    yolo_output_file_name, 
                    yolo_output_bytes,
                    file_options={"content-type": "image/png"}
                )

                # 7. INSERIR DADOS NO BANCO DE DADOS
                new_row = {
                    'lat': latitude_dd,
                    'long': longitude_dd,
                    'data': data_info,
                    'tipo': detected_classes,
                    'url_image': supabase.storage.from_(SUPABASE_BUCKET).get_public_url(yolo_output_file_name)
                }
                
                print(f"Inserindo dados no DB: {new_original_file_name}")
                # A inserção é síncrona, então usamos to_thread (como já está)
                await asyncio.to_thread(
                    supabase.table('imagens').insert(new_row).execute
                )
            else:
                print(f"Nenhuma detecção encontrada para {file_name}.")

        except Exception as e:
            print(f"Erro no processamento do arquivo {file_name}: {e}")


# --- 3. FUNÇÃO DE AGENDAMENTO (Para rodar a cada X segundos) ---
async def periodic_task(interval_seconds: int):
    """Roda process_images_async em um loop infinito com um intervalo."""
    while True:
        await process_images_async()
        await asyncio.sleep(interval_seconds)

# --- 4. CONTEXTO LIFESPAN (Inicialização e Fechamento) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, YOLO_MODEL, task_scheduler

    print("[LIFESPAN] Iniciando aplicação e carregando recursos...")

    # A. Inicialização Supabase
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[LIFESPAN] Cliente Supabase criado.")
    except Exception as e:
        print(f"[ERRO LIFESPAN] Falha ao criar cliente Supabase: {e}")
        # É um erro fatal
        raise e
    
    # B. Inicialização YOLO
    try:
        # Carregar o modelo YOLO é uma operação síncrona, fazemos no thread principal.
        YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        print(f'[LIFESPAN] Modelo YOLO ("{YOLO_MODEL_PATH}") carregado.')
    except Exception as e:
        print(f"[ERRO LIFESPAN] Falha ao carregar modelo YOLO: {e}")
        # É um erro fatal
        raise e
    
    # C. Inicia o agendador (cron job)
    task_scheduler = asyncio.create_task(periodic_task(TASK_INTERVAL_SECONDS))
    print(f"[LIFESPAN] Tarefa de processamento agendada a cada {TASK_INTERVAL_SECONDS} segundos.")

    # Inicia o servidor (FastAPI)
    yield

    # --- FECHAMENTO (Shutdown) ---
    print("[LIFESPAN] Encerrando aplicação e fechando recursos...")

    # Cancela a tarefa agendada
    if task_scheduler:
        task_scheduler.cancel()
        print("[LIFESPAN] Tarefa de processamento cancelada.")
    
    # Não há recursos de DB explícitos para fechar, mas é bom ter o modelo.

# --- 5. INICIALIZAÇÃO DO FASTAPI ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def raiz():
    return {"mensagem": "API EcoDrone funcionando!"}

@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um .zip")

    try:
        # Lê o arquivo ZIP em memória
        zip_bytes = await file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))

        uploaded_files = []

        # Itera sobre os arquivos do ZIP
        for info in zip_file.infolist():
            if info.is_dir():
                continue

            file_data = zip_file.read(info.filename)
            file_name = info.filename.split("/")[-1]  # remove caminhos internos do ZIP

            # Faz upload para o Supabase
            path_on_bucket = f"{file_name}"
            storage_res = supabase.storage.from_("imagens").upload(
                path_on_bucket,
                file_data
            )
            uploaded_files.append(path_on_bucket)

        return {"uploaded_files": uploaded_files}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Arquivo ZIP inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pontos")
def listar_deteccoes():
    """Retorna todas as detecções registradas."""
    results = supabase.table('imagens').select('*').execute()
    return results

@app.get("/stats")
def gerar_estatisticas():
    """Retorna todas as detecções registradas."""
    stats = {}
    # Busca todas as imagens na tabela 'imagens'
    response = supabase.table("imagens").select("*", count="exact").execute()
    imagens = response.data

    # Lista para armazenar os resultados processados
    pontos = []

    for imagem in imagens:
        tipos = imagem.get("tipo", [])
        if not tipos:
            continue  # pula se não houver tipos
        for tipo in tipos:
            if tipo == 'Unknown':
                continue
            imagem_copia = imagem.copy()
            imagem_copia["tipo"] = tipo
            pontos.append(imagem_copia)

    # ---- execução ----
    grupos = agrupar_por_raio(pontos)
    estatisticas = contar_por_tipo(grupos)

    # ordena por número total de pontos
    estatisticas.sort(key=lambda x: x["total_pontos"], reverse=True)

    # imprime resultados
    for i, regiao in enumerate(estatisticas, start=1):
        print(f"Região #{i}")
        print(f"  Centro aproximado: {regiao['centro']}")
        print(f"  Total de ocorrências: {regiao['total_pontos']}")
        print(f"  Por tipo: {regiao['por_tipo']}")
        print(f"  Datas envolvidas: {regiao['datas']}")
        print()

    
    print(estatisticas)
    
    return estatisticas
    