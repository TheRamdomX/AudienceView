# AudienceView

API de recomendaciones multi-dominio (películas, música, conciertos, merch, teatro) con generación de justificaciones vía OpenAI.

## Ejecutar en HTTPS

La aplicación usa **FastAPI + Uvicorn**. Para levantarla con HTTPS necesitas un certificado y clave privada.

### 1. Crear certificado auto-firmado (desarrollo)

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout key.pem -out cert.pem \
  -subj "/C=ES/ST=State/L=City/O=AudienceView/OU=Dev/CN=localhost"
```

Esto genera `cert.pem` y `key.pem` en el directorio actual.

### 2. Definir variables de entorno

Crea/edita `.env` en la raíz del proyecto:

```
SSL_CERTFILE=./cert.pem
SSL_KEYFILE=./key.pem
HOST=0.0.0.0
PORT=8443
RELOAD=true
OPENAI_API_KEY=tu_clave
OPENAI_MODEL=gpt-4o-mini
```

> Asegúrate de tener la clave de OpenAI para generar recomendaciones justificadas.

### 3. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Ejecutar servidor

```bash
python api.py
```

Verás algo como: `Iniciando API en https://0.0.0.0:8443 …`

Ahora puedes probar:

```bash
curl -k https://localhost:8443/health
```

Usa `-k` (insecure) porque es un certificado auto-firmado.

### Endpoints principales

- `GET /health` estado
- `POST /recommendations/json` body `{ "liked_titles": ["Inception"], "top_n": 3 }`
- `POST /recommendations/text`
- `GET /recommendations/json?liked_titles=Inception,Matrix`
- `GET /recommendations/text?liked_titles=Inception,Matrix`

### Notas de producción

En producción se recomienda usar un reverse proxy (Nginx, Traefik, Caddy) que termine TLS y ejecutar Uvicorn sin SSL interno:

```
uvicorn api:app --host 0.0.0.0 --port 8000
```

Luego configurar certificados válidos en el proxy.

## Modo CLI

`main.py` permite obtener recomendaciones justificados por LLM en modo texto o JSON:

```bash
python main.py --text
python main.py --json
```

## Licencia

Proyecto interno / MVP.
