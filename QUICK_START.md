
```bash
cd Travel-RAG-System
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```pip install faiss-cpu


Ensure `data/egypt_places.json` exists with your dataset, then:

```bash
python init_vector_store.py
```


```bash
python run_api.py
```

Visit: http://localhost:8000/docs for interactive documentation
```bash
docker-compose up -d
docker-compose ps
docker-compose down
```


```bash
curl http://localhost:8000/api/health
```

```bash
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/egypt_places.json", "rebuild": false}'
```
