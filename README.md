# Model Credit - Proyecto Credit Score (ANN)


Estructura principal:

- `data/`         : datos crudos, procesados y archivos de scoring
- `models/`       : modelo entrenado (`best_model.pkl`)
- `src/`          : scripts ejecutables (`make_dataset.py`, `train.py`, `evaluate.py`, `predict.py`)
- `notebooks/`    : trabajo exploratorio y documentación técnica
- `.github/`      : workflows de CI

La columna objetivo es **`Credit_Score`** con valores: `Poor`, `Standard`, `Good`.

## Ejecución de Tests Funcionales del Modelo de Crédito

### Paso 0: Ingrese al Escritorio remoto

### Paso 1: Fork del Repositorio Original 

En el navegador, inicie sesión en Github. Luego, suba este proyecto como
un repositorio llamado, por ejemplo, `model-credit`. 

### Paso 2: Levantar el contenedor de Python

```bash
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```

### Paso 3: Configurar git

```bash
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```

### Paso 4: Clonar el Proyecto desde su propio Github

```bash
git clone https://github.com/<USER>/CREDIT-SCORE.git
cd model-credit/
```

### Paso 5: Instalar los pre-requisitos

```bash
pip install -r requirements.txt
```

### Paso 6: Verificar datos crudos


- `data/raw/Score.csv` 

Si se reemplaza por una nueva versión, debe conservar al menos la columna
objetivo `Credit_Score` y las demás columnas usadas en el notebook.

### Paso 7: Ejecutar las pruebas en el entorno

```bash
cd src

python train.py             # entrena el modelo y guarda best_model.pkl
python evaluate.py          # calcula métricas sobre credit_val.csv
python predict.py           # genera data/scores/final_score.csv

cd ..
```

### Paso 8: Guardar los cambios en el Repo

```bash
git add .
git commit -m "Pruebas Finalizadas"
git push
```

