# Installation Guideline

## Installing prerequisites
- Install the `uv` package if it does not exist.
``` bash
pip install uv
```
- Install the Ollama package if it does not exist
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

# Running the project
- Install the model if it does not exist
```bash
ollama run mistral
```

- Create env
``` bash
uv venv  
```

- Activate the env
``` bash
source ./venv/bin/active
```

- Install the necessary packages
``` bash
pip install -r requirements.txt 
```

- Run the file
``` bash
python main.py
```
