Connect to remote computer
```
ssh -i ~/.ssh/`key'.pem ubuntu@12345678910
```

Clone Repository
```
git clone https://github.com/pharringtonp19/llmft.git
```

Download script
```
wget -O setup_env.sh https://raw.githubusercontent.com/pharringtonp19/llmft/main/getting-started/setup_env.sh
```

Make script executable
```
chmod +x setup_env.sh
```

Run script
```
./setup_env.sh
```

Open directory
```
cd llmft
```

Activate Virtual Environment
```
source llms/bin/activate
```

Install Library in editable mode
```
pip install -e .
```
