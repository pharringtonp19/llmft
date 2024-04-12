Connect to remote computer
  - Run this command from the command line and inside your vscode editor
```
ssh -i ~/.ssh/`key'.pem ubuntu@12345678910
```

Download script
```
wget -O setup_env.sh https://raw.githubusercontent.com/pharringtonp19/llmft/main/initialization/setup_env.sh
```

Make script executable
```
chmod +x setup_env.sh
```

Run script
```
./setup_env.sh
```
