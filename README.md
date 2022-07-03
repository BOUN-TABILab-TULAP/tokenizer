## How to run using Docker
1. Launch a terminal in the root directory of the repo and build the docker image where
- `-t` is the tag for the docker image. You can provide any name you want
- `.` is the relative path to the Dockerfile 
```bash
docker build -t tokenizer .
```
2. Run the docker image where
- `-d` indicates "detach", let the container run in the background
- `-p 5000:5000` indicates mapping port 5000 of the container to the port 5000 of the host.
```bash
docker run -d -p 5000:5000 tokenizer
```
3. Send a POST request
- via curl
    ```bash
    curl -X POST http://localhost:5000/evaluate 
   -H 'Content-Type: application/json' 
   -d '{"text":"Merhaba, ben okula gidiyorum"}'
    ```
- via Python's requests library
    ```python
    import requests
    res = requests.post('http://localhost:5000/evaluate', json={'text':'Merhaba, ben okula gidiyorum'})
    print(res.json())
    ```