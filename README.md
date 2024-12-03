


# YOLOV8 Worker Implementation


1) clone the repository

```
    $ git clone https://github.com/kathulhur/yolov8-worker.git
```

2) Create a virtual environment and activate

``` bash
    $ python -m venv env
    $ env/Scipts/activate # for windows
    $ source env/bin/activate # for linux
```

3) Install requirements

``` bash
    $ pip install -r requirements.txt
```

-- This part requires RabbitMQ installation

4) set the environment variables (rabbit MQ variables)
``` bash
    HOST=localhost
    VIRTUAL_HOST=inference-system-dev
    MODEL_TAG=yolov9
    USERNAME=guest
    PASSWORD=guest
```

5) start inference worker
``` bash
    $ inference-worker start
```