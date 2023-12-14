# Turn on GPU access with Docker Compose
> docker容器中，简便运行LLM进行推理加速

地址：https://docs.docker.com/compose/gpu-support/


## docker安装nvidia-container-toolkit

- **首先：**
  1. 确保`nvcc -V`可以正常输出
  2. 确保docker安装成功

- **其次：**

&nbsp;&nbsp;&nbsp;&nbsp;在服务器bash中设置stable存储库和密匙：

  ```shell
distribution=$(./etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

- **再次：**

&nbsp;&nbsp;&nbsp;&nbsp;添加NVIDIA Container Toolkit的软件源，并安装nvidia-docker2软件包：

```shell
sudo apt-get udpate
sudo apt-get install -y nvidia-container-toolkit
```

- **接着：**

&nbsp;&nbsp;&nbsp;&nbsp;重启docker：`sudo systemctl restart docker`

- **最后：**

&nbsp;&nbsp;&nbsp;&nbsp;build一个新的容器查看能否使用nvidia gpu进行加速。`docker build -t xllm-stream`


## Docker Compose

> Docker Compose可以一次起很多个docker，一个镜像编排工具。

启动方式：
```shell
sudo docker-compose -f docker-comppse.yml up -d
```

其中`comppse.yml`文件包含如下配置：

- `capabilities`：此值指定为字符串列表（例如，capabilities: [gpu]）。您必须在 Compose 文件中设置此字段。否则，在服务部署时会返回错误。

- `count`：此值作为整数或值 "all" 指定，表示应该保留的 GPU 设备数量（如果主机持有该数量的 GPU）。如果 count 设置为 "all" 或未指定，则默认情况下使用主机上所有可用的 GPU。

- `device_ids`：此值作为字符串列表指定，表示主机上的 GPU 设备 ID。您可以在主机上的 `nvidia-smi` 输出中找到设备 ID。如果未设置 `device_ids`，则默认使用主机上所有可用的 GPU。

- `driver`：此值作为字符串指定，例如 `driver: 'nvidia'`。

- `options`：键值对表示特定于驱动程序的选项。


### Example of a Compose file for running a service with access to 1 GPU device:

```yml
services:
  test:
    image: nvidia/cuda:12.3.1-base-ubuntu20.04
    command: nvidia-smi
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```



### Access specific devices:

在托管多个 GPU 的计算机上，可以设置 `device_ids` 字段以针对特定的 GPU 设备，`count` 可以用于限制分配给服务容器的 GPU 设备的数量。

您可以在每个服务定义中使用 `count` 或 `device_ids`。如果尝试同时组合两者、指定无效的设备 ID 或使用高于系统中 GPU 数量的 `count` 值，则会返回错误。


要仅允许访问 GPU-0 和 GPU-3 设备：

```yml
services:
  test:
    image: tensorflow/tensorflow:latest-gpu
    command: python -c "import tensorflow as tf;tf.test.gpu_device_name()"
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ['0', '3']
            capabilities: [gpu]
```









