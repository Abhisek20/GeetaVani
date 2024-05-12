## GeetaVani

This project is intended to create a chat application on top of open soourced LLM models like llama3, phi3 and etc. based on bhagwad geeta, people can ask generic questions about life and application will be able to give reply from geeta's perpective.

## Setup information
1. Create a ollama image in the docker from official release by uisng command `docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`

2. Build docker image using the docker file using command `docker build . -t geetavani:<version>`
3. Create a multi-container image using the docker-compose.yaml file by using command `docker-compose up`
4. Check the image names in the multi-container image using `docker-compose ps` and  notedown the ollama image name in multicontainer image.
5. Pull the supported models(phi3 and llama3) from ollama repo by using the command `docker exec -it <ollama_image_name_in_multicontainer_app> ollama run <model_name>`
6. Once pulled, access the app here : `http://localhost:8501/`


## Nvidia GPU container toolkit
Follow instructions : [Nvidia gpu container toolkit ](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
 
## Contributor
[Abhisek](https://www.linkedin.com/in/abhisekghoshml/)
<br>
[Amandeep](https://www.linkedin.com/in/amandeepsinghkhanna/)
