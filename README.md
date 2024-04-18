# MLOps project using _MLflow_
End to End Machine Learning project management with MLFlow

### Model Training setup
* Download and Install Python
* Create your data, model, and reports directory
* Run the MLflow project using **mlflow run .**
* Generate the docker file for the ML model

### Model Deployment using docker, aws ecr, and kubernetes
* Install docker desktop
* Build docker Image
* Tag docker Image
* Configure your AWS CLI
* Login into your aws ecr
* Push docker Image to ecr
* Create Kubernetes namespace
* Create Kubernetes docker config secret
* Deploy Kserve inference script

### CI/CD

1. Configure AWS CLI and login to AWS ECR
```shell
# create an AWS IAM user and configure aws cli
aws configure

# get ecr login password and use it to login docker desktop into your ecr repo
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com
```
2. Build and push docker image
```shell
# Build Image
docker build .

# Tag Image
docker tag my-docker-image:latest <your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-docker-image:latest 

# Push Image
docker push <your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-docker-image:latest
```
3. Install eksctl and create eks cluster
```shell
eksctl create cluster --name mlflow --nodegroup-name ng1 --node-type m5.large --nodes 8 --region us-east-1
```
4. [Install Knative and Istio](https://knative.dev/docs/install/yaml-install/serving/install-serving-with-yaml/#install-a-networking-layer)
5. Create k8s namespace and set it as default
```shell
kubectl create namespace mlflow-serve
kubectl config set-context --current --namespace=mlflow-serve
```
6. Create k8s secret for eks cluster to access ecr
```shell
# get login password
$dockerPassword = aws ecr get-login-password

# create secret using password
kubectl create secret docker-registry ecrsecret `
  --docker-username=AWS `
  --docker-password=$dockerPassword `
  --docker-server=my-registry.example:5000
```
10. Deploy application
```shell
kubectl apply -f kube_deploy.yaml
```
11. Test deployment
```shell
$INGRESS_HOST = kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
$INGRESS_PORT = kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}'

# get service name
$SERVICE_HOSTNAME = kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3
curl -v -H "Host: ${SERVICE_HOSTNAME}" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
```