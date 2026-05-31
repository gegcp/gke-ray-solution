
. /home/gech/demo-magic/demo-magic.sh

clear

pe 'k1 get pod'
pe 'k2 get pod'
pe "k3 get pod"
pe "k1 get serviceexport"
pe "k2 get serviceexport"
pe "k3 get serviceexport"
pe "k1 get serviceimport"
pe "k1 get gatewayclass"
pe "k1 get gateway"


echo "Listing models in the vllm service with the gateway IP"
kubectl run gw-test-$(date +%s) --image=curlimages/curl:latest --rm -i --restart=Never --context=gke_gpu-launchpad-playground_us-east5_gech-gke-us -- curl -s -H "Host: gemma.internal" http://10.128.1.13/v1/models|jq
pe "k1 port-forward svc/chatbot-app 8088:80"


pe "k1 logs load-gen-1 -f"
pe "k1 apply -f gemma-http-route-weighted-us.yaml"


echo "Get the list of backend service"
gcloud compute backend-services list \
  --filter="id=(2057316009562711913 OR 2954457170717288292 OR 4980019642215278445 OR 5967321036020361069)" \
  --format="table(id,name,region,loadBalancingScheme)"

pe "k1 get pod |grep tpu"
pe "k1 describe deploy vllm-gemma-2b-it-tpu" 
pe "k1 get computeclass tpu-class -o yaml"

echo "The list of TPU nodes"
k1 get nodes --show-labels|grep provision|grep tpu