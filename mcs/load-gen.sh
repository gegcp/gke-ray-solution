#!/bin/bash

# Multi-cluster Gateway Load Generator Management Script

CONTEXT="gke_gpu-launchpad-playground_us-east5_gech-gke-us"
GATEWAY_IP="10.128.1.13"
NUM_WORKERS=5

start() {
    echo "Starting $NUM_WORKERS load generator pods..."

    for i in $(seq 1 $NUM_WORKERS); do
        echo "Starting load-gen-$i..."
        kubectl run load-gen-$i --image=curlimages/curl:latest --restart=Never --context=$CONTEXT -- sh -c "
COUNTER=1
while true; do
  echo \"[Worker $i] Request \$COUNTER - \$(date +%H:%M:%S)\"
  curl -s -X POST -H 'Host: gemma.internal' -H 'Content-Type: application/json' \
    -d '{\"model\":\"google/gemma-2b-it\",\"messages\":[{\"role\":\"user\",\"content\":\"hello from worker $i\"}],\"max_tokens\":20}' \
    http://$GATEWAY_IP/v1/chat/completions -w '\nHTTP Status: %{http_code}\n' --max-time 5 || echo 'Request failed'
  echo ''
  COUNTER=\$((COUNTER + 1))
  sleep 1
done
" 2>&1 | grep -v "Warning:" &
    done

    wait
    echo ""
    echo "Load generators started. Check status with: ./load-gen.sh status"
}

stop() {
    echo "Stopping load generator pods..."

    for i in $(seq 1 $NUM_WORKERS); do
        echo "Stopping load-gen-$i..."
        kubectl delete pod load-gen-$i --context=$CONTEXT --wait=false 2>/dev/null
    done

    echo ""
    echo "Load generators stopped."
}

status() {
    echo "Load Generator Status:"
    echo ""
    kubectl get pods --context=$CONTEXT | grep "load-gen" || echo "No load generator pods running"
}

logs() {
    WORKER=${1:-1}
    echo "=== Logs from load-gen-$WORKER (last 20 lines) ==="
    kubectl logs load-gen-$WORKER --tail=20 --context=$CONTEXT 2>/dev/null || echo "Pod load-gen-$WORKER not found"
}

all_logs() {
    for i in $(seq 1 $NUM_WORKERS); do
        echo "=== Worker $i (last 5 lines) ==="
        kubectl logs load-gen-$i --tail=5 --context=$CONTEXT 2>/dev/null | grep -E "Request|Status" || echo "Pod load-gen-$i not found"
        echo ""
    done
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs $2
        ;;
    all-logs)
        all_logs
        ;;
    restart)
        stop
        sleep 3
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs [worker-num]|all-logs|restart}"
        echo ""
        echo "Commands:"
        echo "  start      - Start $NUM_WORKERS concurrent load generator pods"
        echo "  stop       - Stop all load generator pods"
        echo "  status     - Show status of load generator pods"
        echo "  logs [N]   - Show logs from worker N (default: 1)"
        echo "  all-logs   - Show recent logs from all workers"
        echo "  restart    - Stop and start load generators"
        echo ""
        echo "Examples:"
        echo "  ./load-gen.sh start"
        echo "  ./load-gen.sh status"
        echo "  ./load-gen.sh logs 3"
        echo "  ./load-gen.sh all-logs"
        echo "  ./load-gen.sh stop"
        exit 1
        ;;
esac
