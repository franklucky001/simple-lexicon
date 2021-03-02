IMAGE_ID=`docker ps -a | grep simple-lexicon | awk '{print $1}'`
if [ -n "$IMAGE_ID" ];then
    echo "remove cache container"
    docker rm -f $IMAGE_ID
fi
MODEL_NAME="lstm-crf"
DATA_NAME="rmrb"
USE_LEXICON="False"
WORK_DIR=/root/simple-lexicon
nvidia-docker run  \
	--privileged=true \
	--name simple-lexicon \
	-v $PWD/data:$WORK_DIR/data \
	-v $PWD/models:$WORK_DIR/models \
	-v $PWD/summary:$WORK_DIR/summary \
	-v $PWD/logs:$WORK_DIR/logs \
	-e MODEL_NAME=$MODEL_NAME \
        -e DATA_NAME=$DATA_NAME \
	-e USE_LEXICON=$USE_LEXICON \
	-t simple-lexicon:0.1

