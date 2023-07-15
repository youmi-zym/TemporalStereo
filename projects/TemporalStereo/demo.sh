CONFIG=./configs/sceneflow.yaml
CKPT=./exps/TemporalStereo/sceneflow/epoch_best.ckpt
LOGDIR=./exps/TemporalStereo/sceneflow/output/
DATA_ROOT=./datasets/SceneFlow/Flyingthings3D
DATA_TYPE=SceneFlow
ANNFILE=./splits/flyingthings3d/test.json
H=544
W=960
DEVICE=cuda:0

echo Starting running demo...

python demo.py  --config-file $CONFIG \
                --checkpoint-path $CKPT \
                --resize-to-shape $H $W \
                --data-type $DATA_TYPE \
                --data-root  $DATA_ROOT\
                --annfile $ANNFILE \
                --device $DEVICE \
                --log-dir $LOGDIR

echo Results are saved to $LOGDIR.
echo done!