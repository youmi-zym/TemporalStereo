python demo.py  --config-file ./configs/sceneflow_full.yaml \
                --checkpoint-path /home/yzhang/exps/TemporalStereo/sceneflow/baseline/epoch_best.ckpt \
                --resize-to-shape 544 960 \
                --data-type SceneFlow \
                --data-root /mnt/DiskSSD/SceneFlow/ \
                --annfile ./splits/flyingthings3d/view_1_test.json \
                --device cuda:0 \
                --log-dir /home/yzhang/exps/TemporalStereo/sceneflow/baseline/output/
