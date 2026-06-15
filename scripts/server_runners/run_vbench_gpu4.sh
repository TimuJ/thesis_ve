#!/bin/bash
bash /data/disk2/timur/run_vbench_dim.sh color /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 4
bash /data/disk2/timur/run_vbench_dim.sh object_class /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 4
bash /data/disk2/timur/run_vbench_dim.sh multiple_objects /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 4
bash /data/disk2/timur/run_vbench_dim.sh color /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 4
bash /data/disk2/timur/run_vbench_dim.sh object_class /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 4
bash /data/disk2/timur/run_vbench_dim.sh multiple_objects /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 4
echo GPU4_DONE
